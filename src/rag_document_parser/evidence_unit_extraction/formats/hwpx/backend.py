from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from typing import Any
from xml.etree import ElementTree as ET

from ....models import EvidenceUnit, PendingAsset, SourceEvidence
from ...backend import ParsedDocument
from ...ocr import OcrFn
from .diagram import HwpxDiagramBuilder
from .document_assembly import HwpxDocumentAssembler
from .package_reader import HwpxPackageReader
from .source_projection import HwpxTableSourceProjector
from .table_extraction import HwpxTableExtractor
from .xml_utils import paragraph_text
from .xml_utils import q


@dataclass
class HwpxBackend:
    supported_suffixes = (".hwpx",)
    ocr_fn: OcrFn | None = None

    def parse(self, data: bytes, suffix: str) -> ParsedDocument:
        units: list[EvidenceUnit] = []
        assets: list[PendingAsset] = []
        warnings: list[dict[str, Any]] = []
        block_index = 1
        table_index = 1
        package_reader = HwpxPackageReader()
        diagram_builder = HwpxDiagramBuilder()
        table_extractor = HwpxTableExtractor()
        source_projector = HwpxTableSourceProjector()
        assembler = HwpxDocumentAssembler(ocr_fn=self.ocr_fn)

        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            bin_data_map = package_reader.load_bin_data_map(archive)
            for section_name in package_reader.section_names(archive):
                root = ET.fromstring(archive.read(section_name))
                for paragraph in root.findall(q("p")):
                    table = paragraph.find(f".//{q('tbl')}")
                    if table is not None:
                        structured = table_extractor.extract(
                            table,
                            archive,
                            bin_data_map,
                            assets,
                            warnings,
                        )
                        if not structured["columns"] and not structured["rows"]:
                            continue
                        table_diagram = diagram_builder.table_flowchart(structured)
                        table_structured = (
                            diagram_builder.without_flowchart_rows(structured)
                            if table_diagram is not None
                            else structured
                        )
                        if table_structured["rows"] or table_diagram is None:
                            public_table = table_extractor.public(table_structured)
                            text_box = table_extractor.single_cell_text(public_table)
                            if text_box is not None:
                                units.append(
                                    assembler.text_unit(f"b{block_index}", text_box)
                                )
                                block_index += 1
                                continue
                            table_id = f"t{table_index}"
                            table_index += 1
                            units.append(
                                EvidenceUnit(
                                    id=f"b{block_index}",
                                    type="table",
                                    format="structured_table",
                                    source=SourceEvidence(
                                        kind="table",
                                        text=source_projector.project(public_table),
                                    ),
                                    content=public_table,
                                    metadata={
                                        "common": {
                                            "chunk_kind": "table",
                                            "section_path": [],
                                            "display_format": "structured_table",
                                        },
                                        "table": {
                                            "table_id": table_id,
                                            "headers": [
                                                str(column["text"])
                                                for column in public_table["columns"]
                                            ],
                                            "row_count": len(public_table["rows"]),
                                        },
                                    },
                                )
                            )
                            block_index += 1
                        if table_diagram is not None:
                            units.append(
                                assembler.diagram_unit(
                                    f"b{block_index}",
                                    table_diagram,
                                )
                            )
                            block_index += 1
                        continue

                    picture = paragraph.find(f".//{q('pic')}")
                    if picture is not None:
                        image = package_reader.extract_image(
                            picture,
                            archive,
                            bin_data_map,
                            len(assets) + 1,
                            warnings,
                        )
                        if image is None:
                            continue
                        asset_id, asset = image
                        assets.append(asset)
                        units.append(
                            EvidenceUnit(
                                id=f"b{block_index}",
                                type="image",
                                format="asset_ref",
                                source=SourceEvidence(
                                    kind="image",
                                    text=f"image: {asset_id}",
                                ),
                                content={"asset_id": asset_id, "caption": None},
                                metadata={
                                    "common": {
                                        "chunk_kind": "image",
                                        "section_path": [],
                                        "display_format": "image",
                                    },
                                    "asset": {"asset_id": asset_id},
                                },
                            )
                        )
                        block_index += 1
                        continue

                    drawing = diagram_builder.parse_paragraph(paragraph, warnings)
                    if drawing is not None:
                        if drawing.single_text is not None:
                            units.append(
                                assembler.text_unit(
                                    f"b{block_index}",
                                    drawing.single_text,
                                )
                            )
                            block_index += 1
                            continue
                        if drawing.structured is not None:
                            units.append(
                                assembler.diagram_unit(
                                    f"b{block_index}",
                                    drawing.structured,
                                )
                            )
                            block_index += 1
                            continue

                    text = paragraph_text(paragraph).strip()
                    if not text:
                        continue
                    units.append(assembler.text_unit(f"b{block_index}", text))
                    block_index += 1

        assembler.append_ocr_units(units, assets, warnings, block_index)
        return ParsedDocument(units=units, assets=assets, quality_warnings=warnings)
