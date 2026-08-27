from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ....models import EvidenceUnit, SourceEvidence
from ...backend import ParsedDocument
from ...ocr import OcrFn, OcrResult, coerce_ocr_result
from .blocks import DiagramBlock as _DiagramBlock
from .blocks import DrawingLineBlock as _DrawingLineBlock
from .blocks import ImageBlock as _ImageBlock
from .blocks import TableBlock as _TableBlock
from .blocks import TextBlock as _TextBlock
from .diagram import Hwp5DiagramBuilder
from .parsed import ParsedBlocks as _ParsedBlocks
from .source_projection import Hwp5TableSourceProjector
from .table_extraction import Hwp5TableBuilder
from .text import clean_text as _clean_text

_HWP5_DIAGRAM_BUILDER = Hwp5DiagramBuilder()
_HWP5_SOURCE_PROJECTOR = Hwp5TableSourceProjector()
_HWP5_TABLE_BUILDER = Hwp5TableBuilder()
_coalesce_drawing_text_blocks = _HWP5_DIAGRAM_BUILDER.coalesce
_diagram_source_text = _HWP5_DIAGRAM_BUILDER.source_text
_structured_diagram = _HWP5_DIAGRAM_BUILDER.build
_structured_table = _HWP5_TABLE_BUILDER.build
_table_has_content = _HWP5_TABLE_BUILDER.has_content
_table_source_text = _HWP5_SOURCE_PROJECTOR.project
_unresolved_connector_details = _HWP5_DIAGRAM_BUILDER.unresolved_connectors


@dataclass(frozen=True)
class Hwp5DocumentAssembler:
    """Assembles parsed HWP5 blocks and optional OCR into a document."""

    ocr_fn: OcrFn | None = None

    def assemble(self, parsed: _ParsedBlocks) -> ParsedDocument:
        return _apply_ocr_fallback(_to_document(parsed), self.ocr_fn)


def _to_document(parsed: _ParsedBlocks) -> ParsedDocument:
    units: list[EvidenceUnit] = []
    warnings: list[dict[str, Any]] = list(parsed.quality_warnings)
    block_index = 1
    table_index = 1
    saw_structured_diagram = False

    for block in _coalesce_drawing_text_blocks(parsed.blocks):
        if isinstance(block, _TextBlock):
            text = _clean_text(block.text)
            if not text:
                continue
            chunk_kind = "drawing" if block.origin == "drawing" else "text"
            display_format = "drawing_text" if block.origin == "drawing" else "plain"
            units.append(
                EvidenceUnit(
                    id=f"b{block_index}",
                    type="text",
                    format="plain",
                    source=SourceEvidence(kind="text", text=text),
                    content=text,
                    metadata={
                        "common": {
                            "chunk_kind": chunk_kind,
                            "section_path": [],
                            "display_format": display_format,
                        }
                    },
                )
            )
            block_index += 1
            continue

        if isinstance(block, _DiagramBlock):
            structured = _structured_diagram(
                block.text,
                nodes=block.nodes,
                bboxes=block.bboxes,
                connectors=block.connectors,
            )
            source_text = _diagram_source_text(structured)
            if not source_text:
                continue
            saw_structured_diagram = True
            unresolved_connectors = _unresolved_connector_details(structured)
            units.append(
                EvidenceUnit(
                    id=f"b{block_index}",
                    type="diagram",
                    format="structured_diagram",
                    source=SourceEvidence(kind="diagram", text=source_text),
                    content=structured,
                    metadata={
                        "common": {
                            "chunk_kind": "diagram",
                            "section_path": [],
                            "display_format": "structured_diagram",
                        },
                        "diagram": {
                            "node_count": len(structured["nodes"]),
                            "connector_count": len(structured["connectors"]),
                            "edge_count": len(structured["edges"]),
                        },
                    },
                )
            )
            if unresolved_connectors:
                warnings.append(
                    {
                        "type": "hwp5_diagram_connector_unresolved",
                        "severity": "medium",
                        "unit_id": f"b{block_index}",
                        "connector_ids": [
                            str(connector.get("id", ""))
                            for connector in unresolved_connectors
                            if connector.get("id")
                        ],
                        "connectors": unresolved_connectors,
                        "message": (
                            "HWP5 drawing connector(s) were preserved, but could not "
                            "be mapped to diagram edges."
                        ),
                    }
                )
            block_index += 1
            continue

        if isinstance(block, _DrawingLineBlock):
            continue

        if isinstance(block, _ImageBlock):
            asset_metadata = {"asset_id": block.asset_id, **dict(block.metadata)}
            units.append(
                EvidenceUnit(
                    id=f"b{block_index}",
                    type="image",
                    format="asset_ref",
                    source=SourceEvidence(kind="image", text=f"image: {block.asset_id}"),
                    content={"asset_id": block.asset_id, "caption": None},
                    metadata={
                        "common": {
                            "chunk_kind": "image",
                            "section_path": [],
                            "display_format": "image",
                        },
                        "asset": asset_metadata,
                    },
                )
            )
            block_index += 1
            continue

        structured = _structured_table(
            block.rows,
            row_count=block.row_count,
            column_count=block.column_count,
        )
        if not _table_has_content(block.rows):
            continue
        table_id = f"t{table_index}"
        units.append(
            EvidenceUnit(
                id=f"b{block_index}",
                type="table",
                format="structured_table",
                source=SourceEvidence(kind="table", text=_table_source_text(structured)),
                content=structured,
                metadata={
                    "common": {
                        "chunk_kind": "table",
                        "section_path": [],
                        "display_format": "structured_table",
                    },
                    "table": {
                        "table_id": table_id,
                        "headers": [
                            str(column["text"]) for column in structured["columns"]
                        ],
                        "row_count": len(structured["rows"]),
                    },
                },
            )
        )
        block_index += 1
        table_index += 1

    if saw_structured_diagram:
        warnings.append(
            {
                "type": "hwp5_drawing_structure_partial",
                "severity": "medium",
                "message": (
                    "HWP5 drawing object text and some geometry were extracted as "
                    "structured diagram evidence, but unsupported or undocumented "
                    "drawing details may still be incomplete."
                ),
            }
        )
    if parsed.missing_image_count:
        warnings.append(
            {
                "type": "hwp5_images_missing",
                "severity": "medium",
                "message": (
                    f"{parsed.missing_image_count} HWP5 image reference(s) could not "
                    "be resolved from BinData streams."
                ),
            }
        )

    return ParsedDocument(
        units=units,
        assets=parsed.assets,
        quality_warnings=warnings,
    )


def _apply_ocr_fallback(
    document: ParsedDocument,
    ocr_fn: OcrFn | None,
) -> ParsedDocument:
    if ocr_fn is None or not document.assets:
        return document

    units = list(document.units)
    warnings = list(document.quality_warnings)
    native_texts = _native_source_texts(document.units)
    next_index = len(units) + 1
    for image_index, asset in enumerate(document.assets):
        if asset.kind != "image":
            continue
        try:
            raw_result = ocr_fn(asset.data, image_index)
            result = coerce_ocr_result(raw_result)
        except Exception as exc:
            warnings.append(_hwp5_ocr_failed_warning(asset.id, str(exc)))
            continue
        if result.status == "no_text":
            warning = _hwp5_ocr_empty_warning(asset.id)
            if isinstance(raw_result, OcrResult):
                warning["reason"] = result.reason
            warnings.append(warning)
            continue
        if result.status == "uncertain":
            warnings.append(_hwp5_ocr_uncertain_warning(asset.id, result.reason))
            continue
        text = _clean_text(result.text)
        if not text:
            warnings.append(_hwp5_ocr_empty_warning(asset.id))
            continue
        if _ocr_text_duplicates_native_text(text, native_texts):
            continue
        units.append(
            EvidenceUnit(
                id=f"b{next_index}",
                type="text",
                format="plain",
                source=SourceEvidence(kind="text", text=text),
                content=text,
                metadata={
                    "common": {
                        "chunk_kind": "ocr",
                        "section_path": [],
                        "display_format": "plain",
                    },
                    "ocr": {
                        "source": "hwp5_image",
                        "asset_id": asset.id,
                    },
                },
            )
        )
        next_index += 1
    return ParsedDocument(
        units=units,
        assets=document.assets,
        quality_warnings=warnings,
    )


def _native_source_texts(units: list[EvidenceUnit]) -> list[str]:
    return [
        _clean_text(unit.source.text)
        for unit in units
        if unit.type != "image" and unit.source.text.strip()
    ]


def _ocr_text_duplicates_native_text(text: str, native_texts: list[str]) -> bool:
    compact_text = _compact_text_for_ocr_dedupe(text)
    if not compact_text:
        return False
    return any(
        compact_text == compact_native or compact_text in compact_native
        for native_text in native_texts
        if (compact_native := _compact_text_for_ocr_dedupe(native_text))
    )


def _compact_text_for_ocr_dedupe(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _hwp5_ocr_failed_warning(asset_id: str, message: str) -> dict[str, Any]:
    return {
        "type": "hwp5_ocr_failed",
        "severity": "medium",
        "asset_id": asset_id,
        "stage": "ocr",
        "message": message,
    }


def _hwp5_ocr_empty_warning(asset_id: str) -> dict[str, Any]:
    return {
        "type": "hwp5_ocr_empty",
        "severity": "low",
        "asset_id": asset_id,
        "stage": "ocr",
        "message": "empty OCR result",
    }


def _hwp5_ocr_uncertain_warning(
    asset_id: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "type": "hwp5_ocr_uncertain",
        "severity": "medium",
        "asset_id": asset_id,
        "stage": "ocr",
        "reason": reason,
        "message": "OCR could not determine whether the image contains readable text.",
    }
