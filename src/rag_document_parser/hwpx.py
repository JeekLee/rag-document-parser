from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from xml.etree import ElementTree as ET

from .backends import ParsedDocument
from .models import Evidence, EvidenceUnit, PendingAsset, SourceEvidence

_HP = "http://www.hancom.co.kr/hwpml/2011/paragraph"
_OPF = "http://www.idpf.org/2007/opf/"


def _q(local: str) -> str:
    return f"{{{_HP}}}{local}"


@dataclass
class HwpxBackend:
    def parse(self, data: bytes, suffix: str) -> ParsedDocument:
        units: list[EvidenceUnit] = []
        assets: list[PendingAsset] = []
        block_index = 1
        table_index = 1

        with zipfile.ZipFile(io.BytesIO(data)) as z:
            bin_data_map = _load_bin_data_map(z)
            for section_name in _section_names(z):
                root = ET.fromstring(z.read(section_name))
                for paragraph in root.findall(_q("p")):
                    table = paragraph.find(f".//{_q('tbl')}")
                    if table is not None:
                        table_id = f"t{table_index}"
                        table_index += 1
                        structured = _structured_table(table, z, bin_data_map, assets)
                        if not structured["columns"] and not structured["rows"]:
                            continue
                        units.append(
                            EvidenceUnit(
                                id=f"b{block_index}",
                                type="table",
                                source=SourceEvidence(
                                    kind="table",
                                    text=_table_source_text(structured),
                                ),
                                evidence=Evidence(
                                    kind="table",
                                    format="structured_table",
                                    content=structured,
                                ),
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
                                            for column in structured["columns"]
                                        ],
                                        "row_count": len(structured["rows"]),
                                    },
                                },
                            )
                        )
                        block_index += 1
                        continue

                    picture = paragraph.find(f".//{_q('pic')}")
                    if picture is not None:
                        image = _extract_image(picture, z, bin_data_map, len(assets) + 1)
                        if image is None:
                            continue
                        asset_id, asset = image
                        assets.append(asset)
                        units.append(
                            EvidenceUnit(
                                id=f"b{block_index}",
                                type="image",
                                source=SourceEvidence(
                                    kind="image",
                                    text=f"image: {asset_id}",
                                ),
                                evidence=Evidence(
                                    kind="image",
                                    format="asset_ref",
                                    content={"asset_id": asset_id, "caption": None},
                                ),
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

                    text = _paragraph_text(paragraph).strip()
                    if not text:
                        continue
                    units.append(
                        EvidenceUnit(
                            id=f"b{block_index}",
                            type="text",
                            source=SourceEvidence(kind="text", text=text),
                            evidence=Evidence(kind="text", format="plain", content=text),
                            metadata={
                                "common": {
                                    "chunk_kind": "text",
                                    "section_path": [],
                                    "display_format": "plain",
                                }
                            },
                        )
                    )
                    block_index += 1

        return ParsedDocument(units=units, assets=assets)


def _section_names(z: zipfile.ZipFile) -> list[str]:
    names = [name for name in z.namelist() if re.match(r"Contents/section\d+\.xml$", name)]
    return sorted(
        names,
        key=lambda name: int(re.search(r"\d+", name.rsplit("/", 1)[-1]).group()),
    )


def _load_bin_data_map(z: zipfile.ZipFile) -> dict[str, str]:
    if "Contents/content.hpf" not in z.namelist():
        return {}
    root = ET.fromstring(z.read("Contents/content.hpf"))
    result: dict[str, str] = {}
    for item in root.iter(f"{{{_OPF}}}item"):
        item_id = item.get("id", "")
        href = item.get("href", "")
        if item_id and href.startswith("BinData/"):
            result[item_id] = href
    return result


def _paragraph_text(paragraph: ET.Element) -> str:
    parts: list[str] = []
    for run in paragraph.findall(_q("run")):
        if run.find(_q("tbl")) is not None or run.find(_q("pic")) is not None:
            continue
        parts.append(_run_text(run))
    return "".join(parts)


def _run_text(run: ET.Element) -> str:
    parts: list[str] = []
    for text in run.findall(_q("t")):
        if text.text:
            parts.append("".join(char for char in text.text if char > "\x1f"))
    return "".join(parts)


def _structured_table(
    table: ET.Element,
    z: zipfile.ZipFile,
    bin_data_map: dict[str, str],
    assets: list[PendingAsset],
) -> dict[str, object]:
    raw_rows = [
        _table_row(row, z, bin_data_map, assets)
        for row in table.findall(_q("tr"))
    ]
    if not raw_rows:
        return {"caption": None, "columns": [], "rows": []}

    first_row_is_header = not any(cell["children"] for cell in raw_rows[0])
    header_row = raw_rows[0] if first_row_is_header else []
    data_rows = raw_rows[1:] if first_row_is_header else raw_rows
    column_count = max((len(row) for row in raw_rows), default=0)
    headers = [
        str(header_row[index]["text"] or f"Column {index + 1}")
        if index < len(header_row)
        else f"Column {index + 1}"
        for index in range(column_count)
    ]
    columns = [
        {
            "id": f"c{index}",
            "text": header,
        }
        for index, header in enumerate(headers, start=1)
    ]

    rows: list[dict[str, object]] = []
    for row_index, raw_cells in enumerate(data_rows, start=1):
        cells: list[dict[str, object]] = []
        for column, raw_cell in zip(columns, raw_cells, strict=False):
            cells.append(
                {
                    "column_id": column["id"],
                    "text": raw_cell["text"],
                    "rowspan": raw_cell["rowspan"],
                    "colspan": raw_cell["colspan"],
                    "children": raw_cell["children"],
                }
            )
        rows.append({"index": row_index, "cells": cells})

    return {"caption": None, "columns": columns, "rows": rows}


def _table_row(
    row: ET.Element,
    z: zipfile.ZipFile,
    bin_data_map: dict[str, str],
    assets: list[PendingAsset],
) -> list[dict[str, object]]:
    return [
        _table_cell(cell, z, bin_data_map, assets)
        for cell in row.findall(_q("tc"))
    ]


def _table_cell(
    cell: ET.Element,
    z: zipfile.ZipFile,
    bin_data_map: dict[str, str],
    assets: list[PendingAsset],
) -> dict[str, object]:
    sub_list = cell.find(_q("subList"))
    texts: list[str] = []
    children: list[dict[str, object]] = []
    if sub_list is not None:
        for paragraph in sub_list.findall(_q("p")):
            nested = paragraph.find(f".//{_q('tbl')}")
            if nested is not None:
                children.append(
                    {
                        "kind": "table",
                        "format": "structured_table",
                        "content": _structured_table(nested, z, bin_data_map, assets),
                    }
                )
                continue
            for picture in paragraph.findall(f".//{_q('pic')}"):
                image = _extract_image(picture, z, bin_data_map, len(assets) + 1)
                if image is None:
                    continue
                asset_id, asset = image
                assets.append(asset)
                children.append(
                    {
                        "kind": "image",
                        "format": "asset_ref",
                        "content": {"asset_id": asset_id, "caption": None},
                    }
                )
            text = _paragraph_text(paragraph).strip()
            if text:
                texts.append(text)
    return {
        "text": " ".join(texts),
        "rowspan": _cell_span(cell, "rowSpan"),
        "colspan": _cell_span(cell, "colSpan"),
        "children": children,
    }


def _cell_span(cell: ET.Element, name: str) -> int:
    value = cell.get(name)
    if value is None:
        span = cell.find(_q("cellSpan"))
        value = span.get(name) if span is not None else None
    try:
        return max(1, int(value)) if value is not None else 1
    except ValueError:
        return 1


def _table_source_text(table: dict[str, object]) -> str:
    columns = table["columns"]
    rows = table["rows"]
    column_text = {str(column["id"]): str(column["text"]) for column in columns}
    lines: list[str] = []
    if columns:
        lines.append("columns: " + " | ".join(str(column["text"]) for column in columns))
    for row in rows:
        cells: list[str] = []
        for cell in row["cells"]:
            header = column_text.get(str(cell["column_id"]), str(cell["column_id"]))
            value = str(cell["text"])
            child_texts = [
                "nested table: " + _inline_table_source(child["content"])
                for child in cell["children"]
                if child.get("kind") == "table"
            ]
            combined = "; ".join(part for part in [value, *child_texts] if part)
            if combined:
                cells.append(f"{header}={combined}")
        if cells:
            lines.append(f"row {row['index']}: " + "; ".join(cells))
    return "\n".join(lines)


def _inline_table_source(table: dict[str, object]) -> str:
    source = _table_source_text(table)
    return source.replace("\n", " / ")


def _extract_image(
    picture: ET.Element,
    z: zipfile.ZipFile,
    bin_data_map: dict[str, str],
    index: int,
) -> tuple[str, PendingAsset] | None:
    ref = _image_ref(picture)
    if ref is None:
        return None
    href = bin_data_map.get(ref, "")
    if not href or href not in z.namelist():
        return None
    data = z.read(href)
    mime = _detect_mime(data)
    ext = _mime_to_ext(mime)
    asset_id = f"img-{index:04d}"
    return (
        asset_id,
        PendingAsset(
            id=asset_id,
            kind="image",
            data=data,
            mime=mime,
            ext=ext,
            metadata={"source_path": href},
        ),
    )


def _image_ref(picture: ET.Element) -> str | None:
    for element in picture.iter():
        if _local_name(element.tag) not in {"img", "image"}:
            continue
        ref = element.get("binaryItemIDRef")
        if ref:
            return ref
    return None


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _detect_mime(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] in (b"GIF8", b"GIF9"):
        return "image/gif"
    if data[:2] == b"BM":
        return "image/bmp"
    return "application/octet-stream"


def _mime_to_ext(mime: str) -> str:
    return {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/gif": "gif",
        "image/bmp": "bmp",
    }.get(mime, "bin")
