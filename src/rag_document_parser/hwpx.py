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
                        structured = _structured_table(table, z, bin_data_map, assets)
                        if not structured["columns"] and not structured["rows"]:
                            continue
                        text_box = _single_cell_text_table_text(structured)
                        if text_box is not None:
                            units.append(
                                EvidenceUnit(
                                    id=f"b{block_index}",
                                    type="text",
                                    source=SourceEvidence(kind="text", text=text_box),
                                    evidence=Evidence(
                                        kind="text",
                                        format="plain",
                                        content=text_box,
                                    ),
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
                            continue
                        table_id = f"t{table_index}"
                        table_index += 1
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
        _table_row(row, row_index, z, bin_data_map, assets)
        for row_index, row in enumerate(table.findall(_q("tr")))
    ]
    raw_rows = [row for row in raw_rows if row]
    if not raw_rows:
        return {"caption": None, "columns": [], "rows": []}

    column_count = _table_column_count(raw_rows)
    header_count = _header_row_count(raw_rows)
    header_raw_rows = raw_rows[:header_count]
    data_raw_rows = raw_rows[header_count:]
    columns = _table_columns(column_count, header_raw_rows)
    header_rows = [
        {
            "index": index,
            "cells": _evidence_cells(raw_cells, columns),
        }
        for index, raw_cells in enumerate(header_raw_rows, start=1)
    ]

    rows: list[dict[str, object]] = []
    for raw_cells in data_raw_rows:
        if _row_is_blank(raw_cells):
            continue
        rows.append(
            {
                "index": len(rows) + 1,
                "cells": _evidence_cells(raw_cells, columns),
            }
        )

    result: dict[str, object] = {"caption": None, "columns": columns, "rows": rows}
    if header_rows:
        result["header_rows"] = header_rows
    return result


def _table_column_count(raw_rows: list[list[dict[str, object]]]) -> int:
    return max(
        (
            int(cell["col_addr"]) + int(cell["colspan"])
            for row in raw_rows
            for cell in row
        ),
        default=0,
    )


def _table_columns(
    column_count: int,
    header_rows: list[list[dict[str, object]]],
) -> list[dict[str, str]]:
    return [
        {
            "id": f"c{index}",
            "text": _column_header_text(header_rows, index - 1),
        }
        for index in range(1, column_count + 1)
    ]


def _column_header_text(
    header_rows: list[list[dict[str, object]]],
    column_index: int,
) -> str:
    texts: list[str] = []
    last_header_row = len(header_rows) - 1
    for row_index, row in enumerate(header_rows):
        for cell in row:
            if not _header_cell_contributes_to_column(
                cell,
                column_index,
                row_index,
                last_header_row,
            ):
                continue
            text = str(cell["text"]).strip()
            if text and text not in texts:
                texts.append(text)
    return " / ".join(texts)


def _header_cell_contributes_to_column(
    cell: dict[str, object],
    column_index: int,
    row_index: int,
    last_header_row: int,
) -> bool:
    start = int(cell["col_addr"])
    end = start + int(cell["colspan"])
    return column_index == start or (
        start < column_index < end and row_index < last_header_row
    )


def _header_row_count(raw_rows: list[list[dict[str, object]]]) -> int:
    first_row = raw_rows[0]
    if any(cell["children"] for cell in first_row):
        return 0
    if len(raw_rows) == 1:
        return 1
    count = 1
    header_row_end = _row_span_end(first_row)
    while count < len(raw_rows):
        row = raw_rows[count]
        row_start = _row_start(row)
        if (
            row_start < header_row_end
            or _row_is_blank(row)
            or _row_refines_previous_header(row, raw_rows[count - 1])
        ):
            count += 1
            header_row_end = max(header_row_end, _row_span_end(row))
            continue
        break
    return count


def _row_refines_previous_header(
    row: list[dict[str, object]],
    previous_row: list[dict[str, object]],
) -> bool:
    if any(cell["children"] for cell in row):
        return False
    groups = [
        cell
        for cell in previous_row
        if int(cell["colspan"]) > 1 and str(cell["text"]).strip()
    ]
    if not groups:
        return False
    for group in groups:
        group_start = int(group["col_addr"])
        group_end = group_start + int(group["colspan"])
        refiners = [
            cell
            for cell in row
            if group_start <= int(cell["col_addr"])
            and int(cell["col_addr"]) + int(cell["colspan"]) <= group_end
        ]
        if not any(str(cell["text"]).strip() for cell in refiners):
            return False
    return True


def _row_start(row: list[dict[str, object]]) -> int:
    return min((int(cell["row_addr"]) for cell in row), default=0)


def _row_span_end(row: list[dict[str, object]]) -> int:
    return max(
        (
            int(cell["row_addr"]) + int(cell["rowspan"])
            for cell in row
        ),
        default=0,
    )


def _row_is_blank(row: list[dict[str, object]]) -> bool:
    return not any(str(cell["text"]).strip() or cell["children"] for cell in row)


def _evidence_cells(
    raw_cells: list[dict[str, object]],
    columns: list[dict[str, str]],
) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    for raw_cell in sorted(raw_cells, key=lambda cell: int(cell["col_addr"])):
        column_index = int(raw_cell["col_addr"])
        column_id = (
            columns[column_index]["id"]
            if 0 <= column_index < len(columns)
            else f"c{column_index + 1}"
        )
        cells.append(
            {
                "column_id": column_id,
                "text": raw_cell["text"],
                "rowspan": raw_cell["rowspan"],
                "colspan": raw_cell["colspan"],
                "children": raw_cell["children"],
            }
        )
    return cells


def _single_cell_text_table_text(table: dict[str, object]) -> str | None:
    if table["rows"]:
        return None
    header_rows = table.get("header_rows")
    if not isinstance(header_rows, list) or len(header_rows) != 1:
        return None
    cells = header_rows[0].get("cells")
    if not isinstance(cells, list) or len(cells) != 1:
        return None
    cell = cells[0]
    if not isinstance(cell, dict) or cell.get("children"):
        return None
    text = str(cell.get("text", "")).strip()
    return text or None


def _table_row(
    row: ET.Element,
    row_index: int,
    z: zipfile.ZipFile,
    bin_data_map: dict[str, str],
    assets: list[PendingAsset],
) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    col_cursor = 0
    for cell in row.findall(_q("tc")):
        raw_cell = _table_cell(cell, row_index, col_cursor, z, bin_data_map, assets)
        cells.append(raw_cell)
        col_cursor = int(raw_cell["col_addr"]) + int(raw_cell["colspan"])
    return cells


def _table_cell(
    cell: ET.Element,
    row_index: int,
    col_index: int,
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
        "row_addr": _cell_addr(cell, "rowAddr", row_index),
        "col_addr": _cell_addr(cell, "colAddr", col_index),
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


def _cell_addr(cell: ET.Element, name: str, default: int) -> int:
    value = cell.get(name)
    if value is None:
        addr = cell.find(_q("cellAddr"))
        value = addr.get(name) if addr is not None else None
    try:
        return max(0, int(value)) if value is not None else default
    except ValueError:
        return default


def _table_source_text(table: dict[str, object]) -> str:
    columns = table["columns"]
    rows = table["rows"]
    column_text = {
        str(column["id"]): _column_source_label(column)
        for column in columns
    }
    lines: list[str] = []
    if columns:
        lines.append(f"table: {len(columns)} columns")
    for header_row in table.get("header_rows", []):
        cells = _table_source_cells(
            header_row["cells"],
            column_text,
            use_header_labels=False,
        )
        if cells:
            lines.append(f"header {header_row['index']}: " + "; ".join(cells))
    for row in rows:
        cells = _table_source_cells(
            row["cells"],
            column_text,
            use_header_labels=True,
        )
        if cells:
            lines.append(f"row {row['index']}: " + "; ".join(cells))
    return "\n".join(lines)


def _column_source_label(column: dict[str, object]) -> str:
    text = str(column["text"]).strip()
    return text or _column_coordinate_label(str(column["id"]))


def _table_source_cells(
    cells: list[dict[str, object]],
    column_text: dict[str, str],
    *,
    use_header_labels: bool,
) -> list[str]:
    result: list[str] = []
    for cell in cells:
        header = _cell_source_label(
            cell,
            column_text,
            use_header_labels=use_header_labels,
        )
        value = str(cell["text"])
        child_texts = [
            "nested table: " + _inline_table_source(child["content"])
            for child in cell["children"]
            if child.get("kind") == "table"
        ]
        combined = "; ".join(part for part in [value, *child_texts] if part)
        if combined:
            result.append(f"{header}: {combined}")
    return result


def _cell_source_label(
    cell: dict[str, object],
    column_text: dict[str, str],
    *,
    use_header_labels: bool,
) -> str:
    column_id = str(cell["column_id"])
    colspan = int(cell.get("colspan", 1))
    if use_header_labels:
        labels = [
            column_text.get(f"c{column_index}", f"col {column_index}")
            for column_index in range(
                _column_id_number(column_id),
                _column_id_number(column_id) + colspan,
            )
        ]
        common_prefix = _common_semantic_header_prefix(labels)
        if common_prefix is not None:
            return common_prefix
        if len(set(labels)) == 1 and _is_semantic_column_label(labels[0]):
            return labels[0]
        if colspan == 1 and _is_semantic_column_label(labels[0]):
            return labels[0]
    return _cell_coordinate_label(column_id, colspan)


def _is_semantic_column_label(label: str) -> bool:
    return bool(label) and not label.startswith("col ")


def _common_semantic_header_prefix(labels: list[str]) -> str | None:
    if not labels or not all(_is_semantic_column_label(label) for label in labels):
        return None
    split_labels = [label.split(" / ") for label in labels]
    prefix: list[str] = []
    for parts in zip(*split_labels):
        if len(set(parts)) != 1:
            break
        prefix.append(parts[0])
    if not prefix:
        return None
    return " / ".join(prefix)


def _cell_coordinate_label(column_id: str, colspan: int) -> str:
    start = _column_id_number(column_id)
    if colspan <= 1:
        return _column_coordinate_label(column_id)
    return f"cols {start}-{start + colspan - 1}"


def _column_coordinate_label(column_id: str) -> str:
    return f"col {_column_id_number(column_id)}"


def _column_id_number(column_id: str) -> int:
    try:
        return max(1, int(column_id.removeprefix("c")))
    except ValueError:
        return 1


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
