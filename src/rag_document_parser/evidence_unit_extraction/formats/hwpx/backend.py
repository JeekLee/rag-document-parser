from __future__ import annotations

import io
import re
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable
from xml.etree import ElementTree as ET

from ....models import EvidenceUnit, PendingAsset, SourceEvidence
from ...backend import ParsedDocument
from ...schema import (
    structured_diagram as _structured_diagram_content,
    structured_table as _structured_table_content,
)
from ...table_source import (
    build_column_source_labels as _build_column_source_labels,
    common_semantic_header_prefix as _common_semantic_header_prefix,
    is_semantic_column_label as _is_semantic_column_label,
)

_HP = "http://www.hancom.co.kr/hwpml/2011/paragraph"
_OPF = "http://www.idpf.org/2007/opf/"
_DRAWING_NODE_TAGS = {
    "arc",
    "container",
    "curve",
    "ellipse",
    "polygon",
    "rect",
    "roundRect",
    "shapeObject",
    "textBox",
    "textbox",
}
_DRAWING_CONNECTOR_TAGS = {"arc", "connectLine", "curve", "line", "polygon"}
_UNSUPPORTED_DRAWING_TAGS = {
    "button",
    "checkBtn",
    "comboBox",
    "edit",
    "equation",
    "listBox",
    "ole",
    "radio",
    "scrollBar",
    "textart",
    "video",
}
_DIAGRAM_STEP_LABEL_RE = re.compile(
    r"^(?:[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]|\d+[.)])"
)


def _q(local: str) -> str:
    return f"{{{_HP}}}{local}"


@dataclass(frozen=True)
class _DrawingResult:
    structured: dict[str, object] | None = None
    single_text: str | None = None


@dataclass
class HwpxBackend:
    supported_suffixes = (".hwpx",)
    ocr_fn: Callable[[bytes, int], str] | None = None

    def parse(self, data: bytes, suffix: str) -> ParsedDocument:
        units: list[EvidenceUnit] = []
        assets: list[PendingAsset] = []
        warnings: list[dict[str, Any]] = []
        block_index = 1
        table_index = 1

        with zipfile.ZipFile(io.BytesIO(data)) as z:
            bin_data_map = _load_bin_data_map(z)
            for section_name in _section_names(z):
                root = ET.fromstring(z.read(section_name))
                for paragraph in root.findall(_q("p")):
                    table = paragraph.find(f".//{_q('tbl')}")
                    if table is not None:
                        structured = _structured_table(
                            table,
                            z,
                            bin_data_map,
                            assets,
                            warnings,
                        )
                        if not structured["columns"] and not structured["rows"]:
                            continue
                        table_diagram = _table_flowchart_diagram(structured)
                        table_structured = (
                            _table_without_flowchart_rows(structured)
                            if table_diagram is not None
                            else structured
                        )
                        if table_structured["rows"] or table_diagram is None:
                            public_table = _public_structured_table(table_structured)
                            text_box = _single_cell_text_table_text(public_table)
                            if text_box is not None:
                                units.append(
                                    EvidenceUnit(
                                        id=f"b{block_index}",
                                        type="text",
                                        format="plain",
                                        source=SourceEvidence(kind="text", text=text_box),
                                        content=text_box,
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
                                    format="structured_table",
                                    source=SourceEvidence(
                                        kind="table",
                                        text=_table_source_text(public_table),
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
                            units.append(_diagram_unit(f"b{block_index}", table_diagram))
                            block_index += 1
                        continue

                    picture = paragraph.find(f".//{_q('pic')}")
                    if picture is not None:
                        image = _extract_image(
                            picture,
                            z,
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

                    drawing = _paragraph_drawing(paragraph, warnings)
                    if drawing is not None:
                        if drawing.single_text is not None:
                            units.append(
                                _text_unit(f"b{block_index}", drawing.single_text)
                            )
                            block_index += 1
                            continue
                        if drawing.structured is not None:
                            units.append(
                                _diagram_unit(f"b{block_index}", drawing.structured)
                            )
                            block_index += 1
                            continue

                    text = _paragraph_text(paragraph).strip()
                    if not text:
                        continue
                    units.append(_text_unit(f"b{block_index}", text))
                    block_index += 1

        block_index = _append_ocr_fallback_units(
            units,
            assets,
            warnings,
            self.ocr_fn,
            block_index,
        )
        return ParsedDocument(units=units, assets=assets, quality_warnings=warnings)


def _text_unit(unit_id: str, text: str) -> EvidenceUnit:
    return EvidenceUnit(
        id=unit_id,
        type="text",
        format="plain",
        source=SourceEvidence(kind="text", text=text),
        content=text,
        metadata={
            "common": {
                "chunk_kind": "text",
                "section_path": [],
                "display_format": "plain",
            }
        },
    )


def _diagram_unit(unit_id: str, structured: dict[str, object]) -> EvidenceUnit:
    return EvidenceUnit(
        id=unit_id,
        type="diagram",
        format="structured_diagram",
        source=SourceEvidence(kind="diagram", text=_diagram_source_text(structured)),
        content=structured,
        metadata={
            "common": {
                "chunk_kind": "diagram",
                "section_path": [],
                "display_format": "structured_diagram",
            },
            "diagram": {
                "node_count": len(structured["nodes"]),
                "edge_count": len(structured["edges"]),
            },
        },
    )


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
    warnings: list[dict[str, Any]],
) -> dict[str, object]:
    raw_rows = [
        _table_row(row, row_index, z, bin_data_map, assets, warnings)
        for row_index, row in enumerate(table.findall(_q("tr")))
    ]
    raw_rows = [row for row in raw_rows if row]
    if not raw_rows:
        return _structured_table_content(columns=[], rows=[])

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
        rows.append(
            {
                "index": len(rows) + 1,
                "cells": _evidence_cells(raw_cells, columns),
            }
        )

    return _structured_table_content(
        columns=columns,
        rows=rows,
        header_rows=header_rows if header_rows else None,
    )


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
        start < column_index < end
        and (row_index < last_header_row or last_header_row == 0)
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
    group_ranges = [
        (
            int(group["col_addr"]),
            int(group["col_addr"]) + int(group["colspan"]),
        )
        for group in groups
    ]
    for cell in row:
        if not str(cell["text"]).strip():
            continue
        cell_start = int(cell["col_addr"])
        cell_end = cell_start + int(cell["colspan"])
        if not any(
            group_start <= cell_start and cell_end <= group_end
            for group_start, group_end in group_ranges
        ):
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
                "row_addr": raw_cell["row_addr"],
                "col_addr": raw_cell["col_addr"],
                "rowspan": raw_cell["rowspan"],
                "colspan": raw_cell["colspan"],
                "children": raw_cell["children"],
            }
        )
    return cells


def _public_structured_table(table: dict[str, object]) -> dict[str, object]:
    return _without_table_grid_fields(table)


def _without_table_grid_fields(value: object) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _without_table_grid_fields(child)
            for key, child in value.items()
            if key not in {"row_addr", "col_addr"}
        }
    if isinstance(value, list):
        return [_without_table_grid_fields(item) for item in value]
    return value


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
    if not isinstance(cell, Mapping) or cell.get("children"):
        return None
    text = str(cell.get("text", "")).strip()
    return text or None


def _table_row(
    row: ET.Element,
    row_index: int,
    z: zipfile.ZipFile,
    bin_data_map: dict[str, str],
    assets: list[PendingAsset],
    warnings: list[dict[str, Any]],
) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    col_cursor = 0
    for cell in row.findall(_q("tc")):
        raw_cell = _table_cell(
            cell,
            row_index,
            col_cursor,
            z,
            bin_data_map,
            assets,
            warnings,
        )
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
    warnings: list[dict[str, Any]],
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
                        "type": "table",
                        "format": "structured_table",
                        "content": _structured_table(
                            nested,
                            z,
                            bin_data_map,
                            assets,
                            warnings,
                        ),
                    }
                )
                continue
            drawing = _paragraph_drawing(paragraph, warnings)
            if drawing is not None:
                if drawing.single_text is not None:
                    texts.append(drawing.single_text)
                    continue
                if drawing.structured is not None:
                    children.append(
                        {
                            "type": "diagram",
                            "format": "structured_diagram",
                            "content": drawing.structured,
                        }
                    )
                    continue
            for picture in paragraph.findall(f".//{_q('pic')}"):
                image = _extract_image(
                    picture,
                    z,
                    bin_data_map,
                    len(assets) + 1,
                    warnings,
                )
                if image is None:
                    continue
                asset_id, asset = image
                assets.append(asset)
                children.append(
                    {
                        "type": "image",
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
    column_text = _build_column_source_labels(columns, _column_source_label, rows)
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
        child_texts = _nested_child_source_texts(cell["children"])
        combined = "; ".join(part for part in [value, *child_texts] if part)
        if combined:
            result.append(f"{header}: {combined}")
    return result


def _nested_child_source_texts(children: object) -> list[str]:
    if not isinstance(children, list):
        return []
    result: list[str] = []
    for child in children:
        if not isinstance(child, Mapping):
            continue
        child_type = child.get("type", child.get("kind"))
        content = child.get("content")
        if child_type == "table" and isinstance(content, Mapping):
            result.append("nested table: " + _inline_table_source(content))
            continue
        if child_type == "image" and isinstance(content, Mapping):
            asset_id = str(content.get("asset_id", "")).strip()
            if asset_id:
                result.append(f"image: {asset_id}")
            continue
        if child_type == "diagram" and isinstance(content, Mapping):
            source = _diagram_source_text(content).replace("\n", " / ")
            if source:
                result.append(f"diagram: {source}")
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


def _paragraph_drawing(
    paragraph: ET.Element,
    warnings: list[dict[str, Any]],
) -> _DrawingResult | None:
    candidates = list(_drawing_candidates(paragraph))
    if not candidates:
        return None

    nodes: list[dict[str, object]] = []
    connectors: list[dict[str, object]] = []
    for element in candidates:
        local = _local_name(element.tag)
        if local in _UNSUPPORTED_DRAWING_TAGS:
            _warn_unsupported_drawing(warnings, local)
            continue
        text = _element_text(element)
        if local in _DRAWING_NODE_TAGS and text:
            nodes.append(
                {
                    "id": f"n{len(nodes) + 1}",
                    "shape_type": local,
                    "text": text,
                    "bbox": _bbox_from_element(element),
                    "metadata": {"source": "hwpx_drawing_text"},
                }
            )
            continue
        if local in _DRAWING_CONNECTOR_TAGS:
            connector = _structured_connector(len(connectors) + 1, element)
            if connector is not None:
                connectors.append(connector)
            continue
        if local not in _DRAWING_NODE_TAGS:
            continue

    if len(nodes) == 1 and not connectors:
        return _DrawingResult(single_text=str(nodes[0]["text"]))
    if not nodes:
        return None

    structured = _structured_diagram_content(
        nodes=nodes,
        edges=_infer_connector_edges(nodes, connectors),
        connectors=connectors,
    )
    return _DrawingResult(structured=structured)


def _drawing_candidates(paragraph: ET.Element) -> list[ET.Element]:
    candidates: list[ET.Element] = []
    for run in paragraph.findall(_q("run")):
        for child in list(run):
            candidates.extend(_drawing_candidates_from_element(child))
    return candidates


def _drawing_candidates_from_element(element: ET.Element) -> list[ET.Element]:
    local = _local_name(element.tag)
    if local in {"drawText", "pic", "tbl"}:
        return []
    if local in _DRAWING_CONNECTOR_TAGS or local in _UNSUPPORTED_DRAWING_TAGS:
        return [element]
    if local in _DRAWING_NODE_TAGS:
        text = _element_text(element)
        if text or local != "container":
            return [element]
        candidates: list[ET.Element] = []
        for child in list(element):
            candidates.extend(_drawing_candidates_from_element(child))
        return candidates

    candidates = []
    for child in list(element):
        candidates.extend(_drawing_candidates_from_element(child))
    return candidates


def _structured_connector(
    index: int,
    element: ET.Element,
) -> dict[str, object] | None:
    bbox = _bbox_from_element(element, allow_flat=True)
    if bbox is None:
        return None
    local = _local_name(element.tag)
    connector_type = "line" if local == "connectLine" else local
    return {
        "id": f"c{index}",
        "type": connector_type,
        "bbox": bbox,
        "points": _points_from_element(element) or _line_points_from_bbox(bbox),
        "arrow": _line_has_arrow(element),
        "metadata": {"source": f"hwpx_{connector_type}"},
    }


def _bbox_from_element(
    element: ET.Element,
    *,
    allow_flat: bool = False,
) -> dict[str, int | str] | None:
    pos = _first_descendant(element, "pos")
    size = _first_descendant(element, "sz")
    x = _int_attr(pos, ("x", "left")) if pos is not None else 0
    y = _int_attr(pos, ("y", "top")) if pos is not None else 0
    width = _int_attr(size, ("width", "w", "cx")) if size is not None else 0
    height = _int_attr(size, ("height", "h", "cy")) if size is not None else 0
    x = _int_attr(element, ("x", "left"), x)
    y = _int_attr(element, ("y", "top"), y)
    width = _int_attr(element, ("width", "w", "cx"), width)
    height = _int_attr(element, ("height", "h", "cy"), height)
    if allow_flat:
        if width <= 0 and height <= 0:
            return None
    elif width <= 0 or height <= 0:
        return None
    return {
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "unit": "hwpx",
    }


def _points_from_element(element: ET.Element) -> list[dict[str, int]]:
    points: list[dict[str, int]] = []
    for descendant in element.iter():
        local = _local_name(descendant.tag)
        if local not in {"pt", "point"}:
            continue
        point = _xy_from_attrs(descendant, ("x",), ("y",))
        if point is not None:
            points.append(point)
    if len(points) >= 2:
        return points

    start = _xy_from_attrs(
        element,
        ("x1", "startX", "fromX"),
        ("y1", "startY", "fromY"),
    )
    end = _xy_from_attrs(
        element,
        ("x2", "endX", "toX"),
        ("y2", "endY", "toY"),
    )
    if start is not None and end is not None:
        return [start, end]
    return []


def _xy_from_attrs(
    element: ET.Element,
    x_names: tuple[str, ...],
    y_names: tuple[str, ...],
) -> dict[str, int] | None:
    x = _optional_int_attr(element, x_names)
    y = _optional_int_attr(element, y_names)
    if x is None or y is None:
        return None
    return {"x": x, "y": y}


def _first_descendant(element: ET.Element, local_name: str) -> ET.Element | None:
    for descendant in element.iter():
        if descendant is element:
            continue
        if _local_name(descendant.tag) == local_name:
            return descendant
    return None


def _int_attr(
    element: ET.Element | None,
    names: tuple[str, ...],
    default: int = 0,
) -> int:
    value = _optional_int_attr(element, names)
    return default if value is None else value


def _optional_int_attr(
    element: ET.Element | None,
    names: tuple[str, ...],
) -> int | None:
    if element is None:
        return None
    for name in names:
        value = element.get(name)
        if value is None:
            continue
        try:
            return int(float(value))
        except ValueError:
            continue
    return None


def _line_points_from_bbox(bbox: dict[str, int | str]) -> list[dict[str, int]]:
    x = _bbox_int(bbox, "x")
    y = _bbox_int(bbox, "y")
    width = _bbox_int(bbox, "width")
    height = _bbox_int(bbox, "height")
    if width >= max(height, 1) * 3:
        y_mid = y + max(height, 1) // 2
        return [{"x": x, "y": y_mid}, {"x": x + width, "y": y_mid}]
    if height >= max(width, 1) * 3:
        x_mid = x + max(width, 1) // 2
        return [{"x": x_mid, "y": y}, {"x": x_mid, "y": y + height}]
    return [{"x": x, "y": y}, {"x": x + width, "y": y + height}]


def _line_has_arrow(element: ET.Element) -> bool:
    for item in element.iter():
        for attr, value in item.attrib.items():
            if "arrow" not in _local_name(attr).lower():
                continue
            normalized = value.strip().lower()
            if normalized not in {"", "0", "false", "none", "null"}:
                return True
    return False


def _infer_connector_edges(
    nodes: list[dict[str, object]],
    connectors: list[dict[str, object]],
) -> list[dict[str, object]]:
    bbox_nodes = [
        (str(node.get("id", "")), bbox)
        for node in nodes
        if (bbox := _diagram_node_bbox(node)) is not None
    ]
    if len(bbox_nodes) < 2:
        return []

    edges: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    edge_labels = _diagram_connector_labels(nodes)
    for connector_index, connector in enumerate(connectors):
        points = connector.get("points")
        if not isinstance(points, list) or len(points) < 2:
            continue
        start = _diagram_point(points[0])
        end = _diagram_point(points[1])
        if start is None or end is None:
            continue
        from_id = _nearest_node_id(start, bbox_nodes)
        to_id = _nearest_node_id(end, bbox_nodes)
        if from_id is None or to_id is None or from_id == to_id:
            continue
        connector_id = str(connector.get("id", ""))
        key = (from_id, to_id, connector_id)
        if key in seen:
            continue
        seen.add(key)
        edges.append(
            {
                "from": from_id,
                "to": to_id,
                "type": "arrow" if connector.get("arrow") else "line",
                "label": (
                    edge_labels[connector_index]
                    if connector_index < len(edge_labels)
                    else ""
                ),
                "confidence": "inferred_geometry",
                "connector_id": connector_id,
            }
        )
    return edges


def _diagram_connector_labels(nodes: list[dict[str, object]]) -> list[str]:
    labels: list[str] = []
    for node in nodes:
        if _diagram_node_bbox(node) is not None:
            continue
        text = str(node.get("text", "")).strip()
        if not text:
            continue
        if _is_diagram_step_label(text):
            labels.append(text)
            continue
        if (
            labels
            and not _is_diagram_section_heading(text)
            and not _is_diagram_note(text)
        ):
            labels[-1] = f"{labels[-1]}\n{text}"
    return labels


def _is_diagram_step_label(text: str) -> bool:
    return bool(_DIAGRAM_STEP_LABEL_RE.match(text.strip()))


def _is_diagram_section_heading(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith("<") and stripped.endswith(">")


def _is_diagram_note(text: str) -> bool:
    stripped = text.strip()
    return (
        (stripped.startswith("(") and stripped.endswith(")"))
        or (stripped.startswith("[") and stripped.endswith("]"))
    )


def _diagram_node_bbox(node: dict[str, object]) -> dict[str, int] | None:
    bbox = node.get("bbox")
    if not isinstance(bbox, Mapping):
        return None
    x = _bbox_int(bbox, "x")
    y = _bbox_int(bbox, "y")
    width = _bbox_int(bbox, "width")
    height = _bbox_int(bbox, "height")
    if width <= 0 or height <= 0:
        return None
    return {"x": x, "y": y, "width": width, "height": height}


def _diagram_point(point: object) -> dict[str, int] | None:
    if not isinstance(point, Mapping):
        return None
    try:
        return {"x": int(point["x"]), "y": int(point["y"])}
    except (KeyError, TypeError, ValueError):
        return None


def _nearest_node_id(
    point: dict[str, int],
    bbox_nodes: list[tuple[str, dict[str, int]]],
) -> str | None:
    best_id: str | None = None
    best_distance: int | None = None
    for node_id, bbox in bbox_nodes:
        distance = _point_bbox_distance_squared(point, bbox)
        if best_distance is None or distance < best_distance:
            best_id = node_id
            best_distance = distance
    return best_id


def _point_bbox_distance_squared(
    point: dict[str, int],
    bbox: dict[str, int],
) -> int:
    min_x = bbox["x"]
    max_x = bbox["x"] + bbox["width"]
    min_y = bbox["y"]
    max_y = bbox["y"] + bbox["height"]
    dx = max(min_x - point["x"], 0, point["x"] - max_x)
    dy = max(min_y - point["y"], 0, point["y"] - max_y)
    return dx * dx + dy * dy


def _diagram_source_text(diagram: dict[str, object]) -> str:
    nodes = diagram.get("nodes", [])
    if not isinstance(nodes, list):
        return ""
    lines = [
        text
        for text in (
            str(node.get("text", "")).strip()
            for node in nodes
            if isinstance(node, Mapping)
        )
        if text
    ]
    edge_lines = _diagram_edge_source_lines(diagram.get("edges", []))
    if edge_lines:
        lines.append("relations:")
        lines.extend(edge_lines)
    return "\n".join(lines)


def _table_flowchart_diagram(table: dict[str, object]) -> dict[str, object] | None:
    rows = table.get("rows")
    if not isinstance(rows, list):
        return None

    flowchart_rows = _flowchart_rows(rows)
    if not flowchart_rows:
        return None

    min_row_addr = _min_flowchart_row_addr(flowchart_rows)
    nodes: list[dict[str, object]] = []
    label_nodes: list[dict[str, object]] = []
    connectors: list[dict[str, object]] = []
    seen_node_texts: set[str] = set()
    for row in flowchart_rows:
        if not isinstance(row, Mapping):
            continue
        cells = row.get("cells")
        if not isinstance(cells, list):
            continue
        for cell in cells:
            if not isinstance(cell, Mapping):
                continue
            text = _flowchart_cell_text(cell)
            if not text:
                continue
            if _is_flowchart_edge_label(text):
                label = _flowchart_edge_label(text)
                connectors.append(
                    _table_flowchart_connector(
                        len(connectors) + 1,
                        text,
                        label,
                        _table_cell_grid_bbox(cell, min_row_addr),
                    )
                )
                label_nodes.append(
                    {
                        "id": "",
                        "shape_type": "label",
                        "text": label,
                        "bbox": None,
                        "metadata": {
                            "source": "hwpx_table_flowchart_label",
                            "raw_label": text,
                        },
                    }
                )
                continue
            if text not in seen_node_texts:
                seen_node_texts.add(text)
                nodes.append(
                    {
                        "id": "",
                        "shape_type": "label",
                        "text": text,
                        "bbox": _table_cell_grid_bbox(cell, min_row_addr),
                        "metadata": {
                            "source": "hwpx_table_flowchart",
                            "role": (
                                "title" if _is_flowchart_title(text) else "node"
                            ),
                        },
                    }
                )

    if len(nodes) < 2:
        return None

    all_nodes = _assign_node_ids(nodes + label_nodes)
    edges = _infer_table_grid_edges(all_nodes, connectors)
    return _structured_diagram_content(nodes=all_nodes, edges=edges, connectors=connectors)


def _assign_node_ids(nodes: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            **node,
            "id": f"n{index}",
        }
        for index, node in enumerate(nodes, start=1)
    ]


def _min_flowchart_row_addr(rows: list[object]) -> int:
    row_addrs = [
        int(cell["row_addr"])
        for row in rows
        if isinstance(row, Mapping)
        for cell in row.get("cells", [])
        if isinstance(cell, Mapping) and "row_addr" in cell
    ]
    return min(row_addrs, default=0)


def _table_cell_grid_bbox(
    cell: dict[str, object],
    min_row_addr: int,
) -> dict[str, int | str]:
    return {
        "x": _int_value(cell.get("col_addr"), 0),
        "y": max(0, _int_value(cell.get("row_addr"), min_row_addr) - min_row_addr),
        "width": _int_value(cell.get("colspan"), 1),
        "height": _int_value(cell.get("rowspan"), 1),
        "unit": "hwpx_table_grid",
    }


def _int_value(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _table_flowchart_connector(
    index: int,
    raw_label: str,
    label: str,
    bbox: dict[str, int | str],
) -> dict[str, object]:
    return {
        "id": f"c{index}",
        "type": "arrow",
        "bbox": bbox,
        "points": _table_flowchart_connector_points(raw_label, bbox),
        "arrow": True,
        "metadata": {
            "source": "hwpx_table_flowchart",
            "label": label,
            "raw_label": raw_label,
        },
    }


def _table_flowchart_connector_points(
    raw_label: str,
    bbox: dict[str, int | str],
) -> list[dict[str, float | int]]:
    x = _int_value(bbox.get("x"), 0)
    y = _int_value(bbox.get("y"), 0)
    width = _int_value(bbox.get("width"), 1)
    height = _int_value(bbox.get("height"), 1)
    x_mid = x + width / 2
    y_mid = y + height / 2
    arrow = _flowchart_arrow(raw_label)
    if arrow == "←":
        return [{"x": x + width, "y": y_mid}, {"x": x, "y": y_mid}]
    if arrow == "↑":
        return [{"x": x_mid, "y": y + height}, {"x": x_mid, "y": y}]
    if arrow == "↕":
        return [{"x": x_mid, "y": y + height}, {"x": x_mid, "y": y}]
    if arrow == "↓":
        return [{"x": x_mid, "y": y}, {"x": x_mid, "y": y + height}]
    return [{"x": x, "y": y_mid}, {"x": x + width, "y": y_mid}]


def _flowchart_arrow(text: str) -> str:
    match = re.search(r"[→←↑↓↕]", text)
    return match.group(0) if match is not None else "→"


def _flowchart_edge_label(text: str) -> str:
    label = re.sub(r"^[→←↑↓↕\s]+", "", text).strip()
    if label.startswith("(") and label.endswith(")"):
        label = label[1:-1].strip()
    return label or text


def _infer_table_grid_edges(
    nodes: list[dict[str, object]],
    connectors: list[dict[str, object]],
) -> list[dict[str, object]]:
    bbox_nodes = [
        (str(node.get("id", "")), bbox)
        for node in nodes
        if not _is_table_flowchart_title_node(node)
        and (bbox := _diagram_node_bbox(node)) is not None
    ]
    edges: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for connector in connectors:
        points = connector.get("points")
        if not isinstance(points, list) or len(points) < 2:
            continue
        start = _diagram_point(points[0])
        end = _diagram_point(points[1])
        if start is None or end is None:
            continue
        from_id = _nearest_node_id(start, bbox_nodes)
        to_id = _nearest_node_id(end, bbox_nodes)
        if from_id is not None and from_id == to_id:
            from_id, to_id = _directional_table_grid_edge_node_ids(
                connector,
                bbox_nodes,
            )
        if from_id is None or to_id is None or from_id == to_id:
            continue
        connector_id = str(connector.get("id", ""))
        key = (from_id, to_id, connector_id)
        if key in seen:
            continue
        seen.add(key)
        metadata = connector.get("metadata")
        label = (
            str(metadata.get("label", "")).strip()
            if isinstance(metadata, Mapping)
            else ""
        )
        edges.append(
            {
                "from": from_id,
                "to": to_id,
                "type": "arrow",
                "label": label,
                "confidence": "inferred_table_grid",
                "connector_id": connector_id,
            }
        )
    return edges


def _directional_table_grid_edge_node_ids(
    connector: dict[str, object],
    bbox_nodes: list[tuple[str, dict[str, int]]],
) -> tuple[str | None, str | None]:
    bbox = connector.get("bbox")
    if not isinstance(bbox, Mapping):
        return None, None
    center = {
        "x": _int_value(bbox.get("x"), 0)
        + _int_value(bbox.get("width"), 1) / 2,
        "y": _int_value(bbox.get("y"), 0)
        + _int_value(bbox.get("height"), 1) / 2,
    }
    metadata = connector.get("metadata")
    raw_label = (
        str(metadata.get("raw_label", ""))
        if isinstance(metadata, Mapping)
        else ""
    )
    arrow = _flowchart_arrow(raw_label)
    if arrow == "←":
        return (
            _nearest_directional_node_id(center, bbox_nodes, "right"),
            _nearest_directional_node_id(center, bbox_nodes, "left"),
        )
    if arrow in {"↑", "↕"}:
        return (
            _nearest_directional_node_id(center, bbox_nodes, "below"),
            _nearest_directional_node_id(center, bbox_nodes, "above"),
        )
    if arrow == "↓":
        return (
            _nearest_directional_node_id(center, bbox_nodes, "above"),
            _nearest_directional_node_id(center, bbox_nodes, "below"),
        )
    return (
        _nearest_directional_node_id(center, bbox_nodes, "left"),
        _nearest_directional_node_id(center, bbox_nodes, "right"),
    )


def _nearest_directional_node_id(
    point: dict[str, float],
    bbox_nodes: list[tuple[str, dict[str, int]]],
    direction: str,
) -> str | None:
    candidates = [
        (node_id, bbox)
        for node_id, bbox in bbox_nodes
        if _bbox_is_in_direction(point, bbox, direction)
    ]
    return _nearest_node_id(point, candidates)


def _bbox_is_in_direction(
    point: dict[str, float],
    bbox: dict[str, int],
    direction: str,
) -> bool:
    if direction == "left":
        return bbox["x"] + bbox["width"] <= point["x"]
    if direction == "right":
        return bbox["x"] >= point["x"]
    if direction == "above":
        return bbox["y"] + bbox["height"] <= point["y"]
    if direction == "below":
        return bbox["y"] >= point["y"]
    return False


def _is_table_flowchart_title_node(node: dict[str, object]) -> bool:
    metadata = node.get("metadata")
    return isinstance(metadata, Mapping) and metadata.get("role") == "title"


def _table_without_flowchart_rows(
    table: dict[str, object],
) -> dict[str, object]:
    rows = table.get("rows")
    if not isinstance(rows, list):
        return table
    flowchart_rows = _flowchart_rows(rows)
    if not flowchart_rows:
        return table

    flowchart_row_ids = {id(row) for row in flowchart_rows}
    return {
        **table,
        "rows": [
            row
            for row in rows
            if id(row) not in flowchart_row_ids
        ],
    }


def _flowchart_rows(rows: list[object]) -> list[object]:
    start: int | None = None
    for index, row in enumerate(rows):
        texts = _row_cell_texts(row)
        if any(_is_flowchart_title(text) for text in texts):
            start = index
            break
    if start is None:
        return []

    result: list[object] = []
    for row in rows[start:]:
        texts = _row_cell_texts(row)
        if result and any(_is_paper_size_note(text) for text in texts):
            break
        result.append(row)
    return result


def _row_cell_texts(row: object) -> list[str]:
    if not isinstance(row, Mapping):
        return []
    cells = row.get("cells")
    if not isinstance(cells, list):
        return []
    return [
        text
        for cell in cells
        if isinstance(cell, Mapping)
        and (text := _flowchart_cell_text(cell))
    ]


def _flowchart_cell_text(cell: dict[str, object]) -> str:
    return _clean_text(str(cell.get("text", ""))).strip("<> ")


def _is_flowchart_title(text: str) -> bool:
    return "등록절차" in text or "처리절차" in text or "업무처리" in text


def _is_flowchart_edge_label(text: str) -> bool:
    return bool(re.search(r"[→←↑↓↕]", text))


def _is_paper_size_note(text: str) -> bool:
    return "mm×" in text or "일반용지" in text


def _diagram_edge_source_lines(edges: object) -> list[str]:
    if not isinstance(edges, list):
        return []
    lines: list[str] = []
    for edge in edges:
        if not isinstance(edge, Mapping):
            continue
        from_id = str(edge.get("from", "")).strip()
        to_id = str(edge.get("to", "")).strip()
        if not from_id or not to_id:
            continue
        label = str(edge.get("label", "")).strip()
        line = f"{from_id} -> {to_id}"
        if label:
            line = f"{line}: {label}"
        lines.append(line)
    return lines


def _element_text(element: ET.Element) -> str:
    parts: list[str] = []
    for descendant in element.iter():
        if _local_name(descendant.tag) != "t":
            continue
        if descendant.text:
            parts.append("".join(char for char in descendant.text if char > "\x1f"))
    return _clean_text(" ".join(parts))


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _bbox_int(bbox: dict[str, int | str], key: str) -> int:
    try:
        return int(bbox[key])
    except (KeyError, TypeError, ValueError):
        return 0


def _warn_unsupported_drawing(
    warnings: list[dict[str, Any]],
    element: str,
) -> None:
    warning = {
        "type": "hwpx_drawing_structure_unsupported",
        "severity": "medium",
        "element": element,
        "message": f"Unsupported HWPX drawing structure was skipped: {element}",
    }
    if warning not in warnings:
        warnings.append(warning)


def _append_ocr_fallback_units(
    units: list[EvidenceUnit],
    assets: list[PendingAsset],
    warnings: list[dict[str, Any]],
    ocr_fn: Callable[[bytes, int], str] | None,
    block_index: int,
) -> int:
    if ocr_fn is None or not assets or _has_native_source_text(units):
        return block_index

    for image_index, asset in enumerate(assets):
        try:
            text = _clean_text(ocr_fn(asset.data, image_index) or "")
        except Exception as exc:
            warnings.append(
                {
                    "type": "hwpx_ocr_failed",
                    "severity": "medium",
                    "image_index": image_index + 1,
                    "asset_id": asset.id,
                    "message": str(exc),
                }
            )
            continue
        if not text:
            warnings.append(
                {
                    "type": "hwpx_ocr_failed",
                    "severity": "medium",
                    "image_index": image_index + 1,
                    "asset_id": asset.id,
                    "message": "empty OCR result",
                }
            )
            continue
        units.append(_text_unit(f"b{block_index}", text))
        block_index += 1
    return block_index


def _has_native_source_text(units: list[EvidenceUnit]) -> bool:
    return any(
        unit.type in {"text", "table", "diagram"} and unit.source.text.strip()
        for unit in units
    )


def _extract_image(
    picture: ET.Element,
    z: zipfile.ZipFile,
    bin_data_map: dict[str, str],
    index: int,
    warnings: list[dict[str, Any]],
) -> tuple[str, PendingAsset] | None:
    ref = _image_ref(picture)
    if ref is None:
        _warn_unresolved_image(warnings, "")
        return None
    href = bin_data_map.get(ref, "")
    if not href or href not in z.namelist():
        _warn_unresolved_image(warnings, ref)
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


def _warn_unresolved_image(
    warnings: list[dict[str, Any]],
    ref: str,
) -> None:
    warnings.append(
        {
            "type": "hwpx_image_reference_unresolved",
            "severity": "medium",
            "ref": ref,
            "message": f"HWPX image reference could not be resolved: {ref}",
        }
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
