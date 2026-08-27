from __future__ import annotations

import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from xml.etree import ElementTree as ET

from ....models import PendingAsset
from ...schema import structured_table as _structured_table_content
from .diagram import HwpxDiagramBuilder
from .package_reader import HwpxPackageReader
from .xml_utils import paragraph_text as _paragraph_text
from .xml_utils import q as _q

_DIAGRAM_BUILDER = HwpxDiagramBuilder()
_PACKAGE_READER = HwpxPackageReader()
_paragraph_drawing = _DIAGRAM_BUILDER.parse_paragraph
_extract_image = _PACKAGE_READER.extract_image


@dataclass(frozen=True)
class HwpxTableExtractor:
    """Extracts canonical tables and nested evidence from HWPX XML."""

    def extract(
        self,
        table: ET.Element,
        archive: zipfile.ZipFile,
        bin_data_map: dict[str, str],
        assets: list[PendingAsset],
        warnings: list[dict[str, Any]],
    ) -> dict[str, object]:
        return _structured_table(table, archive, bin_data_map, assets, warnings)

    def public(self, table: dict[str, object]) -> dict[str, object]:
        return _public_structured_table(table)

    def single_cell_text(self, table: dict[str, object]) -> str | None:
        return _single_cell_text_table_text(table)


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
