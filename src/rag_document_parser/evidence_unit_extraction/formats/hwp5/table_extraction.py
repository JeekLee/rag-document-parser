from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ...schema import structured_table as _structured_table_content
from .text import clean_text as _clean_text


@dataclass(frozen=True)
class Hwp5TableBuilder:
    """Builds canonical structured tables from parsed HWP5 cells."""

    def build(
        self,
        rows: list[list[Cell]],
        *,
        row_count: int | None = None,
        column_count: int | None = None,
    ) -> dict[str, object]:
        return _structured_table(
            rows,
            row_count=row_count,
            column_count=column_count,
        )

    def has_content(self, rows: list[list[Cell]]) -> bool:
        return _table_has_content(rows)

    def single_cell_text(
        self,
        rows: list[list[Cell]],
        *,
        row_count: int | None,
        column_count: int | None,
    ) -> str | None:
        return _single_cell_table_text(
            rows,
            row_count=row_count,
            column_count=column_count,
        )


@dataclass
class Cell:
    text: str = ""
    children: list[dict[str, object]] = field(default_factory=list)
    row_addr: int | None = None
    col_addr: int | None = None
    rowspan: int = 1
    colspan: int = 1
    synthetic: bool = False


def _structured_table(
    rows: list[list[Cell]],
    *,
    row_count: int | None = None,
    column_count: int | None = None,
) -> dict[str, object]:
    normalized = _normalize_rows(
        rows,
        row_count=row_count,
        column_count=column_count,
    )
    if not normalized:
        return _structured_table_content(columns=[], rows=[])

    actual_column_count = _table_column_count(normalized, column_count)
    header_count = _header_row_count(normalized)
    header_raw_rows = normalized[:header_count]
    visible_header_raw_rows = _visible_header_rows(header_raw_rows)
    data_raw_rows = normalized[header_count:]
    columns = _table_columns(actual_column_count, visible_header_raw_rows)
    omit_blank_cells = _should_omit_blank_cells(
        normalized,
        column_count=actual_column_count,
    )
    header_rows = [
        {
            "index": index,
            "cells": _evidence_cells(
                raw_cells,
                columns,
                omit_blank_cells=omit_blank_cells,
            ),
        }
        for index, raw_cells in enumerate(visible_header_raw_rows, start=1)
    ]

    data_rows: list[dict[str, object]] = []
    for row in data_raw_rows:
        if row_count is None and _row_is_blank(row):
            continue
        data_rows.append(
            {
                "index": len(data_rows) + 1,
                "cells": _evidence_cells(
                    row,
                    columns,
                    omit_blank_cells=omit_blank_cells,
                ),
            }
        )

    structured = _structured_table_content(
        columns=columns,
        rows=data_rows,
        header_rows=header_rows,
    )
    omitted_count = _omitted_blank_cell_count(
        normalized,
        omit=omit_blank_cells,
    )
    if omitted_count:
        structured["compact"] = {"omitted_blank_cells": omitted_count}
    return structured


def _evidence_cells(
    row: list[Cell],
    columns: list[dict[str, str]],
    *,
    omit_blank_cells: bool = False,
) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    for cell in sorted(row, key=lambda item: item.col_addr or 0):
        if omit_blank_cells and not cell.text.strip() and not cell.children:
            continue
        column_index = cell.col_addr or 0
        column_id = (
            columns[column_index]["id"]
            if 0 <= column_index < len(columns)
            else f"c{column_index + 1}"
        )
        cells.append(
            {
                "column_id": column_id,
                "text": cell.text,
                "rowspan": cell.rowspan,
                "colspan": cell.colspan,
                "children": list(cell.children),
            }
        )
    return cells


def _normalize_rows(
    rows: list[list[Cell]],
    *,
    row_count: int | None = None,
    column_count: int | None = None,
) -> list[list[Cell]]:
    row_map: dict[int, list[Cell]] = {}
    all_cells: list[Cell] = []
    for fallback_row, row in enumerate(rows):
        fallback_col = 0
        for cell in row:
            row_addr = (
                cell.row_addr
                if cell.row_addr is not None and cell.row_addr >= 0
                else fallback_row
            )
            col_addr = (
                cell.col_addr
                if cell.col_addr is not None and cell.col_addr >= 0
                else fallback_col
            )
            cleaned = Cell(
                text=_clean_text(cell.text),
                children=list(cell.children),
                row_addr=row_addr,
                col_addr=col_addr,
                rowspan=max(1, cell.rowspan),
                colspan=max(1, cell.colspan),
                synthetic=cell.synthetic,
            )
            row_map.setdefault(row_addr, []).append(cleaned)
            all_cells.append(cleaned)
            fallback_col = col_addr + cleaned.colspan

    _clip_rowspans_that_overlap_real_cells(all_cells)
    max_row_end = max(
        ((cell.row_addr or 0) + cell.rowspan for cell in all_cells),
        default=0,
    )
    max_col_end = max(
        ((cell.col_addr or 0) + cell.colspan for cell in all_cells),
        default=0,
    )
    actual_column_count = max(column_count or 0, max_col_end)
    if actual_column_count <= 0:
        return []
    if row_count is not None:
        row_indexes = list(range(max(row_count, max_row_end)))
    else:
        row_indexes = sorted(row_map)

    normalized: list[list[Cell]] = []
    for row_index in row_indexes:
        current_cells = row_map.get(row_index, [])
        covered = _covered_columns_from_rowspans(all_cells, row_index)
        occupied = set(covered)
        for cell in current_cells:
            start = cell.col_addr or 0
            occupied.update(range(start, min(actual_column_count, start + cell.colspan)))

        materialized = list(current_cells)
        for col_addr in range(actual_column_count):
            if col_addr in occupied:
                continue
            materialized.append(
                Cell(row_addr=row_index, col_addr=col_addr, synthetic=True)
            )
        if materialized:
            normalized.append(sorted(materialized, key=lambda item: item.col_addr or 0))
    return normalized


def _visible_header_rows(rows: list[list[Cell]]) -> list[list[Cell]]:
    visible = [
        (_row_start(row), row)
        for row in rows
        if not _row_is_blank(row)
    ]
    visible_row_addrs = [row_addr for row_addr, _ in visible]
    adjusted_rows: list[list[Cell]] = []
    for row_addr, row in visible:
        adjusted_row: list[Cell] = []
        for cell in row:
            span_start = cell.row_addr if cell.row_addr is not None else row_addr
            span_end = span_start + max(1, cell.rowspan)
            visible_span = sum(
                1
                for visible_addr in visible_row_addrs
                if span_start <= visible_addr < span_end
            )
            adjusted_row.append(
                Cell(
                    text=cell.text,
                    children=list(cell.children),
                    row_addr=cell.row_addr,
                    col_addr=cell.col_addr,
                    rowspan=max(1, visible_span),
                    colspan=cell.colspan,
                    synthetic=cell.synthetic,
                )
            )
        adjusted_rows.append(adjusted_row)
    return adjusted_rows


def _clip_rowspans_that_overlap_real_cells(cells: list[Cell]) -> None:
    cells_by_row: dict[int, list[Cell]] = {}
    for cell in cells:
        cells_by_row.setdefault(cell.row_addr or 0, []).append(cell)

    for cell in cells:
        if cell.rowspan <= 1:
            continue
        row_addr = cell.row_addr or 0
        start = cell.col_addr or 0
        end = start + cell.colspan
        for covered_row in range(row_addr + 1, row_addr + cell.rowspan):
            if any(
                _cells_overlap_columns(start, end, other)
                and _cell_has_real_content(other)
                for other in cells_by_row.get(covered_row, [])
            ):
                cell.rowspan = max(1, covered_row - row_addr)
                break


def _cells_overlap_columns(start: int, end: int, cell: Cell) -> bool:
    other_start = cell.col_addr or 0
    other_end = other_start + cell.colspan
    return start < other_end and other_start < end


def _cell_has_real_content(cell: Cell) -> bool:
    return bool(cell.text.strip() or cell.children)


def _should_omit_blank_cells(
    rows: list[list[Cell]],
    *,
    column_count: int,
) -> bool:
    cells = [cell for row in rows for cell in row]
    if column_count < 10 or len(cells) < 50:
        return False
    blank_count = _omitted_blank_cell_count(rows, omit=True)
    return blank_count / len(cells) >= 0.5


def _omitted_blank_cell_count(
    rows: list[list[Cell]],
    *,
    omit: bool,
) -> int:
    if not omit:
        return 0
    return sum(
        1
        for row in rows
        for cell in row
        if not cell.text.strip() and not cell.children
    )


def _covered_columns_from_rowspans(cells: list[Cell], row_index: int) -> set[int]:
    covered: set[int] = set()
    for cell in cells:
        row_addr = cell.row_addr or 0
        if not row_addr < row_index < row_addr + cell.rowspan:
            continue
        col_addr = cell.col_addr or 0
        covered.update(range(col_addr, col_addr + cell.colspan))
    return covered


def _table_column_count(
    rows: list[list[Cell]],
    declared_column_count: int | None,
) -> int:
    return max(
        declared_column_count or 0,
        max(
            (
                (cell.col_addr or 0) + cell.colspan
                for row in rows
                for cell in row
            ),
            default=0,
        ),
    )


def _table_columns(
    column_count: int,
    header_rows: list[list[Cell]],
) -> list[dict[str, str]]:
    return [
        {
            "id": f"c{index}",
            "text": _column_header_text(header_rows, index - 1),
        }
        for index in range(1, column_count + 1)
    ]


def _column_header_text(
    header_rows: list[list[Cell]],
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
            text = cell.text.strip()
            if text and text not in texts:
                texts.append(text)
    return " / ".join(texts)


def _header_cell_contributes_to_column(
    cell: Cell,
    column_index: int,
    row_index: int,
    last_header_row: int,
) -> bool:
    start = cell.col_addr or 0
    end = start + cell.colspan
    return column_index == start or (
        start < column_index < end and row_index < last_header_row
    )


def _header_row_count(rows: list[list[Cell]]) -> int:
    first_row = rows[0]
    if any(cell.children for cell in first_row):
        return 0
    if len(rows) == 1:
        return 1
    count = 1
    header_row_end = _row_span_end(first_row)
    last_header_row = first_row
    while count < len(rows):
        row = rows[count]
        if _row_is_blank(row):
            if (
                count + 1 < len(rows)
                and _row_refines_previous_header(rows[count + 1], last_header_row)
            ):
                count += 1
                continue
            break
        row_start = _row_start(row)
        if (
            row_start < header_row_end
            or _row_refines_previous_header(row, last_header_row)
        ):
            count += 1
            header_row_end = max(header_row_end, _row_span_end(row))
            last_header_row = row
            continue
        break
    return count


def _row_refines_previous_header(
    row: list[Cell],
    previous_row: list[Cell],
) -> bool:
    if any(cell.children for cell in row) or _row_is_blank(row):
        return False
    groups = [
        cell
        for cell in previous_row
        if cell.colspan > 1 and cell.text.strip()
    ]
    if not groups:
        return False
    for group in groups:
        group_start = group.col_addr or 0
        group_end = group_start + group.colspan
        refiners = [
            cell
            for cell in row
            if group_start <= (cell.col_addr or 0)
            and (cell.col_addr or 0) + cell.colspan <= group_end
        ]
        if not any(cell.text.strip() for cell in refiners):
            return False
    return True


def _row_start(row: list[Cell]) -> int:
    return min((cell.row_addr or 0 for cell in row), default=0)


def _row_span_end(row: list[Cell]) -> int:
    return max(
        (
            (cell.row_addr or 0) + cell.rowspan
            for cell in row
        ),
        default=0,
    )


def _row_is_blank(row: list[Cell]) -> bool:
    return not any(cell.text.strip() or cell.children for cell in row)


def _single_cell_table_text(
    rows: list[list[Cell]],
    *,
    row_count: int | None = None,
    column_count: int | None = None,
) -> str | None:
    normalized = _normalize_rows(
        rows,
        row_count=row_count,
        column_count=column_count,
    )
    if len(normalized) != 1 or len(normalized[0]) != 1:
        return None
    cell = normalized[0][0]
    if cell.children:
        return None
    text = cell.text.strip()
    return text or None


def _table_has_content(rows: list[list[Cell]]) -> bool:
    return any(cell.text.strip() or cell.children for row in rows for cell in row)
