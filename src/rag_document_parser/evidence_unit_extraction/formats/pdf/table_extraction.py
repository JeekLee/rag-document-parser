from __future__ import annotations

import contextlib
import io
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ...schema import structured_table as _structured_table_content
from .geometry import (
    _bbox_area,
    _bbox_in_cell,
    _bbox_near_equal,
    _bbox_overlap_ratio,
    _coerce_bbox,
)
from .models import NestedResolution as _NestedResolution
from .models import TableCellSpans as _TableCellSpans
from .source_projection import PdfTableSourceProjector
from .table_normalization import PdfTableNormalizer

_PAGE_NUM_RE = re.compile(r"(?m)^\s*(?:-\s*)?\d+\s*(?:-\s*)?$")
_CJK = re.compile(r"[가-힣一-鿿㐀-䶿]")
_NORMALIZER = PdfTableNormalizer()
_PDF_TABLE_NORMALIZER = _NORMALIZER
_SOURCE_PROJECTOR = PdfTableSourceProjector()
_cell_has_content = _NORMALIZER.cell_has_content
_expand_parallel_code_action_rows = _NORMALIZER.expand_parallel_rows
_is_table_of_contents = _NORMALIZER.is_table_of_contents
_merge_nested_table_children = _NORMALIZER.merge_nested_children
_normalize_header_text = _NORMALIZER.normalize_header
_promote_ultrasound_code_matrix = _NORMALIZER.promote_ultrasound_codes
_simple_cell = _NORMALIZER.simple_cell
_table_column_signature = _NORMALIZER.column_signature
_cell_source_label = _SOURCE_PROJECTOR.cell_label


@dataclass(frozen=True)
class PdfTableExtractor:
    """Detects PDF tables and reconstructs canonical table evidence."""

    def find(
        self,
        page: object,
        warnings: list[dict[str, Any]],
        page_idx: int,
    ) -> list[object]:
        return _find_tables(page, warnings, page_idx)

    def needs_text_fallback(self, tables: list[object]) -> bool:
        return _needs_pymupdf_blank_cell_fallback(tables)

    def fallback_rows(
        self,
        data: bytes,
        page_idx: int,
        tables: list[object],
    ) -> dict[int, list[list[str]]]:
        return _pymupdf_table_rows_by_pdfplumber_index(data, page_idx, tables)

    def build(
        self,
        page: object,
        tables: list[object],
        table_idx: int,
        nested: _NestedResolution,
        seen: set[int],
        *,
        cell_image_children: dict[int, dict[tuple[int, int], list[dict[str, object]]]] | None = None,
        fallback_rows: list[list[str]] | None = None,
    ) -> dict[str, object] | None:
        return _structured_table_from_pdf_table(
            page,
            tables,
            table_idx,
            nested,
            seen,
            cell_image_children=cell_image_children,
            fallback_rows=fallback_rows,
        )

    def resolve_nested(self, tables: list[object]) -> _NestedResolution:
        return _resolve_nested_tables(tables)

    def clean_text(self, text: str | None) -> str:
        return _clean_text(text)

    def clean_cell(self, cell: object) -> str:
        return _clean_cell(cell)

    def crop_text(
        self,
        page: object,
        x0: float,
        top: float,
        x1: float,
        bottom: float,
    ) -> str:
        return _crop_text(page, x0, top, x1, bottom)

    def is_artifact_text(self, text: str) -> bool:
        return _is_pdf_artifact_text(text)

    def is_continuation(
        self,
        previous: dict[str, object],
        current: dict[str, object],
    ) -> bool:
        return _is_table_continuation(previous, current)


def _find_tables(page: object, warnings: list[dict[str, Any]], page_idx: int) -> list[object]:
    try:
        return list(page.find_tables())
    except Exception as exc:
        warnings.append(
            {
                "type": "pdf_tables_skipped",
                "severity": "medium",
                "page": page_idx + 1,
                "message": str(exc),
            }
        )
        return []


def _needs_pymupdf_blank_cell_fallback(tables: list[object]) -> bool:
    for table in tables:
        rows = _table_rows(table)
        if len(rows) < 2:
            continue
        column_count = max((len(row) for row in rows), default=0)
        if column_count < 2:
            continue
        header = _pad_row(rows[0], column_count)
        for column_index in (0, column_count - 1):
            if not header[column_index].strip():
                continue
            for row in rows[1:]:
                padded = _pad_row(row, column_count)
                if not any(cell.strip() for cell in padded):
                    continue
                if not padded[column_index].strip():
                    return True
    return False


def _pymupdf_table_rows_by_pdfplumber_index(
    data: bytes,
    page_idx: int,
    pdfplumber_tables: list[object],
) -> dict[int, list[list[str]]]:
    if not pdfplumber_tables:
        return {}
    try:
        import fitz
    except ImportError:
        return {}

    try:
        with fitz.open(stream=data, filetype="pdf") as doc:
            page = doc.load_page(page_idx)
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                table_finder = page.find_tables()
            pymupdf_tables = list(getattr(table_finder, "tables", table_finder) or [])
    except Exception:
        return {}

    candidates: list[tuple[int, object, list[list[str]]]] = []
    for candidate_idx, table in enumerate(pymupdf_tables):
        rows = _pymupdf_table_rows(table)
        if rows:
            candidates.append((candidate_idx, table, rows))

    result: dict[int, list[list[str]]] = {}
    used: set[int] = set()
    for pdfplumber_idx, table in enumerate(pdfplumber_tables):
        match = _best_pymupdf_table_match(table, candidates, used)
        if match is None:
            continue
        used.add(candidates[match][0])
        result[pdfplumber_idx] = candidates[match][2]
    return result


def _pymupdf_table_rows(table: object) -> list[list[str]]:
    try:
        rows = table.extract() or []
    except Exception:
        return []
    result: list[list[str]] = []
    for row in rows:
        if row is None:
            continue
        result.append([_clean_cell(cell) for cell in row])
    return result


def _best_pymupdf_table_match(
    pdfplumber_table: object,
    candidates: list[tuple[int, object, list[list[str]]]],
    used: set[int],
) -> int | None:
    raw_rows = _table_rows(pdfplumber_table)
    if not raw_rows:
        return None
    pdfplumber_bbox = _coerce_bbox(getattr(pdfplumber_table, "bbox", None))
    best_index: int | None = None
    best_score = 0.0
    for candidate_list_index, (candidate_idx, table, rows) in enumerate(candidates):
        if candidate_idx in used:
            continue
        if not _fallback_rows_are_compatible(raw_rows, rows):
            continue
        candidate_bbox = _coerce_bbox(getattr(table, "bbox", None))
        score = _bbox_overlap_ratio(pdfplumber_bbox, candidate_bbox)
        if score > best_score:
            best_index = candidate_list_index
            best_score = score

    if best_index is None:
        compatible = [
            candidate_list_index
            for candidate_list_index, (candidate_idx, _table, rows) in enumerate(candidates)
            if candidate_idx not in used
            and _fallback_rows_are_compatible(raw_rows, rows)
        ]
        return compatible[0] if len(compatible) == 1 else None
    return best_index if best_score >= 0.5 else None


def _is_table_continuation(
    previous: dict[str, object],
    current: dict[str, object],
) -> bool:
    previous_signature = _table_column_signature(previous)
    return bool(previous_signature) and previous_signature == _table_column_signature(current)


def _structured_table_from_pdf_table(
    page: object,
    tables: list[object],
    table_idx: int,
    nested: _NestedResolution,
    seen: set[int],
    *,
    cell_image_children: dict[int, dict[tuple[int, int], list[dict[str, object]]]] | None = None,
    fallback_rows: list[list[str]] | None = None,
) -> dict[str, object] | None:
    if table_idx in seen:
        return None
    table = tables[table_idx]
    raw_rows = _table_rows(table)
    if not raw_rows:
        return None
    raw_rows = _fill_blank_cells_from_fallback_rows(raw_rows, fallback_rows)
    raw_rows = _trim_empty_trailing_columns(raw_rows)
    column_count = max((len(row) for row in raw_rows), default=0)
    if column_count == 0:
        return _structured_table_content(columns=[], rows=[])
    normalized_rows = [_pad_row(row, column_count) for row in raw_rows]
    header_depth = _header_depth(normalized_rows)

    columns = _columns_from_header_rows(normalized_rows[:header_depth], column_count)
    child_map = nested.children.get(table_idx, {})
    image_child_map = (cell_image_children or {}).get(table_idx, {})
    cell_spans = _table_cell_spans(
        table,
        row_count=len(normalized_rows),
        column_count=column_count,
        raw_rows=normalized_rows,
    )
    header_rows: list[dict[str, object]] = []
    for header_index, raw_row in enumerate(normalized_rows[:header_depth]):
        header_cells = _row_evidence_cells(
            page,
            tables,
            table_idx,
            row_index=header_index,
            raw_row=raw_row,
            columns=columns,
            child_map=child_map,
            cell_spans=cell_spans,
            nested=nested,
            seen=seen | {table_idx},
            cell_image_children=cell_image_children or {},
            image_child_map=image_child_map,
        )
        if header_cells:
            header_rows.append({"index": header_index + 1, "cells": header_cells})

    rows: list[dict[str, object]] = []
    for raw_index, raw_row in enumerate(normalized_rows[header_depth:], start=header_depth):
        cells = _row_evidence_cells(
            page,
            tables,
            table_idx,
            row_index=raw_index,
            raw_row=raw_row,
            columns=columns,
            child_map=child_map,
            cell_spans=cell_spans,
            nested=nested,
            seen=seen | {table_idx},
            cell_image_children=cell_image_children or {},
            image_child_map=image_child_map,
        )
        if not any(str(cell["text"]).strip() or cell["children"] for cell in cells):
            continue
        rows.append({"index": len(rows) + 1, "cells": cells})

    result = _structured_table_content(
        columns=columns,
        rows=rows,
        header_rows=header_rows if header_rows else None,
    )
    _split_dotted_subrows(result, page, table, header_depth)
    _repair_table_of_contents(result, page)
    _merge_blank_header_rowspans(result)
    _PDF_TABLE_NORMALIZER.normalize(result)
    return result


def _split_dotted_subrows(
    table: dict[str, object],
    page: object,
    pdf_table: object,
    header_depth: int,
) -> None:
    columns = table.get("columns")
    rows = table.get("rows")
    if not isinstance(columns, list) or not isinstance(rows, list):
        return

    column_count = len(columns)
    expanded_rows: list[dict[str, object]] = []
    changed = False
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        pdf_row_index = header_depth + row_index
        row_cells = _pad_row_cells(_row_cells(pdf_table, pdf_row_index), column_count)
        separators = _dotted_subrow_separators(page, row_cells)
        if not separators:
            copied = dict(row)
            copied["index"] = len(expanded_rows) + 1
            expanded_rows.append(copied)
            continue

        split_rows = _split_row_by_subrow_separators(row, page, row_cells, separators)
        if len(split_rows) <= 1:
            copied = dict(row)
            copied["index"] = len(expanded_rows) + 1
            expanded_rows.append(copied)
            continue
        changed = True
        for split_row in split_rows:
            split_row["index"] = len(expanded_rows) + 1
            expanded_rows.append(split_row)

    if changed:
        table["rows"] = expanded_rows


def _dotted_subrow_separators(
    page: object,
    row_cells: list[tuple[float, float, float, float] | None],
    *,
    y_tolerance: float = 1.0,
    min_coverage_ratio: float = 0.55,
) -> list[tuple[float, list[tuple[float, float]]]]:
    row_bbox = _union_bboxes([cell for cell in row_cells if cell is not None])
    if row_bbox is None:
        return []
    row_left, row_top, row_right, row_bottom = row_bbox
    row_width = row_right - row_left
    if row_width <= 0 or row_bottom - row_top <= 12:
        return []

    groups: list[tuple[float, list[tuple[float, float]]]] = []
    for line in getattr(page, "lines", []):
        interval = _horizontal_line_interval(line)
        if interval is None:
            continue
        top, x0, x1 = interval
        if top <= row_top + 4 or top >= row_bottom - 4:
            continue
        if x1 <= row_left or x0 >= row_right:
            continue
        clipped = (max(x0, row_left), min(x1, row_right))
        for index, (group_y, intervals) in enumerate(groups):
            if abs(top - group_y) <= y_tolerance:
                intervals.append(clipped)
                groups[index] = ((group_y + top) / 2, intervals)
                break
        else:
            groups.append((top, [clipped]))

    separators: list[tuple[float, list[tuple[float, float]]]] = []
    for top, intervals in groups:
        coverage = _intervals_coverage(intervals)
        span = _intervals_span(intervals)
        crossed_columns = _separator_crossed_column_count(top, intervals, row_cells)
        if (
            max(coverage, span) / row_width >= min_coverage_ratio
            and 2 <= crossed_columns < len(row_cells)
        ):
            separators.append((top, intervals))
    return sorted(separators, key=lambda separator: separator[0])


def _horizontal_line_interval(line: object) -> tuple[float, float, float] | None:
    if not isinstance(line, Mapping):
        return None
    try:
        top = float(line.get("top", 0.0))
        bottom = float(line.get("bottom", top))
        x0 = float(line.get("x0", 0.0))
        x1 = float(line.get("x1", 0.0))
    except (TypeError, ValueError):
        return None
    if abs(bottom - top) > 1.0 or x1 <= x0:
        return None
    return (top, x0, x1)


def _union_bboxes(
    bboxes: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float] | None:
    if not bboxes:
        return None
    return (
        min(bbox[0] for bbox in bboxes),
        min(bbox[1] for bbox in bboxes),
        max(bbox[2] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes),
    )


def _intervals_coverage(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
            continue
        merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def _intervals_span(intervals: list[tuple[float, float]]) -> float:
    intervals = [(start, end) for start, end in intervals if end > start]
    if not intervals:
        return 0.0
    return max(end for _, end in intervals) - min(start for start, _ in intervals)


def _split_row_by_subrow_separators(
    row: dict[str, object],
    page: object,
    row_cells: list[tuple[float, float, float, float] | None],
    separators: list[tuple[float, list[tuple[float, float]]]],
) -> list[dict[str, object]]:
    row_bbox = _union_bboxes([cell for cell in row_cells if cell is not None])
    if row_bbox is None:
        return [row]
    band_edges = [row_bbox[1], *[separator[0] for separator in separators], row_bbox[3]]
    if len(band_edges) < 3:
        return [row]

    cells = row.get("cells")
    if not isinstance(cells, list):
        return [row]
    if any(
        isinstance(cell, Mapping) and cell.get("children")
        for cell in cells
    ):
        return [row]
    cells_by_column = {
        str(cell.get("column_id")): cell
        for cell in cells
        if isinstance(cell, Mapping)
    }

    split_rows = [{"index": 0, "cells": []} for _ in range(len(band_edges) - 1)]
    for column_index, cell_bbox in enumerate(row_cells):
        column_id = f"c{column_index + 1}"
        cell = cells_by_column.get(column_id)
        if not isinstance(cell, Mapping):
            continue
        if cell_bbox is None or not _cell_is_crossed_by_separator(cell_bbox, separators):
            copied = dict(cell)
            copied["rowspan"] = int(copied.get("rowspan", 1) or 1) + len(split_rows) - 1
            split_rows[0]["cells"].append(copied)
            continue

        for band_index, (band_top, band_bottom) in enumerate(
            zip(band_edges[:-1], band_edges[1:], strict=True)
        ):
            text = _join_lines(
                _crop_text(page, cell_bbox[0], band_top, cell_bbox[2], band_bottom)
            )
            if not text:
                continue
            split_rows[band_index]["cells"].append(
                {
                    "column_id": column_id,
                    "text": text,
                    "rowspan": 1,
                    "colspan": int(cell.get("colspan", 1) or 1),
                    "children": [],
                }
            )

    return [
        split_row
        for split_row in split_rows
        if any(_cell_has_content(cell) for cell in split_row["cells"])
    ]


def _cell_is_crossed_by_separator(
    cell_bbox: tuple[float, float, float, float],
    separators: list[tuple[float, list[tuple[float, float]]]],
    *,
    tolerance: float = 1.0,
    min_horizontal_ratio: float = 0.5,
) -> bool:
    return any(
        cell_bbox[1] + tolerance < separator < cell_bbox[3] - tolerance
        and _intervals_overlap_span_ratio(cell_bbox, intervals) >= min_horizontal_ratio
        for separator, intervals in separators
    )


def _intervals_overlap_span_ratio(
    cell_bbox: tuple[float, float, float, float],
    intervals: list[tuple[float, float]],
) -> float:
    cell_left, _, cell_right, _ = cell_bbox
    cell_width = cell_right - cell_left
    if cell_width <= 0:
        return 0.0
    clipped = [
        (max(start, cell_left), min(end, cell_right))
        for start, end in intervals
        if end > cell_left and start < cell_right
    ]
    return _intervals_span(clipped) / cell_width


def _separator_crossed_column_count(
    separator: float,
    intervals: list[tuple[float, float]],
    row_cells: list[tuple[float, float, float, float] | None],
    *,
    tolerance: float = 1.0,
) -> int:
    return sum(
        1
        for cell_bbox in row_cells
        if cell_bbox is not None
        and _cell_is_crossed_by_separator(
            cell_bbox,
            [(separator, intervals)],
            tolerance=tolerance,
        )
    )


def _merge_blank_header_rowspans(table: dict[str, object]) -> None:
    header_rows = table.get("header_rows")
    if not isinstance(header_rows, list) or len(header_rows) < 2:
        return

    previous_cells_by_column: dict[str, dict[str, object]] = {}
    for header_row in header_rows:
        cells = header_row.get("cells") if isinstance(header_row, Mapping) else None
        if not isinstance(cells, list):
            previous_cells_by_column = {}
            continue

        next_cells: list[dict[str, object]] = []
        current_cells_by_column: dict[str, dict[str, object]] = {}
        for cell in cells:
            if not isinstance(cell, Mapping):
                continue
            column_id = str(cell.get("column_id", ""))
            previous = previous_cells_by_column.get(column_id)
            if _is_blank_header_slot(cell) and _can_absorb_blank_header_slot(previous):
                previous["rowspan"] = int(previous.get("rowspan", 1) or 1) + 1
                continue
            next_cells.append(cell)
            if column_id:
                current_cells_by_column[column_id] = cell

        header_row["cells"] = next_cells
        previous_cells_by_column = current_cells_by_column


def _is_blank_header_slot(cell: dict[str, object]) -> bool:
    return (
        not str(cell.get("text", "")).strip()
        and int(cell.get("rowspan", 1) or 1) == 1
        and int(cell.get("colspan", 1) or 1) == 1
    )


def _can_absorb_blank_header_slot(cell: dict[str, object] | None) -> bool:
    return (
        isinstance(cell, Mapping)
        and bool(str(cell.get("text", "")).strip())
        and int(cell.get("rowspan", 1) or 1) == 1
        and int(cell.get("colspan", 1) or 1) == 1
    )


def _repair_table_of_contents(table: dict[str, object], page: object) -> None:
    if not _is_table_of_contents(table):
        return
    rows = table.get("rows")
    if not isinstance(rows, list) or not rows:
        return

    entries = _table_of_contents_entries(page)
    if len(entries) < len(rows):
        return

    entry_index = 0
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        cells = row.get("cells")
        if not isinstance(cells, list) or len(cells) < 3:
            continue
        title = str(cells[1].get("text", "")) if isinstance(cells[1], Mapping) else ""
        entry = _matching_table_of_contents_entry(title, entries, entry_index)
        if entry is None:
            continue
        entry_index = entries.index(entry, entry_index) + 1
        number, _title, page_number = entry
        if isinstance(cells[0], Mapping) and not str(cells[0].get("text", "")).strip():
            cells[0]["text"] = number
        if isinstance(cells[-1], Mapping) and not str(cells[-1].get("text", "")).strip():
            cells[-1]["text"] = page_number


def _table_of_contents_entries(page: object) -> list[tuple[str, str, str]]:
    try:
        text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
    except TypeError:
        text = page.extract_text() or ""
    except Exception:
        return []

    entries: list[tuple[str, str, str]] = []
    for line in text.splitlines():
        match = re.match(r"^\s*(\d{1,3})\s+(.+?)\s+(\d{1,4})\s*$", line)
        if match is not None:
            entries.append(tuple(part.strip() for part in match.groups()))
    return entries


def _matching_table_of_contents_entry(
    title: str,
    entries: list[tuple[str, str, str]],
    start_index: int,
) -> tuple[str, str, str] | None:
    normalized_title = _compact_text(title)
    for entry in entries[start_index:]:
        if not normalized_title or normalized_title == _compact_text(entry[1]):
            return entry
    return None


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _pad_row(row: list[str], column_count: int) -> list[str]:
    return [*row, *([""] * max(0, column_count - len(row)))]


def _header_depth(rows: list[list[str]]) -> int:
    if len(rows) < 2:
        return 1
    first = rows[0]
    second = rows[1]
    if not any(not cell.strip() for cell in first):
        return 1
    first_count = _non_empty_cell_count(first)
    second_count = _non_empty_cell_count(second)
    if second_count < 2:
        return 1
    if first_count < 2 and not _is_single_group_header_row(first):
        return 1
    if not _is_concise_header_row(second):
        return 1
    return 2


def _is_single_group_header_row(row: list[str]) -> bool:
    return (
        len(row) >= 3
        and _non_empty_cell_count(row) == 1
        and _is_concise_header_row(row)
    )


def _non_empty_cell_count(row: list[str]) -> int:
    return sum(1 for cell in row if cell.strip())


def _is_concise_header_row(row: list[str], max_cell_chars: int = 32) -> bool:
    values = [cell.strip() for cell in row if cell.strip()]
    return bool(values) and all(len(value) <= max_cell_chars for value in values)


def _columns_from_header_rows(
    header_rows: list[list[str]],
    column_count: int,
) -> list[dict[str, str]]:
    group_labels = (
        _forward_fill_header_row(header_rows[0], column_count)
        if len(header_rows) > 1
        else _pad_row(header_rows[0], column_count)
    )
    columns: list[dict[str, str]] = []
    for index in range(column_count):
        parts: list[str] = []
        for row_index, row in enumerate(header_rows):
            value = _normalize_header_text(
                group_labels[index] if row_index == 0 and len(header_rows) > 1 else row[index]
            )
            if value and value not in parts:
                parts.append(value)
        columns.append({"id": f"c{index + 1}", "text": " / ".join(parts)})
    return columns


def _forward_fill_header_row(row: list[str], column_count: int) -> list[str]:
    result: list[str] = []
    current = ""
    for value in _pad_row(row, column_count):
        normalized = _normalize_header_text(value)
        if normalized:
            current = normalized
        result.append(current)
    return result


def _table_cell_spans(
    table: object,
    *,
    row_count: int,
    column_count: int,
    raw_rows: list[list[str]] | None = None,
) -> _TableCellSpans:
    cell_rows = [
        _pad_row_cells(_row_cells(table, row_index), column_count)
        for row_index in range(row_count)
    ]
    column_boundaries = _table_column_boundaries(cell_rows, column_count)
    row_bottoms = [_row_bottom(cells) for cells in cell_rows]
    spans: dict[tuple[int, int], tuple[int, int]] = {}
    covered: set[tuple[int, int]] = set()
    for row_index, row in enumerate(cell_rows):
        for column_index, bbox in enumerate(row):
            if (row_index, column_index) in covered:
                continue
            if bbox is None:
                spans[(row_index, column_index)] = (1, 1)
                continue
            colspan = _pdf_cell_colspan(
                row,
                column_index,
                column_count,
                bbox,
                column_boundaries,
            )
            colspan = _limit_span_colspan_by_extracted_text(
                raw_rows,
                row_index,
                column_index,
                colspan,
            )
            rowspan = _pdf_cell_rowspan(
                cell_rows,
                row_bottoms,
                row_index,
                column_index,
                colspan,
                bbox,
            )
            rowspan = _limit_span_rowspan_by_extracted_text(
                raw_rows,
                row_index,
                column_index,
                rowspan,
                colspan,
            )
            spans[(row_index, column_index)] = (rowspan, colspan)
            for covered_row in range(row_index, row_index + rowspan):
                for covered_column in range(column_index, column_index + colspan):
                    if (covered_row, covered_column) == (row_index, column_index):
                        continue
                    covered_bbox = cell_rows[covered_row][covered_column]
                    if _slot_is_covered_by_cell(covered_bbox, bbox):
                        covered.add((covered_row, covered_column))
    return _TableCellSpans(spans=spans, covered=covered)


def _limit_span_colspan_by_extracted_text(
    raw_rows: list[list[str]] | None,
    row_index: int,
    column_index: int,
    colspan: int,
) -> int:
    if colspan <= 1:
        return colspan
    for covered_column in range(column_index + 1, column_index + colspan):
        if _covered_slot_has_distinct_extracted_text(
            raw_rows,
            row_index,
            column_index,
            row_index,
            covered_column,
        ):
            return covered_column - column_index
    return colspan


def _limit_span_rowspan_by_extracted_text(
    raw_rows: list[list[str]] | None,
    row_index: int,
    column_index: int,
    rowspan: int,
    colspan: int,
) -> int:
    if rowspan <= 1:
        return rowspan
    for covered_row in range(row_index + 1, row_index + rowspan):
        for covered_column in range(column_index, column_index + colspan):
            if _covered_slot_has_distinct_extracted_text(
                raw_rows,
                row_index,
                column_index,
                covered_row,
                covered_column,
            ):
                return covered_row - row_index
    return rowspan


def _covered_slot_has_distinct_extracted_text(
    raw_rows: list[list[str]] | None,
    anchor_row: int,
    anchor_column: int,
    covered_row: int,
    covered_column: int,
) -> bool:
    if raw_rows is None:
        return False
    covered_text = _span_extracted_text(raw_rows, covered_row, covered_column)
    if not covered_text:
        return False
    anchor_text = _span_extracted_text(raw_rows, anchor_row, anchor_column)
    return covered_text != anchor_text


def _span_extracted_text(
    raw_rows: list[list[str]],
    row_index: int,
    column_index: int,
) -> str:
    if row_index >= len(raw_rows) or column_index >= len(raw_rows[row_index]):
        return ""
    return _normalize_header_text(raw_rows[row_index][column_index])


def _pad_row_cells(
    row: list[tuple[float, float, float, float] | None],
    column_count: int,
) -> list[tuple[float, float, float, float] | None]:
    return [*row[:column_count], *([None] * max(0, column_count - len(row)))]


def _row_bottom(
    cells: list[tuple[float, float, float, float] | None],
) -> float | None:
    bottoms = [cell[3] for cell in cells if cell is not None]
    return max(bottoms) if bottoms else None


def _table_column_boundaries(
    cell_rows: list[list[tuple[float, float, float, float] | None]],
    column_count: int,
) -> list[float]:
    positions = [
        position
        for row in cell_rows
        for cell in row
        if cell is not None
        for position in (cell[0], cell[2])
    ]
    boundaries = _cluster_positions(positions)
    if len(boundaries) < column_count + 1:
        return []
    return boundaries


def _cluster_positions(
    positions: list[float],
    *,
    tolerance: float = 2.0,
) -> list[float]:
    clustered: list[float] = []
    for position in sorted(positions):
        if not clustered or abs(position - clustered[-1]) > tolerance:
            clustered.append(position)
            continue
        clustered[-1] = (clustered[-1] + position) / 2
    return clustered


def _pdf_cell_colspan(
    row: list[tuple[float, float, float, float] | None],
    column_index: int,
    column_count: int,
    bbox: tuple[float, float, float, float],
    column_boundaries: list[float],
) -> int:
    colspan = 1
    while column_index + colspan < column_count and _slot_is_covered_by_cell(
        row[column_index + colspan],
        bbox,
    ):
        if column_boundaries and not _cell_reaches_column_end(
            bbox,
            column_boundaries,
            column_index + colspan,
        ):
            break
        colspan += 1
    return colspan


def _cell_reaches_column_end(
    bbox: tuple[float, float, float, float],
    column_boundaries: list[float],
    column_index: int,
    *,
    tolerance: float = 2.0,
) -> bool:
    boundary_index = column_index + 1
    return (
        boundary_index >= len(column_boundaries)
        or bbox[2] >= column_boundaries[boundary_index] - tolerance
    )


def _pdf_cell_rowspan(
    cell_rows: list[list[tuple[float, float, float, float] | None]],
    row_bottoms: list[float | None],
    row_index: int,
    column_index: int,
    colspan: int,
    bbox: tuple[float, float, float, float],
    *,
    tolerance: float = 2.0,
) -> int:
    rowspan = 1
    for next_row in range(row_index + 1, len(cell_rows)):
        row_bottom = row_bottoms[next_row]
        if row_bottom is None or bbox[3] < row_bottom - tolerance:
            break
        if not all(
            _slot_is_covered_by_cell(cell_rows[next_row][covered_column], bbox)
            for covered_column in range(column_index, column_index + colspan)
        ):
            break
        rowspan += 1
    return rowspan


def _slot_is_covered_by_cell(
    slot: tuple[float, float, float, float] | None,
    bbox: tuple[float, float, float, float],
) -> bool:
    return slot is None or _bbox_near_equal(slot, bbox)


def _row_evidence_cells(
    page: object,
    tables: list[object],
    table_idx: int,
    *,
    row_index: int,
    raw_row: list[str],
    columns: list[dict[str, str]],
    child_map: dict[tuple[int, int], list[int]],
    cell_spans: _TableCellSpans,
    nested: _NestedResolution,
    seen: set[int],
    cell_image_children: dict[int, dict[tuple[int, int], list[dict[str, object]]]],
    image_child_map: dict[tuple[int, int], list[dict[str, object]]],
) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    row_cells = _row_cells(tables[table_idx], row_index)
    for column_index, column in enumerate(columns):
        if (row_index, column_index) in cell_spans.covered:
            continue
        rowspan, colspan = cell_spans.spans.get((row_index, column_index), (1, 1))
        text = raw_row[column_index] if column_index < len(raw_row) else ""
        child_indices = child_map.get((row_index, column_index), [])
        table_children = [
            child
            for child in (
                _nested_table_child(
                    page,
                    tables,
                    child_idx,
                    nested,
                    seen,
                    cell_image_children=cell_image_children,
                )
                for child_idx in child_indices
            )
            if child is not None
        ]
        children = [
            *table_children,
            *image_child_map.get((row_index, column_index), []),
        ]
        children = _merge_nested_table_children(children)
        if table_children:
            cell_bbox = row_cells[column_index] if column_index < len(row_cells) else None
            if cell_bbox is not None:
                text = _rebuild_cell_text(
                    page,
                    cell_bbox,
                    [tables[child_idx] for child_idx in child_indices],
                )
            else:
                text = ""
        cells.append(
            {
                "column_id": column["id"],
                "text": text,
                "rowspan": rowspan,
                "colspan": colspan,
                "children": children,
            }
        )
    return cells


def _nested_table_child(
    page: object,
    tables: list[object],
    table_idx: int,
    nested: _NestedResolution,
    seen: set[int],
    *,
    cell_image_children: dict[int, dict[tuple[int, int], list[dict[str, object]]]] | None = None,
) -> dict[str, object] | None:
    structured = _structured_table_from_pdf_table(
        page,
        tables,
        table_idx,
        nested,
        seen,
        cell_image_children=cell_image_children,
    )
    if structured is None:
        return None
    return {
        "type": "table",
        "format": "structured_table",
        "content": structured,
    }


def _table_rows(table: object) -> list[list[str]]:
    try:
        rows = table.extract() or []
    except Exception:
        return []
    result: list[list[str]] = []
    for row in rows:
        if row is None:
            continue
        result.append([_clean_cell(cell) for cell in row])
    return result


def _fill_blank_cells_from_fallback_rows(
    rows: list[list[str]],
    fallback_rows: list[list[str]] | None,
) -> list[list[str]]:
    if fallback_rows is None or not _fallback_rows_are_compatible(rows, fallback_rows):
        return rows

    column_count = max((len(row) for row in rows), default=0)
    merged: list[list[str]] = []
    for row, fallback_row in zip(rows, fallback_rows, strict=True):
        padded = _pad_row(row, column_count)
        fallback_padded = _pad_row(
            [_clean_cell(cell) for cell in fallback_row],
            column_count,
        )
        merged.append(
            [
                padded[index] if padded[index].strip() else fallback_padded[index]
                for index in range(column_count)
            ]
        )
    return merged


def _fallback_rows_are_compatible(
    rows: list[list[str]],
    fallback_rows: list[list[str]],
) -> bool:
    if not rows or len(rows) != len(fallback_rows):
        return False
    column_count = max((len(row) for row in rows), default=0)
    fallback_column_count = max((len(row) for row in fallback_rows), default=0)
    if column_count == 0 or column_count != fallback_column_count:
        return False
    return _table_headers_are_compatible(
        _pad_row(rows[0], column_count),
        _pad_row([_clean_cell(cell) for cell in fallback_rows[0]], column_count),
    )


def _table_headers_are_compatible(
    headers: list[str],
    fallback_headers: list[str],
) -> bool:
    shared = 0
    for header, fallback_header in zip(headers, fallback_headers, strict=True):
        normalized = _normalize_header_text(header)
        fallback_normalized = _normalize_header_text(fallback_header)
        if not normalized or not fallback_normalized:
            continue
        if normalized != fallback_normalized:
            return False
        shared += 1
    return shared > 0


def _trim_empty_trailing_columns(rows: list[list[str]]) -> list[list[str]]:
    trimmed = [list(row) for row in rows]
    while True:
        column_count = max((len(row) for row in trimmed), default=0)
        if column_count <= 1:
            return trimmed
        last_index = column_count - 1
        if any(last_index < len(row) and str(row[last_index]).strip() for row in trimmed):
            return trimmed
        trimmed = [
            row[:last_index] if last_index < len(row) else list(row)
            for row in trimmed
        ]


def _row_cells(table: object, row_index: int) -> list[tuple[float, float, float, float] | None]:
    rows = getattr(table, "rows", [])
    if row_index >= len(rows):
        return []
    return list(getattr(rows[row_index], "cells", []) or [])


def _title_table_text(table: dict[str, object]) -> str | None:
    if table["rows"]:
        return None
    header_rows = table.get("header_rows")
    if not isinstance(header_rows, list) or len(header_rows) != 1:
        return None
    cells = header_rows[0].get("cells")
    if not isinstance(cells, list) or not cells:
        return None
    if len(cells) > 3:
        return None
    parts: list[str] = []
    for cell in cells:
        if not isinstance(cell, Mapping) or cell.get("children"):
            return None
        text = str(cell.get("text", "")).strip()
        if text:
            parts.append(text)
    if not parts:
        return None
    text = _join_pdf_text_fragments(parts).strip()
    if len(cells) > 1 and len(text) > 40:
        return None
    return text or None


def _single_line_header_only_table_text(
    pdf_table: object,
    table: dict[str, object],
) -> str | None:
    if table["rows"]:
        return None
    bbox = getattr(pdf_table, "bbox", None)
    if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
        return None
    if float(bbox[3]) - float(bbox[1]) > 24:
        return None
    header_rows = table.get("header_rows")
    if not isinstance(header_rows, list) or len(header_rows) != 1:
        return None
    cells = header_rows[0].get("cells")
    if not isinstance(cells, list) or not cells or len(cells) > 3:
        return None
    parts: list[str] = []
    for cell in cells:
        if not isinstance(cell, Mapping) or cell.get("children"):
            return None
        text = str(cell.get("text", "")).strip()
        if text:
            parts.append(text)
    if not parts:
        return None
    return _join_pdf_text_fragments(parts).strip() or None


def _is_empty_single_cell_table_artifact(table: dict[str, object]) -> bool:
    columns = table.get("columns")
    if not isinstance(columns, list) or len(columns) != 1:
        return False
    all_rows: list[object] = []
    header_rows = table.get("header_rows")
    rows = table.get("rows")
    if isinstance(header_rows, list):
        all_rows.extend(header_rows)
    if isinstance(rows, list):
        all_rows.extend(rows)
    if len(all_rows) != 1:
        return False
    row = all_rows[0]
    if not isinstance(row, Mapping):
        return False
    cells = row.get("cells")
    if not isinstance(cells, list) or len(cells) != 1:
        return False
    cell = cells[0]
    if not isinstance(cell, Mapping):
        return False
    return not str(cell.get("text", "")).strip() and not cell.get("children")


def _join_pdf_text_fragments(parts: list[str]) -> str:
    if not parts:
        return ""
    out = parts[0].strip()
    for part in parts[1:]:
        current = part.strip()
        if not current:
            continue
        if _should_join_pdf_text_fragment(out, current):
            out = out.rstrip() + current.lstrip()
        else:
            out = out.rstrip() + " " + current.lstrip()
    return out


def _should_join_pdf_text_fragment(previous: str, current: str) -> bool:
    previous_token = previous.rstrip().rsplit(" ", 1)[-1]
    current_token = current.lstrip().split(" ", 1)[0]
    return (
        bool(previous_token)
        and bool(current_token)
        and len(current_token) <= 1
        and _CJK.search(current_token[0]) is not None
        and re.search(r"[0-9가-힣]$", previous_token) is not None
    )


def _resolve_nested_tables(tables: list[object]) -> _NestedResolution:
    contained: dict[int, tuple[int, int, int, float]] = {}
    for sub_idx, sub_table in enumerate(tables):
        sub_bbox = sub_table.bbox
        best: tuple[int, int, int, float] | None = None
        for parent_idx, parent_table in enumerate(tables):
            if parent_idx == sub_idx:
                continue
            parent_bbox = parent_table.bbox
            if _bbox_near_equal(parent_bbox, sub_bbox):
                continue
            if _bbox_area(parent_bbox) <= _bbox_area(sub_bbox):
                continue
            for row_idx, row in enumerate(getattr(parent_table, "rows", [])):
                for col_idx, cell in enumerate(getattr(row, "cells", []) or []):
                    if cell is None:
                        continue
                    if _bbox_in_cell(sub_bbox, cell):
                        area = _bbox_area(cell)
                        if best is None or area < best[3]:
                            best = (parent_idx, row_idx, col_idx, area)
        if best is not None:
            contained[sub_idx] = best

    children: dict[int, dict[tuple[int, int], list[int]]] = {}
    for sub_idx, (parent_idx, row_idx, col_idx, _area) in contained.items():
        children.setdefault(parent_idx, {}).setdefault((row_idx, col_idx), []).append(
            sub_idx
        )
    for cell_map in children.values():
        for child_indices in cell_map.values():
            child_indices.sort(key=lambda idx: float(tables[idx].bbox[1]))

    return _NestedResolution(suppressed=set(contained), children=children)


def _rebuild_cell_text(
    page: object,
    cell_bbox: tuple[float, float, float, float],
    child_tables: list[object],
) -> str:
    x0, top, x1, bottom = cell_bbox
    parts: list[str] = []
    y_cursor = top
    for child in sorted(child_tables, key=lambda table: float(table.bbox[1])):
        child_top = float(child.bbox[1])
        child_bottom = float(child.bbox[3])
        band = _crop_text(page, x0, y_cursor, x1, child_top)
        if band:
            parts.append(band)
        y_cursor = max(y_cursor, child_bottom)
    tail = _crop_text(page, x0, y_cursor, x1, bottom)
    if tail:
        parts.append(tail)
    return " ".join(parts).strip()


def _crop_text(page: object, x0: float, top: float, x1: float, bottom: float) -> str:
    if bottom - top <= 2 or x1 - x0 <= 2:
        return ""
    try:
        crop = page.crop((x0, top, x1, bottom))
        try:
            text = crop.extract_text(x_tolerance=3, y_tolerance=3) or ""
        except TypeError:
            text = crop.extract_text() or ""
    except Exception:
        return ""
    return _clean_text(text)


def _clean_text(text: str | None) -> str:
    stripped = _strip_page_numbers(text or "")
    return "\n".join(line.strip() for line in stripped.splitlines() if line.strip()).strip()


def _is_pdf_artifact_text(text: str) -> bool:
    return bool(re.fullmatch(r"INSID[A-Za-z0-9_:.-]+", text.strip()))


def _strip_page_numbers(text: str) -> str:
    return _PAGE_NUM_RE.sub("", text)


def _join_lines(text: str) -> str:
    parts = text.split("\n")
    if len(parts) == 1:
        return text
    out = parts[0]
    for part in parts[1:]:
        if out and _CJK.search(out[-1]) and part and _CJK.search(part[0]):
            out = out + part
        else:
            out = out.rstrip() + " " + part.lstrip()
    return out


def _join_cell_lines(text: str) -> str:
    parts = text.split("\n")
    if len(parts) == 1:
        return text
    out = parts[0]
    for part in parts[1:]:
        if _is_short_line_continuation(out, part):
            out = out.rstrip() + part.lstrip()
        else:
            out = out.rstrip() + " " + part.lstrip()
    return out


def _is_short_line_continuation(previous: str, current: str) -> bool:
    previous_token = previous.rstrip().rsplit(" ", 1)[-1]
    current_token = current.lstrip().split(" ", 1)[0]
    return (
        bool(previous_token)
        and bool(current_token)
        and _CJK.search(previous_token[-1]) is not None
        and _CJK.search(current_token[0]) is not None
        and (len(previous_token) <= 1 or len(current_token) <= 1)
    )


def _clean_cell(cell: object) -> str:
    if cell is None:
        return ""
    text = _join_cell_lines(str(cell))
    text = re.sub(
        r"(?<![가-힣])([가-힣])( [가-힣])+(?![가-힣])",
        lambda match: match.group(0).replace(" ", ""),
        text,
    )
    return text.strip()
