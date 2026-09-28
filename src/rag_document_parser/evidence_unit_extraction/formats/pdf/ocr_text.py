from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ....models import StructuredTableContent
from ...schema import (
    structured_table as _structured_table_content,
    table_column,
    table_row,
)
from .models import Segment as _Segment
from .table_extraction import PdfTableExtractor
from .table_normalization import PdfTableNormalizer

_TABLE_EXTRACTOR = PdfTableExtractor()
_NORMALIZER = PdfTableNormalizer()
_clean_text = _TABLE_EXTRACTOR.clean_text
_simple_cell = _NORMALIZER.simple_cell


@dataclass(frozen=True)
class _UnstructuredPipeTable:
    text: str
    reason: str
    line_column_counts: list[int]


@dataclass(frozen=True)
class PdfOcrTextParser:
    """Converts OCR text into ordered text and structured-table segments."""

    def parse(
        self,
        text: str,
        page_number: int,
    ) -> tuple[list[_Segment], list[dict[str, Any]]]:
        return _ocr_text_segments(text, page_number)


def _ocr_text_segments(
    text: str,
    page_number: int,
) -> tuple[list[_Segment], list[dict[str, Any]]]:
    segments: list[_Segment] = []
    warnings: list[dict[str, Any]] = []
    part_index = 0
    for kind, payload in _ocr_text_parts(text):
        if isinstance(payload, _UnstructuredPipeTable):
            segments.append(
                _Segment(
                    top=part_index * 0.001,
                    bottom=part_index * 0.001,
                    kind="text",
                    payload=payload.text,
                    page=page_number,
                    metadata={"ocr": True, "table_fallback": True},
                )
            )
            warnings.append(
                {
                    "type": "pdf_ocr_table_unstructured",
                    "severity": "medium",
                    "page": page_number,
                    "reason": payload.reason,
                    "line_column_counts": payload.line_column_counts,
                    "message": (
                        "OCR pipe table structure was ambiguous; the original "
                        "table text was preserved without assigning column meanings."
                    ),
                }
            )
        elif kind in {"table", "table_pipe", "table_text"}:
            table = payload
            segments.append(
                _Segment(
                    top=part_index * 0.001,
                    bottom=part_index * 0.001,
                    kind="table",
                    payload=table,
                    page=page_number,
                    metadata={"ocr": True, "confidence": "medium"},
                )
            )
            table_type = "text table" if kind == "table_text" else "pipe table"
            warnings.append(
                {
                    "type": "pdf_ocr_table_inferred",
                    "severity": "low",
                    "page": page_number,
                    "message": (
                        "OCR text was converted to structured_table from a "
                        f"detected {table_type}."
                    ),
                }
            )
        else:
            segments.append(
                _Segment(
                    top=part_index * 0.001,
                    bottom=part_index * 0.001,
                    kind="text",
                    payload=str(payload),
                    page=page_number,
                    metadata={"ocr": True},
                )
            )
        part_index += 1
    return segments, warnings


def _ocr_text_parts(text: str) -> list[tuple[str, object]]:
    lines = text.splitlines()
    parts: list[tuple[str, object]] = []
    paragraph: list[str] = []
    index = 0

    def flush_paragraph() -> None:
        if not paragraph:
            return
        body = _clean_text("\n".join(paragraph))
        paragraph.clear()
        if body:
            parts.append(("text", body))

    while index < len(lines):
        if _is_pipe_table_row(lines[index]):
            table_lines: list[str] = []
            while index < len(lines) and _is_pipe_table_row(lines[index]):
                table_lines.append(lines[index])
                index += 1
            if any(_is_pipe_separator_line(line) for line in table_lines):
                flush_paragraph()
                parts.extend(_pipe_block_parts(table_lines))
            else:
                paragraph.extend(table_lines)
            continue
        if _looks_like_aligned_table_start(lines, index):
            flush_paragraph()
            table_lines = [lines[index]]
            column_count = len(_split_aligned_table_row(lines[index]))
            index += 1
            while index < len(lines):
                row = _split_aligned_table_row(lines[index])
                if len(row) != column_count:
                    break
                table_lines.append(lines[index])
                index += 1
            table = _structured_table_from_aligned_lines(table_lines)
            if table is not None:
                parts.append(("table_text", table))
                continue
            paragraph.extend(table_lines)
            continue
        paragraph.append(lines[index])
        index += 1

    flush_paragraph()
    return parts


def _looks_like_pipe_table_start(lines: list[str], index: int) -> bool:
    return (
        index + 1 < len(lines)
        and _is_pipe_table_row(lines[index])
        and not _is_pipe_separator_line(lines[index])
        and _is_pipe_separator_line(lines[index + 1])
    )


def _pipe_block_parts(lines: list[str]) -> list[tuple[str, object]]:
    # A new header and separator can identify an adjacent table. If any part is
    # malformed, keep the whole block: it may instead represent a nested table.
    starts = [0] + [
        index
        for index in range(2, len(lines))
        if _looks_like_pipe_table_start(lines, index)
    ]
    ends = starts[1:] + [len(lines)]
    tables = [
        _structured_table_from_pipe_lines(lines[start:end])
        for start, end in zip(starts, ends)
    ]
    if len(tables) > 1 and any(
        isinstance(table, _UnstructuredPipeTable) for table in tables
    ):
        return [
            ("table_fallback", _pipe_table_fallback(lines, "ambiguous_table_boundary"))
        ]
    return [
        (
            "table_fallback"
            if isinstance(table, _UnstructuredPipeTable)
            else "table_pipe",
            table,
        )
        for table in tables
    ]


def _is_pipe_table_row(line: str) -> bool:
    stripped = line.strip()
    return len(stripped) >= 2 and stripped.startswith("|") and stripped.endswith("|")


def _is_pipe_separator_line(line: str) -> bool:
    if not _is_pipe_table_row(line):
        return False
    cells = _split_pipe_row(line)
    return bool(cells) and all(
        re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells
    )


def _pipe_table_fallback(lines: list[str], reason: str) -> _UnstructuredPipeTable:
    return _UnstructuredPipeTable(
        text="\n".join(lines),
        reason=reason,
        line_column_counts=[len(_split_pipe_row(line)) for line in lines],
    )


def _structured_table_from_pipe_lines(
    lines: list[str],
) -> StructuredTableContent | _UnstructuredPipeTable:
    if len(lines) < 3:
        return _pipe_table_fallback(lines, "missing_body")
    if not _looks_like_pipe_table_start(lines, 0):
        return _pipe_table_fallback(lines, "ambiguous_header")
    headers, separator, *rows = [_split_pipe_row(line) for line in lines]
    if any(len(row) != len(headers) for row in [separator, *rows]):
        return _pipe_table_fallback(lines, "column_count_mismatch")
    if not all(headers):
        return _pipe_table_fallback(lines, "ambiguous_header")
    if any(_is_pipe_separator_line(line) for line in lines[2:]):
        return _pipe_table_fallback(lines, "unexpected_separator")
    columns = [
        table_column(f"c{index}", header)
        for index, header in enumerate(headers, start=1)
    ]
    structured_rows: list[dict[str, object]] = []
    for row in rows:
        cells = []
        for index, column in enumerate(columns):
            value = row[index]
            cells.append(_simple_cell(str(column["id"]), value))
        structured_rows.append(table_row(len(structured_rows) + 1, cells))
    return _structured_table_content(columns=columns, rows=structured_rows)


def _looks_like_aligned_table_start(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    header = _split_aligned_table_row(lines[index])
    first_row = _split_aligned_table_row(lines[index + 1])
    return (
        len(header) >= 2
        and len(header) == len(first_row)
        and all(cell.strip() for cell in header)
        and all(cell.strip() for cell in first_row)
    )


def _structured_table_from_aligned_lines(lines: list[str]) -> dict[str, object] | None:
    if len(lines) < 2:
        return None
    headers = _split_aligned_table_row(lines[0])
    rows = [_split_aligned_table_row(line) for line in lines[1:]]
    if not headers or not rows or any(len(row) != len(headers) for row in rows):
        return None
    columns = [
        table_column(f"c{index}", header)
        for index, header in enumerate(headers, start=1)
    ]
    structured_rows: list[dict[str, object]] = []
    for row in rows:
        cells = [
            _simple_cell(str(column["id"]), row[index])
            for index, column in enumerate(columns)
        ]
        structured_rows.append(table_row(len(structured_rows) + 1, cells))
    return _structured_table_content(columns=columns, rows=structured_rows)


def _split_aligned_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if not re.search(r"\t| {2,}", stripped):
        return []
    return [cell.strip() for cell in re.split(r"\t+| {2,}", stripped) if cell.strip()]


def _split_pipe_row(line: str) -> list[str]:
    # Remove exactly the outer delimiters; adjacent pipes are real empty cells.
    body = line.strip()[1:-1]
    cells: list[str] = []
    cell: list[str] = []
    index = 0
    while index < len(body):
        char = body[index]
        if char == "\\" and index + 1 < len(body):
            following = body[index + 1]
            cell.append("|" if following == "|" else char + following)
            index += 2
            continue
        if char == "|":
            cells.append("".join(cell).strip())
            cell = []
        else:
            cell.append(char)
        index += 1
    cells.append("".join(cell).strip())
    return cells
