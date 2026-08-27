from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

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
        if kind in {"table", "table_pipe", "table_text"}:
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
        if _looks_like_pipe_table_start(lines, index):
            flush_paragraph()
            table_lines = [lines[index], lines[index + 1]]
            index += 2
            while index < len(lines) and _is_pipe_table_row(lines[index]):
                table_lines.append(lines[index])
                index += 1
            table = _structured_table_from_pipe_lines(table_lines)
            if table is not None:
                parts.append(("table_pipe", table))
                continue
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
    return parts or [("text", text)]


def _looks_like_pipe_table_start(lines: list[str], index: int) -> bool:
    return (
        index + 2 < len(lines)
        and _is_pipe_table_row(lines[index])
        and _is_pipe_separator_line(lines[index + 1])
        and _is_pipe_table_row(lines[index + 2])
    )


def _is_pipe_table_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 3


def _is_pipe_separator_line(line: str) -> bool:
    if not _is_pipe_table_row(line):
        return False
    cells = _split_pipe_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)


def _structured_table_from_pipe_lines(lines: list[str]) -> dict[str, object] | None:
    if len(lines) < 3:
        return None
    headers = _split_pipe_row(lines[0])
    rows = [_split_pipe_row(line) for line in lines[2:] if _is_pipe_table_row(line)]
    if not headers or not rows:
        return None
    columns = [
        table_column(f"c{index}", header)
        for index, header in enumerate(headers, start=1)
    ]
    structured_rows: list[dict[str, object]] = []
    for row in rows:
        cells = []
        for index, column in enumerate(columns):
            value = row[index] if index < len(row) else ""
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
    return [
        cell.strip()
        for cell in re.split(r"\t+| {2,}", stripped)
        if cell.strip()
    ]


def _split_pipe_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]
