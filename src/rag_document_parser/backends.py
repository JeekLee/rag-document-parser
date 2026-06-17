from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .models import Evidence, EvidenceUnit, SourceEvidence


@dataclass(frozen=True)
class ParsedDocument:
    units: list[EvidenceUnit]
    quality_warnings: list[dict[str, Any]] = field(default_factory=list)


class DocumentBackend(Protocol):
    def parse(self, data: bytes, suffix: str) -> ParsedDocument:
        """Parse raw document bytes into source-preserving evidence units."""
        ...


class MarkdownBackend:
    def parse(self, data: bytes, suffix: str) -> ParsedDocument:
        markdown = data.decode("utf-8", errors="replace")
        return ParsedDocument(units=_units_from_markdown(markdown))


def default_backends() -> dict[str, DocumentBackend]:
    markdown_backend = MarkdownBackend()
    return {
        ".markdown": markdown_backend,
        ".md": markdown_backend,
        ".txt": markdown_backend,
    }


def _units_from_markdown(markdown: str) -> list[EvidenceUnit]:
    units: list[EvidenceUnit] = []
    section_path: list[str] = []
    paragraph_lines: list[str] = []
    table_lines: list[str] = []
    block_index = 1
    table_index = 1

    def flush_paragraph() -> None:
        nonlocal block_index
        meaningful = [line for line in paragraph_lines if line.strip()]
        text = " ".join(line.strip() for line in meaningful).strip()
        paragraph_lines.clear()
        if not text:
            return
        chunk_id = f"b{block_index}"
        block_index += 1
        units.append(
            EvidenceUnit(
                id=chunk_id,
                type="text",
                source=SourceEvidence(
                    kind="text",
                    text=text,
                    section_path=list(section_path),
                ),
                evidence=Evidence(kind="text", format="plain", content=text),
                metadata={
                    "common": {
                        "chunk_kind": "text",
                        "section_path": list(section_path),
                        "display_format": "plain",
                    }
                },
            )
        )

    def flush_table() -> None:
        nonlocal block_index, table_index
        lines = [line for line in table_lines if line.strip()]
        table_lines.clear()
        if not lines:
            return
        table_id = f"t{table_index}"
        block_id = f"b{block_index}"
        block_index += 1
        table_index += 1
        headers, rows = _table_parts(lines)
        table_source_text = _table_source_text(headers, rows)
        units.append(
            EvidenceUnit(
                id=block_id,
                type="table",
                source=SourceEvidence(
                    kind="table",
                    text=table_source_text,
                    section_path=list(section_path),
                    headers=headers,
                    rows=_source_rows(headers, rows),
                ),
                evidence=Evidence(
                    kind="table",
                    format="structured_table",
                    content=_structured_table(headers, rows),
                ),
                metadata={
                    "common": {
                        "chunk_kind": "table",
                        "section_path": list(section_path),
                        "display_format": "structured_table",
                    },
                    "table": {
                        "table_id": table_id,
                        "headers": headers,
                        "row_count": len(rows),
                    },
                },
            )
        )

    for line in markdown.splitlines():
        line = line.rstrip()
        if line.lstrip().startswith("#"):
            flush_paragraph()
            flush_table()
            heading = line.lstrip("#").strip()
            if heading:
                section_path = [heading]
            continue
        if line.lstrip().startswith("|"):
            flush_paragraph()
            table_lines.append(line)
            continue
        if table_lines:
            flush_table()
        if line.strip():
            paragraph_lines.append(line)
        else:
            flush_paragraph()

    flush_paragraph()
    flush_table()
    return units


def _table_parts(lines: list[str]) -> tuple[list[str], list[list[str]]]:
    rows = [_split_table_row(line) for line in lines]
    if len(rows) < 3:
        return [], []
    return rows[0], rows[2:]


def _structured_table(headers: list[str], rows: list[list[str]]) -> dict[str, object]:
    columns = [
        {
            "id": f"c{index}",
            "text": header,
        }
        for index, header in enumerate(headers, start=1)
    ]
    structured_rows: list[dict[str, object]] = []
    for row_index, row in enumerate(rows, start=1):
        cells: list[dict[str, object]] = []
        for column, value in zip(columns, row, strict=False):
            cells.append(
                {
                    "column_id": column["id"],
                    "text": value,
                    "rowspan": 1,
                    "colspan": 1,
                    "children": [],
                }
            )
        structured_rows.append(
            {
                "index": row_index,
                "cells": cells,
            }
        )
    return {
        "caption": None,
        "columns": columns,
        "rows": structured_rows,
    }


def _source_rows(headers: list[str], rows: list[list[str]]) -> list[dict[str, object]]:
    source_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        source_rows.append(
            {
                "index": index,
                "cells": {
                    header: value
                    for header, value in zip(headers, row, strict=False)
                },
            }
        )
    return source_rows


def _table_source_text(headers: list[str], rows: list[list[str]]) -> str:
    parts: list[str] = []
    for row in rows:
        cells = [
            f"{header}={value}"
            for header, value in zip(headers, row, strict=False)
            if value
        ]
        if cells:
            parts.append("; ".join(cells))
    return "\n".join(parts)


def _split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]
