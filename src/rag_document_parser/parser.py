from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .models import Evidence, ParseResult, RagChunk, SourceEvidence, SourceInfo


@dataclass
class RagDocumentParser:
    def parse(
        self,
        source: bytes | str,
        *,
        suffix: str,
        source_id: str | None = None,
        source_name: str | None = None,
        source_url: str | None = None,
    ) -> ParseResult:
        data = source.encode() if isinstance(source, str) else bytes(source)
        normalized_suffix = suffix.lower()
        markdown = data.decode("utf-8", errors="replace")
        sha256 = hashlib.sha256(data).hexdigest()
        source_info = SourceInfo(
            sha256=sha256,
            suffix=normalized_suffix,
            bytes=len(data),
            id=source_id,
            name=source_name,
            url=source_url,
        )
        return ParseResult(
            source=source_info,
            chunks=_chunks_from_markdown(markdown),
        )


def _chunks_from_markdown(markdown: str) -> list[RagChunk]:
    chunks: list[RagChunk] = []
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
        chunks.append(
            RagChunk(
                id=chunk_id,
                type="text",
                source=SourceEvidence(
                    kind="text",
                    text=text,
                    section_path=list(section_path),
                ),
                embedding_text=_with_section(section_path, text),
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
        chunks.append(
            RagChunk(
                id=block_id,
                type="table",
                source=SourceEvidence(
                    kind="table",
                    text=table_source_text,
                    section_path=list(section_path),
                    headers=headers,
                    rows=_source_rows(headers, rows),
                ),
                embedding_text=_table_llm_text(lines, section_path),
                evidence=Evidence(
                    kind="table",
                    format="markdown_table",
                    content="\n".join(lines),
                ),
                metadata={
                    "common": {
                        "chunk_kind": "table",
                        "section_path": list(section_path),
                        "display_format": "markdown_table",
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
    return chunks


def _with_section(section_path: list[str], text: str) -> str:
    if not section_path:
        return text
    return f"section: {' > '.join(section_path)}\n{text}"


def _table_llm_text(lines: list[str], section_path: list[str]) -> str:
    columns, body_rows = _table_parts(lines)
    if not columns:
        return _with_section(section_path, "table:\n" + "\n".join(lines))

    rendered_rows: list[str] = []
    for index, row in enumerate(body_rows, start=1):
        cells = []
        for column, value in zip(columns, row, strict=False):
            cells.append(f"{column}={value}")
        rendered_rows.append(f"row {index}: {'; '.join(cells)}")

    text = "table:\n"
    text += f"columns: {' | '.join(columns)}"
    if rendered_rows:
        text += "\n" + "\n".join(rendered_rows)
    return _with_section(section_path, text)


def _table_parts(lines: list[str]) -> tuple[list[str], list[list[str]]]:
    rows = [_split_table_row(line) for line in lines]
    if len(rows) < 3:
        return [], []
    return rows[0], rows[2:]


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
