from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .models import Evidence, ParseResult, RagChunk, SourceInfo, SourcePointer


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
            chunks=_chunks_from_markdown(markdown, sha256),
        )


@dataclass(frozen=True)
class _Line:
    text: str
    start: int
    end: int


def _chunks_from_markdown(markdown: str, sha256: str) -> list[RagChunk]:
    chunks: list[RagChunk] = []
    section_path: list[str] = []
    paragraph_lines: list[_Line] = []
    table_lines: list[_Line] = []
    block_index = 1
    table_index = 1

    def flush_paragraph() -> None:
        nonlocal block_index
        meaningful = [line for line in paragraph_lines if line.text.strip()]
        text = " ".join(line.text.strip() for line in meaningful).strip()
        paragraph_lines.clear()
        if not text:
            return
        chunk_id = f"b{block_index}"
        block_index += 1
        char_start = _content_start(meaningful[0])
        char_end = _content_end(meaningful[-1])
        chunks.append(
            RagChunk(
                id=chunk_id,
                type="text",
                source=text,
                embedding_text=_with_section(section_path, text),
                evidence=Evidence(format="plain", content=text),
                source_pointer=SourcePointer(
                    sha256=sha256,
                    char_start=char_start,
                    char_end=char_end,
                    byte_start=_byte_offset(markdown, char_start),
                    byte_end=_byte_offset(markdown, char_end),
                    section_path=list(section_path),
                    block_id=chunk_id,
                ),
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
        meaningful = [line for line in table_lines if line.text.strip()]
        lines = [line.text for line in meaningful]
        table_lines.clear()
        if not lines:
            return
        table_id = f"t{table_index}"
        block_id = f"b{block_index}"
        block_index += 1
        table_index += 1
        headers, rows = _table_parts(lines)
        char_start = meaningful[0].start
        char_end = meaningful[-1].end
        chunks.append(
            RagChunk(
                id=block_id,
                type="table",
                source="\n".join(lines),
                embedding_text=_table_llm_text(lines, section_path),
                evidence=Evidence(format="markdown_table", content="\n".join(lines)),
                source_pointer=SourcePointer(
                    sha256=sha256,
                    char_start=char_start,
                    char_end=char_end,
                    byte_start=_byte_offset(markdown, char_start),
                    byte_end=_byte_offset(markdown, char_end),
                    section_path=list(section_path),
                    block_id=block_id,
                    table_id=table_id,
                    row_range=(1, max(1, len(rows))),
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

    for item in _iter_lines(markdown):
        line = item.text
        if line.lstrip().startswith("#"):
            flush_paragraph()
            flush_table()
            heading = line.lstrip("#").strip()
            if heading:
                section_path = [heading]
            continue
        if line.lstrip().startswith("|"):
            flush_paragraph()
            table_lines.append(item)
            continue
        if table_lines:
            flush_table()
        if line.strip():
            paragraph_lines.append(item)
        else:
            flush_paragraph()

    flush_paragraph()
    flush_table()
    return chunks


def _iter_lines(markdown: str) -> list[_Line]:
    lines: list[_Line] = []
    offset = 0
    for raw in markdown.splitlines(keepends=True):
        text = raw.rstrip("\r\n")
        start = offset
        end = start + len(text)
        lines.append(_Line(text=text, start=start, end=end))
        offset += len(raw)
    if not markdown:
        return []
    if markdown and not markdown.endswith(("\n", "\r")) and not lines:
        lines.append(_Line(text=markdown, start=0, end=len(markdown)))
    return lines


def _content_start(line: _Line) -> int:
    return line.start + len(line.text) - len(line.text.lstrip())


def _content_end(line: _Line) -> int:
    return line.start + len(line.text.rstrip())


def _byte_offset(text: str, char_offset: int) -> int:
    return len(text[:char_offset].encode())


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


def _split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]
