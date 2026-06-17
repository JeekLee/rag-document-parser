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
            preview_markdown=markdown,
            chunks=_chunks_from_markdown(markdown, sha256),
        )


def _chunks_from_markdown(markdown: str, sha256: str) -> list[RagChunk]:
    chunks: list[RagChunk] = []
    section_path: list[str] = []
    paragraph_lines: list[str] = []
    table_lines: list[str] = []
    block_index = 1
    table_index = 1

    def flush_paragraph() -> None:
        nonlocal block_index
        text = " ".join(line.strip() for line in paragraph_lines if line.strip()).strip()
        paragraph_lines.clear()
        if not text:
            return
        chunk_id = f"b{block_index}"
        block_index += 1
        chunks.append(
            RagChunk(
                id=chunk_id,
                type="text",
                llm_text=_with_section(section_path, text),
                display=Evidence(format="markdown", content=text),
                source=SourcePointer(
                    sha256=sha256,
                    section_path=list(section_path),
                    block_id=chunk_id,
                ),
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
        chunks.append(
            RagChunk(
                id=block_id,
                type="table",
                llm_text=_table_llm_text(lines, section_path),
                display=Evidence(format="markdown", content="\n".join(lines)),
                source=SourcePointer(
                    sha256=sha256,
                    section_path=list(section_path),
                    block_id=block_id,
                    table_id=table_id,
                    row_range=(1, max(1, len(lines) - 2)),
                ),
                metadata={"table_id": table_id},
            )
        )

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
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
    rows = [_split_table_row(line) for line in lines]
    if len(rows) < 3:
        return _with_section(section_path, "table:\n" + "\n".join(lines))

    columns = rows[0]
    body_rows = rows[2:]
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


def _split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]
