from __future__ import annotations

from ....models import EvidenceUnit, SourceEvidence
from ...backend import ParsedDocument
from ...schema import common_metadata
from .tables import structured_table, table_parts, table_source_text


class MarkdownBackend:
    def parse(self, data: bytes, suffix: str) -> ParsedDocument:
        markdown = data.decode("utf-8", errors="replace")
        return ParsedDocument(units=_units_from_markdown(markdown))


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
                    text=_with_section(section_path, text),
                ),
                format="plain",
                content=text,
                metadata=common_metadata("text", "plain", section_path=section_path),
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
        headers, rows = table_parts(lines)
        source_text = table_source_text(headers, rows, section_path)
        units.append(
            EvidenceUnit(
                id=block_id,
                type="table",
                source=SourceEvidence(kind="table", text=source_text),
                format="structured_table",
                content=structured_table(headers, rows),
                metadata={
                    **common_metadata(
                        "table",
                        "structured_table",
                        section_path=section_path,
                    ),
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


def _with_section(section_path: list[str], text: str) -> str:
    if not section_path:
        return text
    return f"section: {' > '.join(section_path)}\n{text}"
