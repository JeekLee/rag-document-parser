from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HtmlSourceProjector:
    """Projects HTML evidence and section context into source text."""

    def table(
        self,
        headers: list[str],
        rows: list[dict[str, str]],
        section_path: list[str],
        caption: str | None,
    ) -> str:
        return _table_source_text(headers, rows, section_path, caption)

    def with_section(self, section_path: list[str], text: str) -> str:
        return _with_section(section_path, text)


def _table_source_text(
    headers: list[str],
    rows: list[dict[str, str]],
    section_path: list[str],
    caption: str | None,
) -> str:
    lines: list[str] = []
    if section_path:
        lines.append(f"section: {' > '.join(section_path)}")
    if caption:
        lines.append(f"caption: {caption}")
    lines.append(f"columns: {' | '.join(headers)}")
    for index, row in enumerate(rows, start=1):
        values = [
            f"{header}={value}"
            for header in headers
            if (value := row.get(header))
        ]
        if values:
            lines.append(f"row {index}: {'; '.join(values)}")
    return "\n".join(lines)


def _with_section(section_path: list[str], text: str) -> str:
    if not section_path:
        return text
    return f"section: {' > '.join(section_path)}\n{text}"
