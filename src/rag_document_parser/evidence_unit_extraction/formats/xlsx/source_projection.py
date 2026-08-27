from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class XlsxTableSourceProjector:
    """Projects canonical worksheet tables into deterministic source text."""

    def project(
        self,
        sheet_name: str,
        cell_range: str,
        columns: list[dict[str, Any]],
        header_rows: list[dict[str, Any]],
        rows: list[dict[str, Any]],
    ) -> str:
        return _table_source_text(
            sheet_name,
            cell_range,
            columns,
            header_rows,
            rows,
        )

    def cell_value(self, payload: dict[str, Any]) -> str:
        return _cell_payload_source_value(payload)


def _cell_payload_source_value(payload: dict[str, Any]) -> str:
    text = str(payload["text"])
    hyperlink = str(payload.get("hyperlink", "")).strip()
    if hyperlink and hyperlink != text:
        text = f"{text} ({hyperlink})" if text else hyperlink
    child_texts = [
        str(child.get("content", "")).strip()
        for child in payload["children"]
        if str(child.get("content", "")).strip()
    ]
    if child_texts:
        comment_text = " / ".join(child_texts)
        text = f"{text} [comment: {comment_text}]" if text else comment_text
    return text


def _table_source_text(
    sheet_name: str,
    cell_range: str,
    columns: list[dict[str, Any]],
    header_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> str:
    headers = [str(column["text"]) for column in columns]
    labels = _unique_source_labels(headers)
    lines = [
        f"section: {sheet_name}",
        f"range: {cell_range}",
        f"columns: {' | '.join(labels)}",
    ]
    for header_row in header_rows:
        header_values = [
            f"{cell['address']}={source_value}"
            for cell in header_row["cells"]
            if (source_value := _cell_payload_source_value(cell))
        ]
        if header_values:
            lines.append(
                f"header {header_row['index']} "
                f"[sheet row {header_row['source_row']}]: " + "; ".join(header_values)
            )
    for row in rows:
        values: list[str] = []
        for cell in row["cells"]:
            text = _cell_payload_source_value(cell).strip()
            if not text:
                continue
            column_number = _column_id_number(str(cell["column_id"]))
            label = (
                labels[column_number - 1]
                if 0 < column_number <= len(labels)
                else str(cell["column_id"])
            )
            values.append(f"{label}={text}")
        if values:
            lines.append(
                f"row {row['index']} [sheet row {row['source_row']}]: "
                + "; ".join(values)
            )
    return "\n".join(lines)


def _unique_source_labels(headers: list[str]) -> list[str]:
    totals = Counter(headers)
    seen: dict[str, int] = {}
    labels: list[str] = []
    for header in headers:
        seen[header] = seen.get(header, 0) + 1
        if totals[header] > 1:
            labels.append(f"{header} [{seen[header]}]")
        else:
            labels.append(header)
    return labels


def _column_id_number(column_id: str) -> int:
    match = re.search(r"\d+$", column_id)
    return int(match.group()) if match else 0
