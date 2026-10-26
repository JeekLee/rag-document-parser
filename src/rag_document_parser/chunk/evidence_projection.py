from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any


def _plain(value: Any) -> Any:
    """Copy both typed evidence models and stored dictionaries into JSON values."""
    if isinstance(value, Mapping):
        return {key: _plain(value[key]) for key in value}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _records(value: Any) -> list[Mapping[str, Any]]:
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def column_label(columns: list[Any], column_id: Any) -> str:
    for column in _records(columns):
        if column.get("id") == column_id and str(column.get("text", "")).strip():
            return str(column["text"]).strip()
    if isinstance(column_id, str):
        match = re.fullmatch(r"c([1-9][0-9]*)", column_id)
        if match:
            index = int(match.group(1)) - 1
            if index < len(columns) and isinstance(columns[index], Mapping):
                text = str(columns[index].get("text", "")).strip()
                if text:
                    return text
        return column_id
    return "col"


def table_payload(table: Mapping[str, Any]) -> dict[str, Any]:
    """Project exactly the supplied rows; compact caches may contain other rows."""
    columns = _records(table.get("columns"))
    payload: dict[str, Any] = {
        "caption": table.get("caption") or "",
        "columns": [str(column.get("text", "")).strip() for column in columns],
        "rows": [table_row_payload(row, columns) for row in _records(table.get("rows"))],
    }
    headers = _records(table.get("header_rows"))
    if headers:
        payload["header_rows"] = [table_row_payload(row, columns) for row in headers]
    return payload


def table_row_payload(row: Mapping[str, Any], columns: list[Any]) -> dict[str, Any]:
    payload = {key: _plain(row[key]) for key in row if key != "cells"}
    cells = []
    for cell in _records(row.get("cells")):
        projected = {key: _plain(cell[key]) for key in cell if key != "children"}
        projected["column"] = column_label(columns, cell.get("column_id"))
        children = _records(cell.get("children"))
        if children:
            projected["children"] = [child_payload(child) for child in children]
        cells.append(projected)
    payload["cells"] = cells
    return payload


def child_payload(child: Mapping[str, Any]) -> dict[str, Any]:
    payload = {key: _plain(child[key]) for key in child if key != "content"}
    content = child.get("content")
    if child.get("type", child.get("kind")) == "table" and isinstance(content, Mapping):
        payload["content"] = table_payload(content)
    else:
        payload["content"] = _plain(content)
    return payload


def table_source_text(table: Mapping[str, Any]) -> str:
    return _table_text(table_payload(table))


def _table_text(table: Mapping[str, Any]) -> str:
    columns = table["columns"]
    lines = [f"table: {len(columns)} columns"]
    if table.get("caption"):
        lines.append(f"caption: {table['caption']}")
    if columns:
        lines.append("columns: " + " | ".join(columns))
    for key, label in (("header_rows", "header"), ("rows", "row")):
        for row in table.get(key, []):
            values = []
            for cell in row["cells"]:
                parts = [str(cell.get("text", "")).strip()]
                parts.extend(_child_text(child) for child in cell.get("children", []))
                text = "; ".join(part for part in parts if part)
                fragment = cell.get("metadata", {}).get("cell_fragment")
                if isinstance(fragment, Mapping):
                    role = fragment.get("role")
                    if role == "omitted":
                        text = "[omitted in this row fragment]"
                    elif role == "context":
                        text += " [repeated context]"
                    elif (
                        role == "content"
                        and isinstance(fragment.get("text_range"), list)
                        and len(fragment["text_range"]) == 2
                    ):
                        start, end = fragment["text_range"]
                        text += f" [row fragment; text {start}:{end}]"
                if not text:
                    continue
                spans = [
                    f"{span}={cell[span]}"
                    for span in ("rowspan", "colspan")
                    if cell.get(span, 1) > 1
                ]
                value = f"{cell['column']}={text}"
                if spans:
                    value += " [" + ", ".join(spans) + "]"
                values.append(value)
            lines.append(f"{label} {row.get('index', '?')}: " + "; ".join(values))
    return "\n".join(lines)


def _child_text(child: Mapping[str, Any]) -> str:
    kind = child.get("type", child.get("kind"))
    content = child.get("content")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, Mapping):
        return ""
    if kind == "table":
        return "nested " + _table_text(content)
    if kind in {"image", "asset"}:
        return f"{kind}: " + "; ".join(
            str(content[key]) for key in ("asset_id", "caption") if content.get(key)
        )
    if kind == "diagram":
        # Keep node/edge relationships as well as captions and asset references.
        return "diagram: " + json.dumps(content, ensure_ascii=False)
    return ""
