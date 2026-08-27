from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ...table_source import (
    build_column_source_labels as _build_column_source_labels,
    common_semantic_header_prefix as _common_semantic_header_prefix,
    is_semantic_column_label as _is_semantic_column_label,
    semantic_column_group_label as _semantic_column_group_label,
)


@dataclass(frozen=True)
class PdfTableSourceProjector:
    """Projects canonical PDF table evidence into deterministic source text."""

    def project(self, table: dict[str, object]) -> str:
        return _table_source_text(table)

    def cell_label(
        self,
        cell: dict[str, object],
        column_text: dict[str, str],
        *,
        use_header_labels: bool,
    ) -> str:
        return _cell_source_label(
            cell,
            column_text,
            use_header_labels=use_header_labels,
        )


def _table_source_text(table: dict[str, object]) -> str:
    columns = table["columns"]
    rows = table["rows"]
    column_text = _build_column_source_labels(columns, _column_source_label, rows)
    lines: list[str] = []
    if columns:
        lines.append(f"table: {len(columns)} columns")
    for header_row in table.get("header_rows", []):
        cells = _table_source_cells(
            header_row["cells"],
            column_text,
            use_header_labels=False,
        )
        if cells:
            lines.append(f"header {header_row['index']}: " + "; ".join(cells))
    for row in rows:
        cells = _table_source_cells(
            row["cells"],
            column_text,
            use_header_labels=True,
        )
        if cells:
            lines.append(f"row {row['index']}: " + "; ".join(cells))
    return "\n".join(lines)


def _column_source_label(column: dict[str, object]) -> str:
    text = str(column["text"]).strip()
    return text or _column_coordinate_label(str(column["id"]))


def _table_source_cells(
    cells: list[dict[str, object]],
    column_text: dict[str, str],
    *,
    use_header_labels: bool,
) -> list[str]:
    result: list[str] = []
    for cell in cells:
        header = _cell_source_label(
            cell,
            column_text,
            use_header_labels=use_header_labels,
        )
        value = str(cell["text"])
        child_texts = [
            "nested table: " + _inline_table_source(child["content"])
            for child in cell["children"]
            if child.get("type", child.get("kind")) == "table"
        ]
        image_texts = [
            f"image: {child['content']['asset_id']}"
            for child in cell["children"]
            if child.get("type", child.get("kind")) == "image"
        ]
        diagram_texts = [
            "diagram: " + _inline_diagram_source(child["content"])
            for child in cell["children"]
            if child.get("type", child.get("kind")) == "diagram"
        ]
        combined = "; ".join(
            part for part in [value, *child_texts, *image_texts, *diagram_texts] if part
        )
        if combined:
            result.append(f"{header}: {combined}")
    return result


def _inline_diagram_source(diagram: object) -> str:
    if not isinstance(diagram, Mapping):
        return ""
    nodes = diagram.get("nodes")
    labels = [
        text
        for text in (
            str(node.get("text", "")).strip()
            for node in nodes
            if isinstance(node, Mapping)
        )
        if text
    ] if isinstance(nodes, list) else []
    if labels:
        return " / ".join(labels)
    asset_id = str(diagram.get("asset_id", "")).strip()
    return f"image: {asset_id}" if asset_id else ""


def _cell_source_label(
    cell: dict[str, object],
    column_text: dict[str, str],
    *,
    use_header_labels: bool,
) -> str:
    column_id = str(cell["column_id"])
    colspan = int(cell.get("colspan", 1))
    if use_header_labels:
        labels = [
            column_text.get(f"c{column_index}", f"col {column_index}")
            for column_index in range(
                _column_id_number(column_id),
                _column_id_number(column_id) + colspan,
            )
        ]
        common_prefix = _common_semantic_header_prefix(labels)
        if common_prefix is not None:
            return common_prefix
        group_label = _semantic_column_group_label(labels)
        if group_label is not None:
            return group_label
        if len(set(labels)) == 1 and _is_semantic_column_label(labels[0]):
            return labels[0]
        if colspan == 1 and _is_semantic_column_label(labels[0]):
            return labels[0]
    return _cell_coordinate_label(column_id, colspan)


def _cell_coordinate_label(column_id: str, colspan: int) -> str:
    start = _column_id_number(column_id)
    if colspan <= 1:
        return _column_coordinate_label(column_id)
    return f"cols {start}-{start + colspan - 1}"


def _column_coordinate_label(column_id: str) -> str:
    return f"col {_column_id_number(column_id)}"


def _column_id_number(column_id: str) -> int:
    try:
        return max(1, int(column_id.removeprefix("c")))
    except ValueError:
        return 1


def _inline_table_source(table: dict[str, object]) -> str:
    source = _table_source_text(table)
    return source.replace("\n", " / ")
