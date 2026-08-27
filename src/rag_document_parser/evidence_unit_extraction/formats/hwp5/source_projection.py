from __future__ import annotations

from dataclasses import dataclass

from ...table_source import (
    build_column_source_labels as _build_column_source_labels,
    common_semantic_header_prefix as _common_semantic_header_prefix,
    is_semantic_column_label as _is_semantic_column_label,
)
from .diagram import Hwp5DiagramBuilder

_DIAGRAM_BUILDER = Hwp5DiagramBuilder()


@dataclass(frozen=True)
class Hwp5TableSourceProjector:
    """Projects canonical HWP5 tables into deterministic source text."""

    def project(self, table: dict[str, object]) -> str:
        return _table_source_text(table)


def _table_source_text(table: dict[str, object]) -> str:
    columns = table["columns"]
    rows = table["rows"]
    column_text = _build_column_source_labels(columns, _column_source_label, rows)
    lines: list[str] = []
    active_rowspans: list[tuple[int, int, int, dict[str, object]]] = []
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
        row_index = int(row["index"])
        row_cells = list(row["cells"])
        source_row_cells = _source_cells_with_rowspan_context(
            row_cells,
            active_rowspans,
            row_index,
        )
        cells = _table_source_cells(
            source_row_cells,
            column_text,
            use_header_labels=True,
        )
        if cells:
            lines.append(f"row {row['index']}: " + "; ".join(cells))
        active_rowspans = [
            span for span in active_rowspans if span[2] > row_index
        ]
        active_rowspans.extend(_rowspan_source_spans(row_cells, row_index))
    return "\n".join(lines)


def _source_cells_with_rowspan_context(
    cells: list[dict[str, object]],
    active_rowspans: list[tuple[int, int, int, dict[str, object]]],
    row_index: int,
) -> list[dict[str, object]]:
    occupied = [_cell_column_range(cell) for cell in cells]
    context_cells = [
        cell
        for start, end, last_row, cell in active_rowspans
        if last_row >= row_index
        and not any(
            _ranges_overlap(start, end, other_start, other_end)
            for other_start, other_end in occupied
        )
    ]
    return sorted(
        [*context_cells, *cells],
        key=lambda cell: _cell_column_range(cell)[0],
    )


def _rowspan_source_spans(
    cells: list[dict[str, object]],
    row_index: int,
) -> list[tuple[int, int, int, dict[str, object]]]:
    spans: list[tuple[int, int, int, dict[str, object]]] = []
    for cell in cells:
        rowspan = _positive_int(cell.get("rowspan"), default=1)
        if rowspan <= 1:
            continue
        start, end = _cell_column_range(cell)
        spans.append((start, end, row_index + rowspan - 1, cell))
    return spans


def _cell_column_range(cell: dict[str, object]) -> tuple[int, int]:
    column = _column_id_number(str(cell.get("column_id", "c1")))
    colspan = _positive_int(cell.get("colspan"), default=1)
    return column, column + colspan


def _ranges_overlap(
    start: int,
    end: int,
    other_start: int,
    other_end: int,
) -> bool:
    return start < other_end and other_start < end


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
        value = _inline_cell_text(str(cell["text"]))
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
            "nested diagram: " + diagram_source
            for child in cell["children"]
            if child.get("type", child.get("kind")) == "diagram"
            and (diagram_source := _inline_diagram_source(child["content"]))
        ]
        combined = "; ".join(
            part for part in [value, *child_texts, *image_texts, *diagram_texts] if part
        )
        if combined:
            result.append(f"{header}: {combined}")
    return result


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
        if len(set(labels)) == 1 and _is_semantic_column_label(labels[0]):
            return labels[0]
        if colspan == 1 and _is_semantic_column_label(labels[0]):
            return labels[0]
    return _cell_coordinate_label(column_id, colspan)


def _column_source_label(column: dict[str, object]) -> str:
    text = str(column["text"]).strip()
    return text or _column_coordinate_label(str(column["id"]))


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


def _positive_int(value: object, *, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _inline_table_source(table: dict[str, object]) -> str:
    source = _table_source_text(table)
    return source.replace("\n", " / ")


def _inline_diagram_source(diagram: dict[str, object]) -> str:
    source = _DIAGRAM_BUILDER.source_text(diagram)
    return source.replace("\n", " / ")


def _inline_cell_text(text: str) -> str:
    return " / ".join(part.strip() for part in text.splitlines() if part.strip())
