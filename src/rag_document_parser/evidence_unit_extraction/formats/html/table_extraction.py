from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from bs4.element import Tag

from ...schema import structured_table, table_cell, table_column, table_row
from .assets import HtmlImageExtractor
from .state import HtmlParseState
from .text import HtmlTextExtractor
from .text import tag_name as _tag_name

_IMAGE_EXTRACTOR = HtmlImageExtractor()
_TEXT_EXTRACTOR = HtmlTextExtractor()
_HtmlParseState = HtmlParseState
_text_with_links = _TEXT_EXTRACTOR.text
_figure_caption = _IMAGE_EXTRACTOR.figure_caption
_image_caption_from_context = _IMAGE_EXTRACTOR.caption_from_context
_image_asset_ref = _IMAGE_EXTRACTOR.reference


@dataclass(frozen=True)
class HtmlTableExtractor:
    """Extracts canonical tables and nested evidence from HTML."""

    def extract(
        self,
        table: Tag,
        state: HtmlParseState,
    ) -> tuple[object, list[str], list[dict[str, str]], str | None] | None:
        return _parse_table_content(table, state)


def _parse_table_content(
    table: Tag,
    state: _HtmlParseState,
) -> tuple[object, list[str], list[dict[str, str]], str | None] | None:
    rows = _direct_table_rows(table)
    if not rows:
        return None

    header_index = _header_row_index(rows)
    if header_index is None:
        header_index = 0
    header_cells = _direct_cells(rows[header_index])
    headers = [_text_with_links(cell) for cell in header_cells]
    headers = [
        header or f"Column {index}"
        for index, header in enumerate(headers, start=1)
    ]
    if not headers:
        return None

    columns = [
        table_column(f"c{index}", header)
        for index, header in enumerate(headers, start=1)
    ]
    data_rows = [row for index, row in enumerate(rows) if index != header_index]
    rowspan_slots: dict[int, int] = {}
    structured_rows = []
    row_source_values: list[dict[str, str]] = []

    for row_index, row in enumerate(data_rows, start=1):
        parsed_cells = []
        source_values: dict[str, str] = {}
        column_index = 0
        for cell in _direct_cells(row):
            column_index = _next_open_column(column_index, rowspan_slots)
            if column_index >= len(columns):
                break
            text = _text_with_links(cell, skip_tags={"figure", "img", "table"})
            children = _cell_children(cell, state)
            rowspan = _positive_int(cell.get("rowspan"), default=1)
            colspan = _positive_int(cell.get("colspan"), default=1)
            column_id = columns[column_index]["id"]
            parsed_cells.append(
                table_cell(
                    column_id,
                    text,
                    rowspan=rowspan,
                    colspan=colspan,
                    children=children,
                )
            )
            source_value = _cell_source_value(text, children)
            if source_value:
                source_values[headers[column_index]] = source_value
            if rowspan > 1:
                for offset in range(colspan):
                    rowspan_slots[column_index + offset] = rowspan
            column_index += colspan
        _advance_rowspans(rowspan_slots)
        if parsed_cells:
            structured_rows.append(table_row(row_index, parsed_cells))
            row_source_values.append(source_values)

    caption = _table_caption(table)
    return (
        structured_table(columns=columns, rows=structured_rows, caption=caption),
        headers,
        row_source_values,
        caption,
    )


def _cell_children(cell: Tag, state: _HtmlParseState) -> list[dict[str, object]]:
    children: list[dict[str, object]] = []
    for nested in _cell_nested_tables(cell):
        parsed = _parse_table_content(nested, state)
        if parsed is None:
            continue
        nested_content, _, _, _ = parsed
        children.append(
            {
                "type": "table",
                "format": "structured_table",
                "content": nested_content,
            }
        )
    for image in _cell_images(cell):
        caption = _image_caption_from_context(image)
        image_ref = _image_asset_ref(image, state)
        if image_ref is None:
            continue
        asset_id, alt = image_ref
        children.append(
            {
                "type": "image",
                "format": "asset_ref",
                "content": {
                    "asset_id": asset_id,
                    "caption": caption or alt,
                },
            }
        )
    return children


def _cell_nested_tables(cell: Tag) -> list[Tag]:
    parent_table = cell.find_parent("table")
    return [
        table
        for table in cell.find_all("table")
        if table.find_parent("table") is parent_table
    ]


def _cell_images(cell: Tag) -> list[Tag]:
    parent_table = cell.find_parent("table")
    return [
        image
        for image in cell.find_all("img")
        if image.find_parent("table") is parent_table
    ]


def _cell_source_value(text: str, children: list[dict[str, object]]) -> str:
    parts = [text] if text else []
    for child in children:
        child_type = child.get("type")
        content = child.get("content")
        if child_type == "image" and isinstance(content, Mapping):
            asset_id = content.get("asset_id")
            if asset_id:
                parts.append(f"image: {asset_id}")
        elif child_type == "table":
            parts.append("nested table: " + _nested_table_summary(content))
    return "; ".join(parts)


def _nested_table_summary(content: object) -> str:
    if not isinstance(content, Mapping):
        return "structured_table"
    columns = content.get("columns")
    rows = content.get("rows")
    parts: list[str] = []
    if isinstance(columns, list):
        labels = [
            str(column.get("text"))
            for column in columns
            if isinstance(column, Mapping) and column.get("text")
        ]
        if labels:
            parts.append(f"columns: {' | '.join(labels)}")
    if isinstance(rows, list):
        parts.append(f"rows: {len(rows)}")
    return " / ".join(parts) if parts else "structured_table"


def _direct_table_rows(table: Tag) -> list[Tag]:
    rows: list[Tag] = []
    for child in table.children:
        if not isinstance(child, Tag):
            continue
        name = _tag_name(child)
        if name in {"thead", "tbody", "tfoot"}:
            rows.extend(
                row
                for row in child.children
                if isinstance(row, Tag) and _tag_name(row) == "tr"
            )
        elif name == "tr":
            rows.append(child)
    return rows


def _direct_cells(row: Tag) -> list[Tag]:
    return [
        child
        for child in row.children
        if isinstance(child, Tag) and _tag_name(child) in {"td", "th"}
    ]


def _header_row_index(rows: list[Tag]) -> int | None:
    for index, row in enumerate(rows):
        if any(_tag_name(cell) == "th" for cell in _direct_cells(row)):
            return index
    return None


def _next_open_column(column_index: int, rowspan_slots: dict[int, int]) -> int:
    while rowspan_slots.get(column_index, 0) > 0:
        column_index += 1
    return column_index


def _advance_rowspans(rowspan_slots: dict[int, int]) -> None:
    for column_index in list(rowspan_slots):
        rowspan_slots[column_index] -= 1
        if rowspan_slots[column_index] <= 0:
            del rowspan_slots[column_index]


def _positive_int(value: object, *, default: int) -> int:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _table_caption(table: Tag) -> str | None:
    for child in table.children:
        if isinstance(child, Tag) and _tag_name(child) == "caption":
            text = _text_with_links(child)
            return text or None
    return None
