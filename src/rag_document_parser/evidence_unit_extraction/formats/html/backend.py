from __future__ import annotations

import base64
import binascii
import re
from collections.abc import Mapping
from dataclasses import dataclass

from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag

from ....models import EvidenceUnit, PendingAsset, SourceEvidence
from ...backend import ParsedDocument
from ...schema import (
    common_metadata,
    structured_table,
    table_cell,
    table_column,
    table_row,
)

_BLOCK_TEXT_TAGS = {"blockquote", "p"}
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_DATA_URI_RE = re.compile(r"^data:([^;,]+);base64,(.*)$", re.IGNORECASE | re.DOTALL)
_SUPPORTED_IMAGE_MIME = {
    "image/gif": "gif",
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
}


@dataclass
class HtmlBackend:
    supported_suffixes = (".html", ".htm")

    def parse(self, data: bytes, suffix: str) -> ParsedDocument:
        html = data.decode("utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")
        root = soup.body or soup
        state = _HtmlParseState()
        units: list[EvidenceUnit] = []
        self._walk_blocks(root, state, units)
        return ParsedDocument(
            units=units,
            assets=state.assets,
            quality_warnings=state.quality_warnings,
        )

    def _walk_blocks(
        self,
        parent: Tag,
        state: _HtmlParseState,
        units: list[EvidenceUnit],
    ) -> None:
        for child in parent.children:
            if not isinstance(child, Tag):
                continue
            name = _tag_name(child)
            if name in _HEADING_TAGS:
                state.set_heading(int(name[1]), _text_with_links(child))
                continue
            if name in _BLOCK_TEXT_TAGS:
                self._append_text_unit(units, state, _text_with_links(child))
                continue
            if name == "pre":
                self._append_text_unit(
                    units,
                    state,
                    _text_with_links(child, preserve_pre=True),
                )
                continue
            if name == "table":
                table_unit = self._table_unit(child, state)
                if table_unit is not None:
                    units.append(table_unit)
                continue
            if name == "figure":
                image = child.find("img")
                if isinstance(image, Tag):
                    image_unit = self._image_unit(
                        image,
                        state,
                        caption=_figure_caption(child),
                    )
                    if image_unit is not None:
                        units.append(image_unit)
                continue
            if name == "img":
                image_unit = self._image_unit(child, state)
                if image_unit is not None:
                    units.append(image_unit)
                continue
            if name in {"ol", "ul"}:
                for item in child.find_all("li", recursive=False):
                    self._append_text_unit(units, state, _text_with_links(item))
                continue
            if name == "li":
                self._append_text_unit(units, state, _text_with_links(child))
                continue
            self._walk_blocks(child, state, units)

    def _append_text_unit(
        self,
        units: list[EvidenceUnit],
        state: _HtmlParseState,
        text: str,
    ) -> None:
        if not text:
            return
        units.append(
            EvidenceUnit(
                id=state.next_block_id(),
                type="text",
                source=SourceEvidence(
                    kind="text",
                    text=_with_section(state.section_path, text),
                ),
                format="plain",
                content=text,
                metadata=common_metadata(
                    "text",
                    "plain",
                    section_path=state.section_path,
                ),
            )
        )

    def _table_unit(self, table: Tag, state: _HtmlParseState) -> EvidenceUnit | None:
        parsed = _parse_table_content(table, state)
        if parsed is None:
            return None
        content, headers, row_source_values, caption = parsed
        table_id = state.next_table_id()
        return EvidenceUnit(
            id=state.next_block_id(),
            type="table",
            source=SourceEvidence(
                kind="table",
                text=_table_source_text(
                    headers,
                    row_source_values,
                    state.section_path,
                    caption,
                ),
            ),
            format="structured_table",
            content=content,
            metadata={
                **common_metadata(
                    "table",
                    "structured_table",
                    section_path=state.section_path,
                ),
                "table": {
                    "table_id": table_id,
                    "headers": headers,
                    "row_count": len(row_source_values),
                },
            },
        )

    def _image_unit(
        self,
        image: Tag,
        state: _HtmlParseState,
        *,
        caption: str | None = None,
    ) -> EvidenceUnit | None:
        image_ref = _image_asset_ref(image, state)
        if image_ref is None:
            return None
        asset_id, alt = image_ref
        content_caption = caption or alt
        return EvidenceUnit(
            id=state.next_block_id(),
            type="image",
            source=SourceEvidence(
                kind="image",
                text=_image_source_text(
                    state.section_path,
                    asset_id,
                    caption,
                    alt,
                ),
            ),
            format="asset_ref",
            content={"asset_id": asset_id, "caption": content_caption},
            metadata=common_metadata(
                "image",
                "asset_ref",
                section_path=state.section_path,
            ),
        )


class _HtmlParseState:
    def __init__(self) -> None:
        self._block_index = 1
        self._table_index = 1
        self._image_index = 1
        self._headings: list[tuple[int, str]] = []
        self.assets: list[PendingAsset] = []
        self.quality_warnings: list[dict[str, object]] = []

    @property
    def section_path(self) -> list[str]:
        return [text for _, text in self._headings]

    def next_block_id(self) -> str:
        block_id = f"b{self._block_index}"
        self._block_index += 1
        return block_id

    def next_table_id(self) -> str:
        table_id = f"t{self._table_index}"
        self._table_index += 1
        return table_id

    def next_image_id(self) -> str:
        image_id = f"img-{self._image_index:04d}"
        self._image_index += 1
        return image_id

    def set_heading(self, level: int, text: str) -> None:
        if not text:
            return
        while self._headings and self._headings[-1][0] >= level:
            self._headings.pop()
        self._headings.append((level, text))


def _tag_name(tag: Tag) -> str:
    return str(tag.name or "").lower()


def _text_with_links(
    node: Tag,
    *,
    preserve_pre: bool = False,
    skip_tags: set[str] | None = None,
) -> str:
    parts: list[str] = []
    skipped = skip_tags or set()

    def visit(current: object) -> None:
        if isinstance(current, NavigableString):
            parts.append(str(current))
            return
        if not isinstance(current, Tag):
            return
        name = _tag_name(current)
        if name in skipped:
            return
        if name in {"script", "style"}:
            return
        if name == "a":
            label = _normalize_whitespace("".join(current.stripped_strings))
            href = str(current.get("href") or "").strip()
            if label and href:
                parts.append(f"{label} ({href})")
            elif label:
                parts.append(label)
            elif href:
                parts.append(href)
            return
        for child in current.children:
            visit(child)

    visit(node)
    text = "".join(parts)
    if preserve_pre:
        return "\n".join(line.rstrip() for line in text.strip().splitlines()).strip()
    return _normalize_whitespace(text)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


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


def _figure_caption(figure: Tag) -> str | None:
    for child in figure.children:
        if isinstance(child, Tag) and _tag_name(child) == "figcaption":
            text = _text_with_links(child)
            return text or None
    return None


def _image_caption_from_context(image: Tag) -> str | None:
    figure = image.find_parent("figure")
    if not isinstance(figure, Tag):
        return None
    return _figure_caption(figure)


def _image_asset_ref(
    image: Tag,
    state: _HtmlParseState,
) -> tuple[str, str | None] | None:
    src = str(image.get("src") or "").strip()
    alt = str(image.get("alt") or "").strip() or None
    if not src:
        state.quality_warnings.append(
            {
                "type": "html_image_external_reference",
                "severity": "medium",
                "src": src,
                "message": "HTML image source did not contain embedded bytes.",
            }
        )
        return None
    match = _DATA_URI_RE.match(src)
    if match is None:
        state.quality_warnings.append(
            {
                "type": "html_image_external_reference",
                "severity": "medium",
                "src": src,
                "message": "HTML image references external bytes and was not fetched.",
            }
        )
        return None
    mime = match.group(1).lower()
    ext = _SUPPORTED_IMAGE_MIME.get(mime)
    if ext is None:
        state.quality_warnings.append(
            {
                "type": "html_image_mime_unsupported",
                "severity": "medium",
                "mime": mime,
                "message": f"HTML embedded image MIME type is unsupported: {mime}",
            }
        )
        return None
    try:
        data = base64.b64decode(match.group(2).strip(), validate=True)
    except (binascii.Error, ValueError):
        state.quality_warnings.append(
            {
                "type": "html_image_data_uri_invalid",
                "severity": "medium",
                "mime": mime,
                "message": "HTML embedded image data URI could not be decoded.",
            }
        )
        return None

    asset_id = state.next_image_id()
    state.assets.append(
        PendingAsset(
            id=asset_id,
            kind="image",
            data=data,
            mime=mime,
            ext=ext,
            metadata={"source": "html_data_uri"},
        )
    )
    return asset_id, alt


def _image_source_text(
    section_path: list[str],
    asset_id: str,
    caption: str | None,
    alt: str | None,
) -> str:
    lines: list[str] = []
    if section_path:
        lines.append(f"section: {' > '.join(section_path)}")
    lines.append(f"image: {asset_id}")
    if caption:
        lines.append(f"caption: {caption}")
    if alt:
        lines.append(f"alt: {alt}")
    return "\n".join(lines)


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
