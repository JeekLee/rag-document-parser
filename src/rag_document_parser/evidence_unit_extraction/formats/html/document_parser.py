from __future__ import annotations

from bs4 import BeautifulSoup
from bs4.element import Comment, NavigableString, Tag

from ....models import EvidenceUnit, SourceEvidence
from ...backend import ParsedDocument
from ...schema import common_metadata
from .assets import HtmlImageExtractor
from .repair import HtmlRepairer
from .source_projection import HtmlSourceProjector
from .state import HtmlParseState
from .table_extraction import HtmlTableExtractor
from .text import HtmlTextExtractor
from .text import tag_name as _tag_name

_BLOCK_TEXT_TAGS = {"blockquote", "p"}
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_REPAIRER = HtmlRepairer()
_SOURCE_PROJECTOR = HtmlSourceProjector()
_TEXT_EXTRACTOR = HtmlTextExtractor()
_IMAGE_EXTRACTOR = HtmlImageExtractor()
_TABLE_EXTRACTOR = HtmlTableExtractor()
_HtmlParseState = HtmlParseState
_repair_unsafe_html_text = _REPAIRER.repair
_content_root = _TEXT_EXTRACTOR.content_root
_should_skip_element = _TEXT_EXTRACTOR.should_skip
_is_hira_title_element = _TEXT_EXTRACTOR.is_hira_title
_text_with_links = _TEXT_EXTRACTOR.text
_normalize_whitespace = _TEXT_EXTRACTOR.normalize
_aligned_text_selection = _TEXT_EXTRACTOR.aligned
_contains_aligned_text = _TEXT_EXTRACTOR.contains_aligned
_append_link_href = _TEXT_EXTRACTOR.append_link_href
_parse_table_content = _TABLE_EXTRACTOR.extract
_figure_caption = _IMAGE_EXTRACTOR.figure_caption
_image_asset_ref = _IMAGE_EXTRACTOR.reference
_image_source_text = _IMAGE_EXTRACTOR.source_text
_table_source_text = _SOURCE_PROJECTOR.table
_with_section = _SOURCE_PROJECTOR.with_section


class HtmlDocumentParser:
    supported_suffixes = (".html", ".htm")

    def parse(self, data: bytes, suffix: str) -> ParsedDocument:
        html = data.decode("utf-8", errors="replace")
        repairs = _repair_unsafe_html_text(html)
        soup = BeautifulSoup(repairs.html, "html.parser")
        root = _content_root(soup)
        state = _HtmlParseState()
        if repairs.unescaped_less_than_count:
            state.quality_warnings.append(
                {
                    "type": "html_unescaped_less_than_repaired",
                    "severity": "medium",
                    "count": repairs.unescaped_less_than_count,
                    "message": (
                        "Repaired unescaped less-than text before HTML parsing to "
                        "prevent content loss."
                    ),
                }
            )
        if repairs.pseudo_tag_count:
            state.quality_warnings.append(
                {
                    "type": "html_pseudo_tag_repaired",
                    "severity": "medium",
                    "count": repairs.pseudo_tag_count,
                    "message": (
                        "Repaired unpaired tag-like text before HTML parsing to "
                        "prevent content loss."
                    ),
                }
            )
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
        children = list(parent.children)
        aligned_text = _aligned_text_selection(children)
        for child_index, child in enumerate(children):
            if child_index in aligned_text.rows_by_start_index:
                self._append_text_unit(
                    units,
                    state,
                    aligned_text.rows_by_start_index[child_index],
                )
                continue
            if child_index in aligned_text.child_indexes:
                continue
            if isinstance(child, NavigableString):
                if isinstance(child, Comment):
                    continue
                self._append_text_unit(units, state, _normalize_whitespace(str(child)))
                continue
            if not isinstance(child, Tag):
                continue
            if _should_skip_element(child):
                continue
            name = _tag_name(child)
            if name in {"script", "style"}:
                continue
            if _is_hira_title_element(child):
                title = _text_with_links(child)
                self._append_text_unit(units, state, title)
                state.set_heading(1, title)
                continue
            if name in _HEADING_TAGS:
                state.set_heading(int(name[1]), _text_with_links(child))
                continue
            if name == "a":
                if _contains_aligned_text(child):
                    first_link_unit = len(units)
                    self._walk_blocks(child, state, units)
                    _append_link_href(
                        units[first_link_unit:],
                        state.section_path,
                        str(child.get("href") or "").strip(),
                    )
                else:
                    self._append_text_unit(units, state, _text_with_links(child))
                continue
            if name in _BLOCK_TEXT_TAGS:
                if _contains_aligned_text(child):
                    self._walk_blocks(child, state, units)
                else:
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
