from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag

from ....models import EvidenceUnit, SourceEvidence
from ...backend import ParsedDocument
from ...schema import common_metadata

_BLOCK_TEXT_TAGS = {"blockquote", "p"}
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}


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
        return ParsedDocument(units=units)

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


class _HtmlParseState:
    def __init__(self) -> None:
        self._block_index = 1
        self._headings: list[tuple[int, str]] = []

    @property
    def section_path(self) -> list[str]:
        return [text for _, text in self._headings]

    def next_block_id(self) -> str:
        block_id = f"b{self._block_index}"
        self._block_index += 1
        return block_id

    def set_heading(self, level: int, text: str) -> None:
        if not text:
            return
        while self._headings and self._headings[-1][0] >= level:
            self._headings.pop()
        self._headings.append((level, text))


def _tag_name(tag: Tag) -> str:
    return str(tag.name or "").lower()


def _text_with_links(node: Tag, *, preserve_pre: bool = False) -> str:
    parts: list[str] = []

    def visit(current: object) -> None:
        if isinstance(current, NavigableString):
            parts.append(str(current))
            return
        if not isinstance(current, Tag):
            return
        name = _tag_name(current)
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


def _with_section(section_path: list[str], text: str) -> str:
    if not section_path:
        return text
    return f"section: {' > '.join(section_path)}\n{text}"
