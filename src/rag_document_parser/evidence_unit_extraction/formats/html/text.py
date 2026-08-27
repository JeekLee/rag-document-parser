from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import pairwise

from bs4 import BeautifulSoup
from bs4.element import Comment, NavigableString, Tag

from ....models import EvidenceUnit
from .source_projection import HtmlSourceProjector

_ALIGNMENT_GAP_RE = re.compile(
    r"(?<=\S)(?:[ \u00a0]*\t[ \t\u00a0]*|[ \u00a0]{3,})(?=\S)"
)
_ALIGNMENT_MIN_ROWS = 2
_ALIGNMENT_INLINE_TAGS = {
    "a", "abbr", "b", "em", "font", "i", "mark", "small", "span",
    "strong", "sub", "sup", "u",
}
_SOURCE_PROJECTOR = HtmlSourceProjector()
_with_section = _SOURCE_PROJECTOR.with_section


@dataclass(frozen=True)
class HtmlTextExtractor:
    """Extracts normalized or alignment-preserving text from HTML nodes."""

    def content_root(self, soup: BeautifulSoup) -> Tag:
        return _content_root(soup)

    def should_skip(self, tag: Tag) -> bool:
        return _should_skip_element(tag)

    def is_hira_title(self, tag: Tag) -> bool:
        return _is_hira_title_element(tag)

    def text(
        self,
        node: Tag,
        *,
        preserve_pre: bool = False,
        skip_tags: set[str] | None = None,
    ) -> str:
        return _text_with_links(
            node,
            preserve_pre=preserve_pre,
            skip_tags=skip_tags,
        )

    def normalize(self, text: str) -> str:
        return _normalize_whitespace(text)

    def aligned(self, children: list[object]) -> _AlignedTextSelection:
        return _aligned_text_selection(children)

    def contains_aligned(self, tag: Tag) -> bool:
        return _contains_aligned_text(tag)

    def append_link_href(
        self,
        units: list[EvidenceUnit],
        section_path: list[str],
        href: str,
    ) -> None:
        _append_link_href(units, section_path, href)


def _tag_name(tag: Tag) -> str:
    return str(tag.name or "").lower()


def _content_root(soup: BeautifulSoup) -> Tag:
    for selector in (".viewCont", "main", "article"):
        selected = soup.select_one(selector)
        if isinstance(selected, Tag):
            return selected
    return soup.body or soup


def _should_skip_element(tag: Tag) -> bool:
    name = _tag_name(tag)
    if name in {"footer", "header", "nav"}:
        return True
    marker = " ".join(
        [
            str(tag.get("id") or ""),
            " ".join(str(value) for value in tag.get("class", [])),
        ]
    ).lower()
    return any(
        token in marker
        for token in (
            "btnarea",
            "filebox",
            "popfooter",
            "popheader",
            "popupfooter",
            "popupheader",
        )
    )


def _is_hira_title_element(tag: Tag) -> bool:
    parent = tag.parent
    return (
        "title" in _class_names(tag)
        and isinstance(parent, Tag)
        and "viewCont" in _class_names(parent)
    )


def _class_names(tag: Tag) -> set[str]:
    raw_classes = tag.get("class", [])
    if isinstance(raw_classes, str):
        return {raw_classes}
    if isinstance(raw_classes, list):
        return {str(value) for value in raw_classes}
    return set()


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
            if isinstance(current, Comment):
                return
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
            if not preserve_pre and isinstance(child, Tag):
                parts.append(" ")

    visit(node)
    text = "".join(parts)
    if preserve_pre:
        return "\n".join(line.rstrip() for line in text.strip().splitlines()).strip()
    return _normalize_whitespace(text)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class _HtmlTextRow:
    child_indexes: list[int]
    text: str


@dataclass
class _AlignedTextSelection:
    rows_by_start_index: dict[int, str]
    child_indexes: set[int]


def _aligned_text_selection(children: list[object]) -> _AlignedTextSelection:
    rows_by_start_index: dict[int, str] = {}
    child_indexes: set[int] = set()
    for rows in _direct_text_row_groups(children):
        aligned_row_indexes = _aligned_row_indexes(rows)
        if not aligned_row_indexes:
            continue
        for row_index, row in enumerate(rows):
            if not row.child_indexes:
                continue
            rows_by_start_index[row.child_indexes[0]] = (
                row.text
                if row_index in aligned_row_indexes
                else _normalize_whitespace(row.text)
            )
            child_indexes.update(row.child_indexes)
    return _AlignedTextSelection(rows_by_start_index, child_indexes)


def _contains_aligned_text(tag: Tag) -> bool:
    if _aligned_text_selection(list(tag.children)).rows_by_start_index:
        return True
    return any(
        isinstance(child, Tag)
        and _tag_name(child) in _ALIGNMENT_INLINE_TAGS
        and _contains_aligned_text(child)
        for child in tag.children
    )


def _direct_text_row_groups(children: list[object]) -> list[list[_HtmlTextRow]]:
    """Build logical rows without mistaking comments or inline tags for rows."""
    groups: list[list[_HtmlTextRow]] = []
    rows: list[_HtmlTextRow] = []
    row_child_indexes: list[int] = []
    row_parts: list[str] = []
    has_line_break = False

    def flush_row(*, include_empty: bool = False) -> None:
        nonlocal row_child_indexes, row_parts
        if row_child_indexes or include_empty:
            rows.append(
                _HtmlTextRow(
                    child_indexes=row_child_indexes,
                    text=_preserve_alignment_whitespace("".join(row_parts)),
                )
            )
        row_child_indexes = []
        row_parts = []

    def flush_group() -> None:
        nonlocal rows, has_line_break
        flush_row()
        if has_line_break and rows:
            groups.append(rows)
        rows = []
        has_line_break = False

    for child_index, child in enumerate(children):
        if isinstance(child, Comment):
            continue
        if isinstance(child, NavigableString):
            row_child_indexes.append(child_index)
            row_parts.append(str(child))
            continue
        if isinstance(child, Tag) and _tag_name(child) == "br":
            has_line_break = True
            flush_row(include_empty=True)
            continue
        if isinstance(child, Tag) and _is_alignment_inline_tag(child):
            row_child_indexes.append(child_index)
            row_parts.append(_alignment_inline_text(child))
            continue
        flush_group()
    flush_group()
    return groups


def _is_alignment_inline_tag(tag: Tag) -> bool:
    if _tag_name(tag) not in _ALIGNMENT_INLINE_TAGS:
        return False
    return not any(
        isinstance(descendant, Tag)
        and _tag_name(descendant) in {"br", "figure", "img", "script", "style", "table"}
        for descendant in tag.descendants
    )


def _alignment_inline_text(tag: Tag) -> str:
    parts: list[str] = []
    for child in tag.children:
        if isinstance(child, Comment):
            continue
        if isinstance(child, NavigableString):
            parts.append(str(child))
        elif isinstance(child, Tag):
            parts.append(_alignment_inline_text(child))
    text = "".join(parts)
    if _tag_name(tag) != "a":
        return text
    label = _normalize_whitespace(text)
    href = str(tag.get("href") or "").strip()
    if label and href:
        return f"{label} ({href})"
    return label or href


def _aligned_row_indexes(rows: list[_HtmlTextRow]) -> set[int]:
    candidate_indexes = [
        index for index, row in enumerate(rows) if _alignment_gap_count(row.text) >= 2
    ]
    if len(candidate_indexes) < _ALIGNMENT_MIN_ROWS:
        return set()

    preserved: set[int] = set()
    cluster = [candidate_indexes[0]]
    for candidate_index in candidate_indexes[1:]:
        if _rows_share_alignment_cluster(rows, cluster[-1], candidate_index):
            cluster.append(candidate_index)
            continue
        _add_alignment_cluster(rows, cluster, preserved)
        cluster = [candidate_index]
    _add_alignment_cluster(rows, cluster, preserved)
    return preserved


def _rows_share_alignment_cluster(
    rows: list[_HtmlTextRow],
    left_index: int,
    right_index: int,
) -> bool:
    intervening = rows[left_index + 1 : right_index]
    if any(
        not left.text.strip() and not right.text.strip()
        for left, right in pairwise(intervening)
    ):
        return False
    content_rows = [row for row in intervening if row.text.strip()]
    if len(content_rows) > 2:
        return False
    if not content_rows:
        return len(intervening) <= 1
    return len(intervening) <= 5


def _add_alignment_cluster(
    rows: list[_HtmlTextRow],
    candidate_indexes: list[int],
    preserved: set[int],
) -> None:
    if len(candidate_indexes) < _ALIGNMENT_MIN_ROWS:
        return
    start = _adjacent_weak_alignment_row(rows, candidate_indexes[0], -1)
    end = _adjacent_weak_alignment_row(rows, candidate_indexes[-1], 1)
    for row_index in range(start, end + 1):
        preserved.add(row_index)


def _adjacent_weak_alignment_row(
    rows: list[_HtmlTextRow],
    boundary_index: int,
    direction: int,
) -> int:
    index = boundary_index + direction
    skipped_blank_rows = 0
    while 0 <= index < len(rows) and not rows[index].text.strip():
        skipped_blank_rows += 1
        if skipped_blank_rows > 1:
            return boundary_index
        index += direction
    if 0 <= index < len(rows) and _alignment_gap_count(rows[index].text) >= 1:
        return index
    return boundary_index


def _alignment_gap_count(text: str) -> int:
    return max(
        (
            len(_ALIGNMENT_GAP_RE.findall(line.strip()))
            for line in text.splitlines()
            if line.strip()
        ),
        default=0,
    )


def _preserve_alignment_whitespace(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(line.rstrip() for line in lines)


def _append_link_href(
    units: list[EvidenceUnit],
    section_path: list[str],
    href: str,
) -> None:
    if not href:
        return
    for unit in units:
        if unit.type != "text" or not isinstance(unit.content, str):
            continue
        linked_text = f"{unit.content} ({href})"
        unit.content = linked_text
        unit.source.text = _with_section(section_path, linked_text)


tag_name = _tag_name
