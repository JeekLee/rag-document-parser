from __future__ import annotations

import re
from dataclasses import dataclass

_KNOWN_HTML_TAGS = frozenset(
    """
    a abbr acronym address applet area article aside audio b base basefont bdi bdo
    bgsound big blink blockquote body br button canvas caption center cite code col
    colgroup command data datalist dd del details dfn dialog dir div dl dt em embed
    fencedframe fieldset figcaption figure font footer form frame frameset h1 h2 h3
    h4 h5 h6 head header hgroup hr html i iframe image img input ins isindex kbd
    keygen label legend li link main map mark marquee math menu menuitem meta meter
    multicol nav nextid nobr noembed noframes noscript object ol optgroup option
    output p param picture plaintext portal pre progress q rb rp rt rtc ruby s samp
    script search section select selectedcontent shadow slot small source spacer span
    strike strong style sub summary sup svg table tbody td template textarea tfoot th
    thead time title tr track tt u ul var video wbr xmp
    """.split()
)


@dataclass(frozen=True)
class HtmlRepairResult:
    html: str
    unescaped_less_than_count: int = 0
    pseudo_tag_count: int = 0


_HtmlRepairResult = HtmlRepairResult


@dataclass(frozen=True)
class HtmlRepairer:
    """Repairs unsafe tag-like source text before DOM parsing."""

    def repair(self, html: str) -> HtmlRepairResult:
        return _repair_unsafe_html_text(html)


def _repair_unsafe_html_text(html: str) -> _HtmlRepairResult:
    """Escape tag-like source text that ``html.parser`` would discard.

    A less-than sign followed by an ASCII letter enters HTML's tag parsing
    path. If that would-be tag runs into another ``<`` (usually a real closing
    tag), reaches EOF, or contains a comparison glyph in its name, it cannot be
    safely treated as markup. Complete tag-like text is also unsafe when its
    name is not a known HTML tag and it has neither self-closing syntax nor a
    matching end tag. Escaping only the opening delimiter preserves the source
    text while leaving genuine markup untouched.
    """
    chunks: list[str] = []
    copy_from = 0
    search_from = 0
    unescaped_less_than_count = 0
    pseudo_tag_count = 0

    while (less_than := html.find("<", search_from)) >= 0:
        repair_kind, search_from = _less_than_scan_result(html, less_than)
        if repair_kind is not None:
            chunks.extend((html[copy_from:less_than], "&lt;"))
            copy_from = less_than + 1
            if repair_kind == "pseudo_tag":
                pseudo_tag_count += 1
            else:
                unescaped_less_than_count += 1

    if not unescaped_less_than_count and not pseudo_tag_count:
        return _HtmlRepairResult(html=html)
    chunks.append(html[copy_from:])
    return _HtmlRepairResult(
        html="".join(chunks),
        unescaped_less_than_count=unescaped_less_than_count,
        pseudo_tag_count=pseudo_tag_count,
    )


def _less_than_scan_result(html: str, less_than: int) -> tuple[str | None, int]:
    name_start = less_than + 1
    if name_start >= len(html):
        return None, len(html)

    if html.startswith("<!--", less_than):
        comment_end = html.find("-->", name_start + 3)
        return None, len(html) if comment_end < 0 else comment_end + 3

    first = html[name_start]
    if first in {"!", "?", "/"}:
        markup_end = html.find(">", name_start + 1)
        return None, len(html) if markup_end < 0 else markup_end + 1
    if not first.isascii() or not first.isalpha():
        return None, name_start

    quote: str | None = None
    position = name_start
    while position < len(html):
        char = html[position]
        if quote is not None:
            if char == quote:
                quote = None
        elif char in {'"', "'"}:
            quote = char
        elif char == "<":
            return "unescaped_less_than", name_start
        elif char == ">":
            tag_source = html[name_start:position]
            tag_name = re.split(r"[\s/]", tag_source, maxsplit=1)[0]
            if "≤" in tag_name or "≥" in tag_name:
                return "unescaped_less_than", name_start
            if (
                tag_name.lower() in {"script", "style"}
                and not tag_source.rstrip().endswith("/")
            ):
                closing_tag = re.search(
                    rf"</\s*{re.escape(tag_name)}\b",
                    html[position + 1 :],
                    re.IGNORECASE,
                )
                if closing_tag is None:
                    return None, len(html)
                return None, position + 1 + closing_tag.start()
            if _is_preserved_markup_tag(
                html,
                tag_name,
                tag_source,
                search_from=position + 1,
            ):
                return None, position + 1
            return "pseudo_tag", name_start
        position += 1
    return "unescaped_less_than", name_start


def _is_preserved_markup_tag(
    html: str,
    tag_name: str,
    tag_source: str,
    *,
    search_from: int,
) -> bool:
    if tag_name.lower() in _KNOWN_HTML_TAGS:
        return True
    if tag_source.rstrip().endswith("/"):
        return True
    closing_tag = re.compile(
        rf"</\s*{re.escape(tag_name)}\s*>",
        re.IGNORECASE,
    )
    return closing_tag.search(html, search_from) is not None
