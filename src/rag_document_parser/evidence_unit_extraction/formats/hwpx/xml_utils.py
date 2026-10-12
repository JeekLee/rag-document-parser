from __future__ import annotations

import re
from xml.etree import ElementTree as ET

_HP = "http://www.hancom.co.kr/hwpml/2011/paragraph"
_INLINE_SEPARATORS = {
    "lineBreak": "\n",
    "fwSpace": " ",
    "nbSpace": " ",
    "tab": "\t",
}


def q(local: str) -> str:
    return f"{{{_HP}}}{local}"


def paragraph_text(paragraph: ET.Element) -> str:
    parts: list[str] = []
    for run in paragraph.findall(q("run")):
        if run.find(q("tbl")) is not None or run.find(q("pic")) is not None:
            continue
        parts.append(run_text(run))
    return "".join(parts)


def run_text(run: ET.Element) -> str:
    # Only direct text nodes belong to this run; nested objects own their text.
    return "".join(_inline_text(text) for text in run.findall(q("t")))


def _inline_text(element: ET.Element) -> str:
    """Read one text node's mixed content, excluding its own external tail."""
    parts = ["".join(char for char in element.text or "" if char > "\x1f")]
    for child in element:
        parts.append(_INLINE_SEPARATORS.get(local_name(child.tag), ""))
        parts.append(_inline_text(child))
        parts.append("".join(char for char in child.tail or "" if char > "\x1f"))
    return "".join(parts)


def element_text(element: ET.Element) -> str:
    parts: list[str] = []
    for descendant in element.iter():
        if local_name(descendant.tag) != "t":
            continue
        text = _inline_text(descendant)
        if text:
            parts.append(text)
    return clean_text(" ".join(parts))


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag
