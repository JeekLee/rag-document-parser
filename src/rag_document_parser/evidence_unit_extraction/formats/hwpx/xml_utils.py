from __future__ import annotations

import re
from xml.etree import ElementTree as ET

_HP = "http://www.hancom.co.kr/hwpml/2011/paragraph"


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
    parts: list[str] = []
    for text in run.findall(q("t")):
        if text.text:
            parts.append("".join(char for char in text.text if char > "\x1f"))
    return "".join(parts)


def element_text(element: ET.Element) -> str:
    parts: list[str] = []
    for descendant in element.iter():
        if local_name(descendant.tag) != "t":
            continue
        if descendant.text:
            parts.append("".join(char for char in descendant.text if char > "\x1f"))
    return clean_text(" ".join(parts))


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag
