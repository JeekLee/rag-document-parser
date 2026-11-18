from __future__ import annotations

from collections.abc import Iterable, Iterator
from xml.etree import ElementTree as ET

from .diagram import is_drawing_element
from .xml_utils import q


def paragraph_parts(paragraphs: Iterable[ET.Element]) -> Iterator[ET.Element]:
    """Yield text, tables, pictures and drawing groups in XML order.

    Adjacent drawing objects stay together so connectors can still be inferred.
    Only runs and drawing containers are expanded. Table cells and drawing
    labels stay with their objects; notes are not read as surrounding text.
    The original XML is left intact.
    """
    for paragraph in paragraphs:
        pending = ET.Element(paragraph.tag, paragraph.attrib)
        pending_kind: str | None = None
        pending_run: ET.Element | None = None
        for run in paragraph.findall(q("run")):
            pending_run = None
            for child in _run_items(run):
                if child.tag in {q("tbl"), q("pic")}:
                    if pending_kind is not None:
                        yield pending
                    part = ET.Element(paragraph.tag, paragraph.attrib)
                    ET.SubElement(part, run.tag, run.attrib).append(child)
                    yield part
                    pending = ET.Element(paragraph.tag, paragraph.attrib)
                    pending_kind = None
                    pending_run = None
                    continue
                kind = "text" if child.tag == q("t") else "drawing"
                # Empty text runs commonly separate shapes in a single diagram.
                if (
                    kind == "text"
                    and pending_kind != "text"
                    and not "".join(child.itertext()).strip()
                ):
                    continue
                if pending_kind is not None and kind != pending_kind:
                    yield pending
                    pending = ET.Element(paragraph.tag, paragraph.attrib)
                    pending_run = None
                pending_kind = kind
                if pending_run is None:
                    pending_run = ET.SubElement(pending, run.tag, run.attrib)
                pending_run.append(child)
        if pending_kind is not None:
            yield pending


def _run_items(element: ET.Element) -> Iterator[ET.Element]:
    for child in element:
        if child.tag == q("container"):
            yield from _run_items(child)
        elif child.tag in {q("t"), q("tbl"), q("pic")} or is_drawing_element(child):
            yield child
