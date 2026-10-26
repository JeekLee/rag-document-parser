from __future__ import annotations

from collections.abc import Iterable, Iterator
from xml.etree import ElementTree as ET

from .xml_utils import q


def expand_layout_tables(
    paragraphs: Iterable[ET.Element],
    *,
    inside_layout: bool = False,
) -> Iterator[ET.Element]:
    """Expand wrappers before table extraction separates text from children.

    Work on ordered XML, not the lossy canonical cell projection. Real data
    tables are emitted whole, so their nested cell evidence is unchanged.
    """
    for paragraph in paragraphs:
        tables = paragraph.findall(f"{q('run')}/{q('tbl')}")
        pictures = paragraph.findall(f"{q('run')}/{q('pic')}")
        has_layout = any(_layout_cell(table) is not None for table in tables)
        if not (tables or pictures) or (not inside_layout and not has_layout):
            yield paragraph
            continue
        for part in _split_at_objects(paragraph):
            table = part.find(f"{q('run')}/{q('tbl')}")
            cell = _layout_cell(table) if table is not None else None
            if cell is None:
                yield part
            else:
                yield from expand_layout_tables(
                    cell.findall(f"{q('subList')}/{q('p')}"), inside_layout=True,
                )


def _layout_cell(table: ET.Element) -> ET.Element | None:
    rows = table.findall(q("tr"))
    if len(rows) != 1:
        return None
    cells = rows[0].findall(q("tc"))
    if len(cells) != 1:
        return None
    cell = cells[0]
    try:
        if any(int(table.get(name, "1")) != 1 for name in ("rowCnt", "colCnt")):
            return None
        for tag, names, default in (
            ("cellSpan", ("rowSpan", "colSpan"), 1),
            ("cellAddr", ("rowAddr", "colAddr"), 0),
        ):
            element = cell.find(q(tag))
            for name in names:
                value = cell.get(name)
                if value is None and element is not None:
                    value = element.get(name)
                if value is not None and int(value) != default:
                    return None
    except ValueError:
        return None
    return cell if cell.find(f"{q('subList')}/.//{q('tbl')}") is not None else None


def _split_at_objects(paragraph: ET.Element) -> Iterator[ET.Element]:
    pending = ET.Element(paragraph.tag, paragraph.attrib)
    for node in paragraph:
        if node.tag != q("run"):
            pending.append(node)
            continue
        run = ET.SubElement(pending, node.tag, node.attrib)
        for child in node:
            if child.tag not in {q("tbl"), q("pic")}:
                run.append(child)
                continue
            yield pending
            part = ET.Element(paragraph.tag, paragraph.attrib)
            ET.SubElement(part, node.tag, node.attrib).append(child)
            yield part
            pending = ET.Element(paragraph.tag, paragraph.attrib)
            run = ET.SubElement(pending, node.tag, node.attrib)
    yield pending
