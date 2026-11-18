from __future__ import annotations

from collections.abc import Iterable, Iterator
from xml.etree import ElementTree as ET

from .paragraph_parts import paragraph_parts
from .xml_utils import q


def expand_layout_tables(
    paragraphs: Iterable[ET.Element],
) -> Iterator[ET.Element]:
    """Expand wrappers before table extraction separates text from children.

    Work on ordered XML, not the lossy canonical cell projection. Real data
    tables are emitted whole, so their nested cell evidence is unchanged.
    """
    for part in paragraph_parts(paragraphs):
        table = part.find(f"{q('run')}/{q('tbl')}")
        cell = _layout_cell(table) if table is not None else None
        if cell is None:
            yield part
        else:
            yield from expand_layout_tables(cell.findall(f"{q('subList')}/{q('p')}"))


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
