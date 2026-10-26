"""Geometry-based header refinement shared by the HWP table extractors."""
from __future__ import annotations

from typing import NamedTuple


class HeaderCellSpan(NamedTuple):
    start: int
    span: int
    text: str

    @property
    def end(self) -> int:
        return self.start + self.span


def refines_header_groups(
    row: list[HeaderCellSpan], previous: list[HeaderCellSpan],
) -> bool:
    """A lower header subdivides a group, not merely repeats its merged shape.

    Values in columns outside the groups identify a data row. An unchanged
    merged cell alone is not refinement: otherwise every similarly merged
    record (dates, diagnoses, etc.) is consumed into the header indefinitely.
    """
    groups = [cell for cell in previous if cell.span > 1 and cell.text.strip()]
    if not groups:
        return False
    populated = [cell for cell in row if cell.text.strip()]
    if any(not any(g.start <= c.start and c.end <= g.end for g in groups) for c in populated):
        return False
    refined = False
    for group in groups:
        children = [c for c in row if group.start <= c.start and c.end <= group.end]
        if not any(c.text.strip() for c in children):
            return False
        refined |= any(c.text.strip() and c.span < group.span for c in children)
    return refined
