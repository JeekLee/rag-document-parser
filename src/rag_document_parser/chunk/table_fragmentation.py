"""Lossless fragments of an oversized table row, with cell-level provenance."""
from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from copy import deepcopy

from ..models import EvidenceItem
from .evidence_projection import _plain
from .tokens import count_tokens


def split_oversized_row(
    item: EvidenceItem, max_tokens: int, render: Callable[[EvidenceItem], str],
) -> list[EvidenceItem]:
    """Keep headers and short keys; split text without aligning unrelated cells.

    Offsets are zero-based Python string positions, with exclusive ends. Child
    objects stay atomic and occur once. If headers or an atomic child cannot
    fit, return the original row so the caller can report the size warning.
    """
    if not isinstance(item.content, Mapping) or len(item.content.get("rows", [])) != 1:
        return [item]
    if count_tokens(render(item)) <= max_tokens:
        return [item]
    table = _plain(item.content)
    table.pop("compact", None)
    row = table["rows"][0]
    cells = row["cells"]
    repeated = {
        i for i, cell in enumerate(cells)
        if not cell.get("children") and count_tokens(cell.get("text", "")) <= min(80, max_tokens // 8)
    }

    def build(selections: dict[int, tuple[int, int, int, int]], first: bool) -> EvidenceItem:
        content = deepcopy(table)
        for i, cell in enumerate(content["rows"][0]["cells"]):
            if i in repeated:
                start, end, child_start, child_end = 0, len(cells[i]["text"]), 0, 0
                role = "content" if first else "context"
            else:
                start, end, child_start, child_end = selections.get(i, (0, 0, 0, 0))
                role = "content" if i in selections else "omitted"
            cell["text"] = cells[i]["text"][start:end]
            cell["children"] = deepcopy(cells[i].get("children", [])[child_start:child_end])
            cell.setdefault("metadata", {})["cell_fragment"] = {
                "role": role, "text_range": [start, end],
                "children_range": [child_start, child_end],
            }
        metadata = deepcopy(item.metadata)
        metadata["table_fragment"] = {"row_index": row["index"]}
        return EvidenceItem(
            type=item.type, format=item.format, content=content,
            source_unit_ids=list(item.source_unit_ids), metadata=metadata,
        )

    # A wide row may have many small values. Do not let repeated context use
    # the entire budget; retain those values once as ordinary content instead.
    if count_tokens(render(build({}, False))) + 8 >= max_tokens:
        repeated.clear()

    fragments: list[dict[int, tuple[int, int, int, int]]] = []

    def fits(selection: dict[int, tuple[int, int, int, int]], *, first: bool | None = None) -> bool:
        return count_tokens(render(build(selection, not fragments if first is None else first))) <= max_tokens

    for i, cell in enumerate(cells):
        if i in repeated:
            continue
        text = cell.get("text", "")
        start = 0
        while start < len(text):
            # Always retain a candidate known to fit: tokenizer counts need
            # not be strictly monotone as characters are appended.
            low, high, end = start + 1, len(text), start
            while low <= high:
                middle = (low + high) // 2
                if fits({i: (start, middle, 0, 0)}):
                    end, low = middle, middle + 1
                else:
                    high = middle - 1
            if end == start:
                return [item]
            if end < len(text):
                boundary = _natural_boundary(text, start, end)
                if fits({i: (start, boundary, 0, 0)}):
                    end = boundary
            fragments.append({i: (start, end, 0, 0)})
            start = end

        for child_index, _ in enumerate(cell.get("children", [])):
            selection = {i: (len(text), len(text), child_index, child_index + 1)}
            if fragments and i in fragments[-1]:
                text_start, text_end, child_start, child_end = fragments[-1][i]
                joined = {i: (text_start, text_end, child_start, child_index + 1)}
                if child_end == child_index and fits(joined, first=len(fragments) == 1):
                    fragments[-1] = joined
                    continue
            if not fits(selection):
                return [item]
            fragments.append(selection)

    if len(fragments) <= 1:
        return [item]
    result = []
    for index, selection in enumerate(fragments, start=1):
        fragment = build(selection, index == 1)
        fragment.metadata["table_fragment"].update(part=index, count=len(fragments))
        result.append(fragment)
    return result


def _natural_boundary(text: str, start: int, end: int) -> int:
    """Prefer paragraph, sentence, then word boundaries without dropping space."""
    lower = start + (end - start) // 2
    newline = text.rfind("\n", lower, end)
    if newline >= lower:
        return newline + 1
    matches = list(re.finditer(r"[.!?。！？](?:\s+|$)", text[lower:end]))
    if matches:
        return lower + matches[-1].end()
    spaces = list(re.finditer(r"\s+", text[lower:end]))
    return lower + spaces[-1].end() if spaces else end
