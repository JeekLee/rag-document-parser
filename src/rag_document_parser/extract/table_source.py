from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Callable

_DUPLICATE_LABEL_SUFFIX_RE = re.compile(r"^(?P<label>.+) \[\d+\]$")


def build_column_source_labels(
    columns: list[dict[str, object]],
    label_fn: Callable[[dict[str, object]], str],
    rows: list[dict[str, object]] | None = None,
) -> dict[str, str]:
    labels = [(str(column["id"]), label_fn(column)) for column in columns]
    labels_by_id = dict(labels)
    counts = Counter(label for _, label in labels if is_semantic_column_label(label))
    ambiguous_labels = _ambiguous_body_labels(labels_by_id, counts, rows)
    ordinals: defaultdict[str, int] = defaultdict(int)

    result: dict[str, str] = {}
    for column_id, label in labels:
        if label in ambiguous_labels:
            ordinals[label] += 1
            label = f"{label} [{ordinals[label]}]"
        result[column_id] = label
    return result


def is_semantic_column_label(label: str) -> bool:
    return bool(label) and not label.startswith("col ")


def common_semantic_header_prefix(labels: list[str]) -> str | None:
    if (
        len(labels) < 2
        or not all(is_semantic_column_label(label) for label in labels)
    ):
        return None
    split_labels = [_source_label_base(label).split(" / ") for label in labels]
    prefix: list[str] = []
    for parts in zip(*split_labels):
        if len(set(parts)) != 1:
            break
        prefix.append(parts[0])
    if not prefix:
        return None
    return " / ".join(prefix)


def semantic_column_group_label(labels: list[str]) -> str | None:
    if (
        len(labels) < 2
        or not all(is_semantic_column_label(label) for label in labels)
    ):
        return None
    groups: list[str] = []
    for label in labels:
        group = _source_label_base(label).split(" / ", 1)[0]
        if group and group not in groups:
            groups.append(group)
    if not groups:
        return None
    return " / ".join(groups)


def _ambiguous_body_labels(
    labels_by_id: dict[str, str],
    counts: Counter[str],
    rows: list[dict[str, object]] | None,
) -> set[str]:
    if rows is None:
        return {label for label, count in counts.items() if count > 1}

    ambiguous: set[str] = set()
    for row in rows:
        row_labels: list[str] = []
        for cell in row.get("cells", []):
            if int(cell.get("colspan", 1)) != 1:
                continue
            if not _cell_has_source_content(cell):
                continue
            label = labels_by_id.get(str(cell.get("column_id")))
            if label is not None and counts[label] > 1:
                row_labels.append(label)
        ambiguous.update(
            label
            for label, count in Counter(row_labels).items()
            if count > 1 and is_semantic_column_label(label)
        )
    return ambiguous


def _cell_has_source_content(cell: dict[str, object]) -> bool:
    text = cell.get("text")
    return bool((str(text) if text is not None else "") or cell.get("children"))


def _source_label_base(label: str) -> str:
    match = _DUPLICATE_LABEL_SUFFIX_RE.match(label)
    if match is None:
        return label
    return match.group("label")
