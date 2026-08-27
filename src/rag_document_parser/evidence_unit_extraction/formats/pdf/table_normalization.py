from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

_CJK = re.compile(r"[가-힣一-鿿㐀-䶿]")


@dataclass(frozen=True)
class PdfTableNormalizer:
    """Applies semantic repairs to canonical PDF table payloads."""

    def simple_cell(
        self,
        column_id: str,
        text: str,
        *,
        rowspan: int = 1,
        colspan: int = 1,
    ) -> dict[str, object]:
        return _simple_cell(
            column_id,
            text,
            rowspan=rowspan,
            colspan=colspan,
        )

    def column_signature(self, table: dict[str, object]) -> tuple[str, ...]:
        return _table_column_signature(table)

    def normalize_header(self, text: str) -> str:
        return _normalize_header_text(text)

    def is_table_of_contents(self, table: dict[str, object]) -> bool:
        return _is_table_of_contents(table)

    def cell_has_content(self, cell: object) -> bool:
        return _cell_has_content(cell)

    def merge_nested_children(
        self,
        children: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        return _merge_nested_table_children(children)

    def promote_ultrasound_codes(self, table: dict[str, object]) -> None:
        _promote_ultrasound_code_matrix(table)

    def expand_parallel_rows(self, table: dict[str, object]) -> None:
        _expand_parallel_code_action_rows(table)

    def normalize(self, table: dict[str, object]) -> None:
        _merge_wrapped_table_rows(table)
        _promote_code_table_leaf_headers(table)
        _promote_ultrasound_code_matrix(table)
        _expand_parallel_code_action_rows(table)

    def append_rows(
        self,
        target: dict[str, object],
        continuation: dict[str, object],
    ) -> None:
        _append_table_rows(target, continuation)


def _simple_cell(
    column_id: str,
    text: str,
    *,
    rowspan: int = 1,
    colspan: int = 1,
) -> dict[str, object]:
    return {
        "column_id": column_id,
        "text": text,
        "rowspan": rowspan,
        "colspan": colspan,
        "children": [],
    }


def _table_column_signature(table: dict[str, object]) -> tuple[str, ...]:
    return tuple(
        _normalize_header_text(str(column.get("text", "")))
        for column in table.get("columns", [])
    )


def _append_table_rows(
    target: dict[str, object],
    continuation: dict[str, object],
) -> None:
    target_rows = target.get("rows")
    continuation_rows = continuation.get("rows")
    if not isinstance(target_rows, list) or not isinstance(continuation_rows, list):
        return
    for row in continuation_rows:
        if not isinstance(row, Mapping):
            continue
        copied = dict(row)
        copied["index"] = len(target_rows) + 1
        target_rows.append(copied)
    _merge_wrapped_table_rows(target)
    _promote_code_table_leaf_headers(target)
    _promote_ultrasound_code_matrix(target)
    _expand_parallel_code_action_rows(target)


def _merge_wrapped_table_rows(table: dict[str, object]) -> None:
    if _is_table_of_contents(table):
        return
    rows = table.get("rows")
    if not isinstance(rows, list) or len(rows) < 2:
        return
    if not _has_leading_column_content(rows):
        return

    merged: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if merged and _is_wrapped_table_row(row):
            _append_wrapped_row_cells(merged[-1], row)
            continue
        row["index"] = len(merged) + 1
        merged.append(row)

    table["rows"] = merged


def _has_leading_column_content(rows: list[object]) -> bool:
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        cells = row.get("cells")
        if not isinstance(cells, list):
            continue
        for cell in cells:
            if (
                isinstance(cell, Mapping)
                and str(cell.get("column_id", "")) == "c1"
                and _cell_has_content(cell)
            ):
                return True
    return False


def _is_table_of_contents(table: dict[str, object]) -> bool:
    columns = table.get("columns")
    if not isinstance(columns, list):
        return False
    labels = [
        _normalize_header_text(str(column.get("text", "")))
        for column in columns
        if isinstance(column, Mapping)
    ]
    return "제목" in labels and "페이지" in labels


def _promote_code_table_leaf_headers(table: dict[str, object]) -> None:
    columns = table.get("columns")
    header_rows = table.get("header_rows")
    rows = table.get("rows")
    if (
        not isinstance(columns, list)
        or len(columns) < 2
        or not isinstance(header_rows, list)
        or len(header_rows) != 1
        or not isinstance(rows, list)
        or not rows
    ):
        return

    labels = [
        _normalize_header_text(str(column.get("text", "")))
        for column in columns
        if isinstance(column, Mapping)
    ]
    if len(labels) != len(columns) or labels[0] != "질병코드" or any(labels[1:]):
        return

    first_row = rows[0]
    if not isinstance(first_row, Mapping):
        return
    first_cells = first_row.get("cells")
    if not isinstance(first_cells, list) or len(first_cells) != len(columns):
        return
    leaf_values = [str(cell.get("text", "")).strip() for cell in first_cells]
    if not leaf_values or not all(_looks_like_disease_code(value) for value in leaf_values):
        return

    for index, column in enumerate(columns):
        if isinstance(column, Mapping):
            column["text"] = f"질병코드 / {leaf_values[index]}"

    header_rows[0] = {
        "index": 1,
        "cells": [
            {
                "column_id": "c1",
                "text": "질병코드",
                "rowspan": 1,
                "colspan": len(columns),
                "children": [],
            }
        ],
    }
    header_rows.append(
        {
            "index": 2,
            "cells": [
                _simple_cell(str(column.get("id", f"c{index + 1}")), leaf_values[index])
                for index, column in enumerate(columns)
                if isinstance(column, Mapping)
            ],
        }
    )

    table["rows"] = [
        _reindexed_row(row, index + 1)
        for index, row in enumerate(rows[1:])
        if isinstance(row, Mapping)
    ]


def _looks_like_disease_code(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]\d{2}(?:\.\d+)?(?:~[A-Z]?\d{2}(?:\.\d+)?)?", value))


def _promote_ultrasound_code_matrix(table: dict[str, object]) -> None:
    columns = table.get("columns")
    rows = table.get("rows")
    if (
        not isinstance(columns, list)
        or [str(column.get("text", "")) for column in columns if isinstance(column, Mapping)]
        != ["구분", "EDI코드"]
        or not isinstance(rows, list)
        or len(rows) < 3
    ):
        return

    row_cells = [_row_cell_texts(row) for row in rows]
    if len(row_cells[0]) != 2 or set(_edi_codes(row_cells[0][1])) != {"EB401", "EB402"}:
        return

    promoted_rows: list[dict[str, object]] = []
    for cells in row_cells:
        if len(cells) != 2:
            return
        promoted = _ultrasound_code_rows(cells[0], cells[1])
        if promoted is None:
            return
        for group, label, code in promoted:
            promoted_rows.append(
                {
                    "index": len(promoted_rows) + 1,
                    "cells": [
                        _simple_cell("c1", group),
                        _simple_cell("c2", label),
                        _simple_cell("c3", code),
                    ],
                }
            )

    if not promoted_rows:
        return

    table["columns"] = [
        {"id": "c1", "text": "구분"},
        {"id": "c2", "text": "구분"},
        {"id": "c3", "text": "EDI코드"},
    ]
    table["header_rows"] = [
        {
            "index": 1,
            "cells": [
                {
                    "column_id": "c1",
                    "text": "구분",
                    "rowspan": 1,
                    "colspan": 2,
                    "children": [],
                },
                _simple_cell("c3", "EDI코드"),
            ],
        },
    ]
    table["rows"] = _collapse_leading_empty_rowspan_cells(promoted_rows, "c1")


def _collapse_leading_empty_rowspan_cells(
    rows: list[dict[str, object]],
    column_id: str,
) -> list[dict[str, object]]:
    active: dict[str, object] | None = None
    for row in rows:
        cells = row.get("cells")
        if not isinstance(cells, list):
            active = None
            continue
        target_index = next(
            (
                index
                for index, cell in enumerate(cells)
                if isinstance(cell, Mapping) and cell.get("column_id") == column_id
            ),
            None,
        )
        if target_index is None:
            active = None
            continue
        cell = cells[target_index]
        if not isinstance(cell, Mapping):
            active = None
            continue
        if str(cell.get("text", "")).strip():
            active = cell
            continue
        if active is None:
            continue
        active["rowspan"] = int(active.get("rowspan", 1)) + 1
        del cells[target_index]
    return rows


def _row_cell_texts(row: object) -> list[str]:
    if not isinstance(row, Mapping):
        return []
    cells = row.get("cells")
    if not isinstance(cells, list):
        return []
    return [
        str(cell.get("text", "")).strip()
        for cell in cells
        if isinstance(cell, Mapping)
    ]


def _ultrasound_code_rows(
    label_text: str,
    code_text: str,
) -> list[tuple[str, str, str]] | None:
    codes = _edi_codes(code_text)
    if len(codes) < 2:
        return None
    group_word = _ultrasound_group_word(label_text)
    if not group_word:
        return None

    labels = _known_ultrasound_labels(codes)
    if labels is None:
        labels = _ultrasound_labels_from_text(label_text, group_word, len(codes))
    if len(labels) != len(codes) or any(not label for label in labels):
        return None

    rows: list[tuple[str, str, str]] = []
    for index, (label, code) in enumerate(zip(labels, codes, strict=True)):
        group = f"{group_word} 초음파" if index == 0 else ""
        rows.append((group, label, code))
    return rows


def _ultrasound_group_word(text: str) -> str:
    padded = f" {text} "
    for word in ("기본", "진단", "제한적", "특수"):
        if f" {word} " in padded:
            return word
    return ""


def _known_ultrasound_labels(codes: list[str]) -> list[str] | None:
    labels: list[str] = []
    for code in codes:
        label = _ULTRASOUND_EDI_LABELS.get(_base_ultrasound_edi_code(code))
        if label is None:
            return None
        labels.append(label)
    return labels


def _base_ultrasound_edi_code(code: str) -> str:
    return code[:-3] if code.endswith("001") else code


def _ultrasound_labels_from_text(
    label_text: str,
    group_word: str,
    count: int,
) -> list[str]:
    without_group = re.sub(
        rf"(^|\s){re.escape(group_word)}(?=\s|$)",
        " ",
        label_text,
        count=1,
    ).strip()
    without_separator = re.sub(
        r"\s+초음파\s+",
        " ",
        without_group,
        count=1,
    ).strip()
    return _split_ultrasound_label_tail(without_separator, count)


def _split_ultrasound_label_tail(text: str, count: int) -> list[str]:
    if count <= 0:
        return []
    tokens = text.split()
    if count == 1:
        return [text.strip()]
    if len(tokens) <= count:
        return tokens
    return [*tokens[: count - 1], " ".join(tokens[count - 1:])]


_ULTRASOUND_EDI_LABELS = {
    "EB411": "안구",
    "EB412": "안와",
    "EB414": "갑상선·부갑상선",
    "EB415": "갑상선·부갑상선 제외한 경부",
    "EB421": "유방·액와부-일반",
    "EB422": "흉벽, 흉막, 늑골 등",
    "EB423": "유방·액와부-정밀",
    "EB424": "자동유방초음파",
    "EB430": "선천성 심질환 경흉부",
    "EB431": "경흉부-단순",
    "EB432": "경흉부-일반",
    "EB433": "경흉부-전문",
    "EB434": "부하-약물부하",
    "EB435": "부하-운동부하",
    "EB436": "태아정밀",
    "EB441": "간·담낭·담도·비장·췌장(일반)",
    "EB442": "간·담낭·담도·비장·췌장(정밀)",
    "EB443": "충수",
    "EB444": "소장·대장",
    "EB445": "서혜부",
    "EB446": "직장·항문",
    "EB447": "항문",
    "EB448": "신장·부신·방광",
    "EB449": "신장·부신",
    "EB450": "방광",
    "EB610": "선천성 심질환 경식도",
    "EB611": "경식도",
    "EB612": "심장내",
}


def _edi_codes(text: str) -> list[str]:
    return re.findall(r"\bEB\d{3}(?:001)?\b", text)


def _expand_parallel_code_action_rows(table: dict[str, object]) -> None:
    columns = table.get("columns")
    rows = table.get("rows")
    if (
        not isinstance(columns, list)
        or [str(column.get("text", "")) for column in columns if isinstance(column, Mapping)]
        != ["분류", "코드", "행위명"]
        or not isinstance(rows, list)
    ):
        return

    expanded_rows: list[dict[str, object]] = []
    changed = False
    for row in rows:
        cells = _row_cell_texts(row)
        if len(cells) != 3:
            continue
        codes = _parallel_cell_parts(cells[1])
        actions = _parallel_cell_parts(cells[2])
        if len(codes) == len(actions) and len(codes) > 1:
            changed = True
            for index, (code, action) in enumerate(zip(codes, actions, strict=True)):
                expanded_rows.append(
                    {
                        "index": len(expanded_rows) + 1,
                        "cells": [
                            _simple_cell("c1", cells[0] if index == 0 else ""),
                            _simple_cell("c2", code),
                            _simple_cell("c3", action),
                        ],
                    }
                )
            continue
        if isinstance(row, Mapping):
            copied = dict(row)
            copied["index"] = len(expanded_rows) + 1
            expanded_rows.append(copied)

    if changed:
        table["rows"] = expanded_rows


def _parallel_cell_parts(text: str) -> list[str]:
    line_parts = [part.strip() for part in text.splitlines() if part.strip()]
    if len(line_parts) > 1:
        return line_parts
    return _slash_parts(text)


def _slash_parts(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\s*/\s*", text) if part.strip()]


def _reindexed_row(row: dict[str, object], index: int) -> dict[str, object]:
    copied = dict(row)
    copied["index"] = index
    return copied


def _is_wrapped_table_row(row: dict[str, object]) -> bool:
    cells = row.get("cells")
    if not isinstance(cells, list) or not cells:
        return False

    first = cells[0]
    if _cell_has_content(first):
        return False

    content_cells = [
        cell
        for cell in cells
        if isinstance(cell, Mapping) and _cell_has_content(cell)
    ]
    if not content_cells:
        return False
    if len(content_cells) == 1:
        return True

    second = cells[1] if len(cells) > 1 else None
    if not _cell_has_text(second):
        return False
    return not _starts_new_table_subrow(str(second.get("text", "")))


def _append_wrapped_row_cells(
    target_row: dict[str, object],
    wrapped_row: dict[str, object],
) -> None:
    target_cells = target_row.get("cells")
    wrapped_cells = wrapped_row.get("cells")
    if not isinstance(target_cells, list) or not isinstance(wrapped_cells, list):
        return

    target_by_column = {
        str(cell.get("column_id")): cell
        for cell in target_cells
        if isinstance(cell, Mapping)
    }
    for wrapped_cell in wrapped_cells:
        if not isinstance(wrapped_cell, Mapping) or not _cell_has_content(wrapped_cell):
            continue
        target_cell = target_by_column.get(str(wrapped_cell.get("column_id")))
        if not isinstance(target_cell, Mapping):
            continue
        wrapped_text = str(wrapped_cell.get("text", "")).strip()
        if wrapped_text:
            target_text = str(target_cell.get("text", "")).strip()
            target_cell["text"] = (
                f"{target_text}\n{wrapped_text}"
                if target_text
                else wrapped_text
            )
        children = wrapped_cell.get("children")
        if isinstance(children, list) and children:
            target_children = target_cell.get("children")
            if isinstance(target_children, list):
                target_children.extend(children)
                target_cell["children"] = _merge_nested_table_children(target_children)
            else:
                target_cell["children"] = _merge_nested_table_children(list(children))


def _cell_has_content(cell: object) -> bool:
    if not isinstance(cell, Mapping):
        return False
    return _cell_has_text(cell) or bool(cell.get("children"))


def _cell_has_text(cell: object) -> bool:
    return isinstance(cell, Mapping) and bool(str(cell.get("text", "")).strip())


def _starts_new_table_subrow(text: str) -> bool:
    normalized = text.lstrip()
    return bool(re.match(r"(?:[-ㆍ•·]|[oO]\s|<)", normalized))


def _normalize_header_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _merge_nested_table_children(
    children: list[dict[str, object]],
) -> list[dict[str, object]]:
    if len(children) < 2:
        return children

    merged: list[dict[str, object]] = []
    for child in children:
        if merged and _can_merge_nested_table_child(merged[-1], child):
            _append_nested_child_rows(merged[-1]["content"], child["content"])
            continue
        merged.append(child)
    return merged


def _can_merge_nested_table_child(
    previous: dict[str, object],
    current: dict[str, object],
) -> bool:
    if (
        previous.get("type", previous.get("kind")) != "table"
        or current.get("type", current.get("kind")) != "table"
    ):
        return False
    previous_content = previous.get("content")
    current_content = current.get("content")
    if not isinstance(previous_content, Mapping) or not isinstance(current_content, Mapping):
        return False
    previous_columns = previous_content.get("columns")
    current_columns = current_content.get("columns")
    if (
        _table_column_signature(previous_content)
        != _table_column_signature(current_content)
        and not _looks_like_nested_table_data_continuation(
            previous_content,
            current_content,
        )
    ):
        return False
    return (
        isinstance(previous_columns, list)
        and isinstance(current_columns, list)
        and len(previous_columns) == len(current_columns)
        and len(previous_columns) > 1
    )


def _looks_like_nested_table_data_continuation(
    previous: dict[str, object],
    current: dict[str, object],
) -> bool:
    previous_columns = previous.get("columns")
    current_columns = current.get("columns")
    if not isinstance(previous_columns, list) or not isinstance(current_columns, list):
        return False
    if len(previous_columns) != len(current_columns) or len(previous_columns) <= 1:
        return False
    labels = [
        str(column.get("text", "")).strip()
        for column in current_columns
        if isinstance(column, Mapping)
    ]
    if len(labels) != len(current_columns):
        return False
    data_like = sum(_looks_like_pdf_table_data_value(label) for label in labels)
    return data_like >= max(2, len(labels) // 2)


def _looks_like_pdf_table_data_value(text: str) -> bool:
    stripped = text.strip()
    if not stripped or _CJK.search(stripped):
        return False
    if len(stripped) > 40:
        return False
    return bool(re.search(r"\d", stripped))


def _append_nested_child_rows(
    target: dict[str, object],
    continuation: dict[str, object],
) -> None:
    target_columns = target.get("columns")
    target_rows = target.get("rows")
    continuation_columns = continuation.get("columns")
    continuation_rows = continuation.get("rows")
    if (
        not isinstance(target_columns, list)
        or not isinstance(target_rows, list)
        or not isinstance(continuation_columns, list)
        or not isinstance(continuation_rows, list)
    ):
        return

    if _table_column_signature(target) != _table_column_signature(continuation):
        target_rows.append(
            {
                "index": len(target_rows) + 1,
                "cells": [
                    _simple_cell(
                        str(target_columns[index].get("id", f"c{index + 1}")),
                        str(column.get("text", "")),
                    )
                    for index, column in enumerate(continuation_columns)
                    if isinstance(column, Mapping)
                ],
            }
        )

    for row in continuation_rows:
        if not isinstance(row, Mapping):
            continue
        copied = dict(row)
        copied["index"] = len(target_rows) + 1
        target_rows.append(copied)
    _promote_code_table_leaf_headers(target)
    _promote_ultrasound_code_matrix(target)
    _expand_parallel_code_action_rows(target)
