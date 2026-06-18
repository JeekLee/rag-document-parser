from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, TypedDict


EvidenceChild = dict[str, Any]


class TableColumn(TypedDict):
    id: str
    text: str


class TableCell(TypedDict):
    column_id: str
    text: str
    rowspan: int
    colspan: int
    children: list[EvidenceChild]


class TableRow(TypedDict):
    index: int
    cells: list[TableCell]


class StructuredTableContentBase(TypedDict):
    caption: str | None
    columns: list[TableColumn]
    rows: list[TableRow]


class StructuredTableContent(StructuredTableContentBase, total=False):
    header_rows: list[TableRow]
    compact: dict[str, Any]


def table_column(column_id: str, text: str) -> TableColumn:
    return {"id": column_id, "text": text}


def table_cell(
    column_id: str,
    text: str = "",
    *,
    rowspan: int = 1,
    colspan: int = 1,
    children: Iterable[EvidenceChild] | None = None,
) -> TableCell:
    return {
        "column_id": column_id,
        "text": text,
        "rowspan": rowspan,
        "colspan": colspan,
        "children": list(children or []),
    }


def table_row(index: int, cells: Iterable[TableCell]) -> TableRow:
    return {"index": index, "cells": list(cells)}


def structured_table(
    *,
    columns: Iterable[TableColumn],
    rows: Iterable[TableRow],
    caption: str | None = None,
    header_rows: Iterable[TableRow] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> StructuredTableContent:
    payload: StructuredTableContent = {
        "caption": caption,
        "columns": list(columns),
        "rows": list(rows),
    }
    if header_rows is not None:
        payload["header_rows"] = list(header_rows)
    if extra:
        payload.update(extra)
    return payload
