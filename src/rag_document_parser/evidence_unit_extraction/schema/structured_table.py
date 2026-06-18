from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ...models import (
    EvidenceChild,
    StructuredTableContent,
    TableCell,
    TableColumn,
    TableRow,
)


def table_column(column_id: str, text: str) -> TableColumn:
    return TableColumn(id=column_id, text=text)


def table_cell(
    column_id: str,
    text: str = "",
    *,
    rowspan: int = 1,
    colspan: int = 1,
    children: Iterable[EvidenceChild | Mapping[str, Any]] | None = None,
) -> TableCell:
    return TableCell(
        column_id=column_id,
        text=text,
        rowspan=rowspan,
        colspan=colspan,
        children=list(children or []),
    )


def table_row(index: int, cells: Iterable[TableCell | Mapping[str, Any]]) -> TableRow:
    return TableRow(index=index, cells=list(cells))


def structured_table(
    *,
    columns: Iterable[TableColumn | Mapping[str, Any]],
    rows: Iterable[TableRow | Mapping[str, Any]],
    caption: str | None = None,
    header_rows: Iterable[TableRow | Mapping[str, Any]] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> StructuredTableContent:
    payload: dict[str, Any] = {
        "caption": caption,
        "columns": list(columns),
        "rows": list(rows),
    }
    if header_rows is not None:
        payload["header_rows"] = list(header_rows)
    if extra:
        payload.update(extra)
    return StructuredTableContent(**payload)
