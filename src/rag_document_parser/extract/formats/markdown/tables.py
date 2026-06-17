from __future__ import annotations


def table_parts(lines: list[str]) -> tuple[list[str], list[list[str]]]:
    rows = [split_table_row(line) for line in lines]
    if len(rows) < 3:
        return [], []
    return rows[0], rows[2:]


def structured_table(headers: list[str], rows: list[list[str]]) -> dict[str, object]:
    columns = [
        {
            "id": f"c{index}",
            "text": header,
        }
        for index, header in enumerate(headers, start=1)
    ]
    structured_rows: list[dict[str, object]] = []
    for row_index, row in enumerate(rows, start=1):
        cells: list[dict[str, object]] = []
        for column, value in zip(columns, row, strict=False):
            cells.append(
                {
                    "column_id": column["id"],
                    "text": value,
                    "rowspan": 1,
                    "colspan": 1,
                    "children": [],
                }
            )
        structured_rows.append(
            {
                "index": row_index,
                "cells": cells,
            }
        )
    return {
        "caption": None,
        "columns": columns,
        "rows": structured_rows,
    }


def table_source_text(
    headers: list[str],
    rows: list[list[str]],
    section_path: list[str],
) -> str:
    lines: list[str] = []
    if section_path:
        lines.append(f"section: {' > '.join(section_path)}")
    if headers:
        lines.append(f"columns: {' | '.join(headers)}")
    for index, row in enumerate(rows, start=1):
        cells = [
            f"{header}={value}"
            for header, value in zip(headers, row, strict=False)
            if value
        ]
        if cells:
            lines.append(f"row {index}: {'; '.join(cells)}")
    return "\n".join(lines)


def split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]
