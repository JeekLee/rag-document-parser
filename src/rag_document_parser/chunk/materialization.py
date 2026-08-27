from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..models import Evidence, EvidenceItem, EvidenceUnit, RagChunk, SourceEvidence


@dataclass(frozen=True)
class ChunkPlanMaterializer:
    """Validates and materializes an LLM chunk plan from domain evidence units."""

    max_units_per_chunk: int

    def validate(self, units: list[EvidenceUnit]) -> None:
        _validate_unique_unit_ids(units)

    def materialize(self, units: list[EvidenceUnit], raw_plan: Any) -> list[RagChunk]:
        self.validate(units)
        return _materialize_window(units, raw_plan, self.max_units_per_chunk)

    def fallback(
        self,
        units: list[EvidenceUnit],
        reason: str,
        raw_plan: Any | None = None,
    ) -> list[RagChunk]:
        return _fallback_chunks(units, reason, raw_plan)


@dataclass(frozen=True)
class _OmittedTableRows:
    unit: EvidenceUnit
    row_indexes: list[int]


def _materialize_window(
    units: list[EvidenceUnit],
    raw_plan: Any,
    max_units_per_chunk: int | None = None,
) -> list[RagChunk]:
    if not isinstance(raw_plan, list):
        raise ValueError("chunk plan must be a list")

    by_id = {unit.id: unit for unit in units}
    full_assigned: set[str] = set()
    row_ranges_by_unit: dict[str, list[tuple[int, int]]] = {}
    chunks: list[RagChunk] = []

    for item in raw_plan:
        if not isinstance(item, Mapping):
            raise ValueError("chunk plan item must be an object")

        operations = item.get("operations")
        if not isinstance(operations, list) or not operations:
            raise ValueError("chunk plan item requires operations")
        operations, operation_warnings = _normalize_plan_operations(operations, by_id)
        operation_unit_ids = _operation_unit_ids(operations, by_id)
        plan_warnings = [
            *operation_warnings,
            *_plan_unit_id_warnings(item, by_id, operation_unit_ids),
        ]

        prior_covered = _covered_unit_ids(full_assigned, row_ranges_by_unit)
        next_full_assigned = set(full_assigned)
        next_row_ranges = {
            unit_id: list(ranges)
            for unit_id, ranges in row_ranges_by_unit.items()
        }
        chunk_units: list[EvidenceUnit] = []
        evidence_items: list[EvidenceItem] = []
        source_parts: list[str] = []
        normalized_ops: list[dict[str, Any]] = []

        for operation in operations:
            if not isinstance(operation, Mapping):
                raise ValueError("operation must be an object")

            unit_id = operation.get("unit_id")
            if not isinstance(unit_id, str) or unit_id not in by_id:
                raise ValueError(f"unknown unit id: {unit_id!r}")

            unit = by_id[unit_id]
            (
                evidence_item,
                source_text,
                normalized,
                repair_warnings,
            ) = _materialize_operation_with_repair(unit, operation)
            plan_warnings.extend(repair_warnings)
            _register_assignment(unit, normalized, next_full_assigned, next_row_ranges)
            chunk_units.append(unit)
            evidence_items.append(evidence_item)
            if source_text:
                source_parts.append(source_text)
            normalized_ops.append(normalized)

        context_unit_ids = _context_unit_ids(item.get("context_unit_ids"), by_id, prior_covered)
        chunks.append(
            _chunk_from_items(
                len(chunks) + 1,
                chunk_units,
                evidence_items,
                source_parts,
                item,
                normalized_ops,
                context_unit_ids,
                max_units_per_chunk,
                plan_warnings=plan_warnings,
            )
        )
        full_assigned = next_full_assigned
        row_ranges_by_unit = next_row_ranges

    chunks = _repair_omitted_table_rows(
        chunks,
        units,
        full_assigned,
        row_ranges_by_unit,
        raw_plan,
    )
    covered = _covered_unit_ids(full_assigned, row_ranges_by_unit)
    missing_units = [unit for unit in units if unit.id not in covered]
    if missing_units:
        reason = f"chunk plan omitted units: {', '.join(unit.id for unit in missing_units)}"
        for unit in missing_units:
            chunks = _insert_repair_chunk(
                chunks,
                _fallback_chunks([unit], reason, raw_plan)[0],
                _chunk_sort_key_for_unit(unit, units),
                units,
            )
    return chunks


def _validate_unique_unit_ids(units: list[EvidenceUnit]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for unit in units:
        if unit.id in seen and unit.id not in duplicates:
            duplicates.append(unit.id)
        seen.add(unit.id)

    if duplicates:
        raise ValueError(f"duplicate unit id: {', '.join(duplicates)}")


def _repair_omitted_table_rows(
    chunks: list[RagChunk],
    units: list[EvidenceUnit],
    full_assigned: set[str],
    row_ranges_by_unit: dict[str, list[tuple[int, int]]],
    raw_plan: Any,
) -> list[RagChunk]:
    repaired = list(chunks)
    for omitted in _omitted_table_rows(units, full_assigned, row_ranges_by_unit):
        row_ranges = contiguous_ranges(omitted.row_indexes)
        row_ranges_payload = [[start, end] for start, end in row_ranges]
        evidence_item, source_text, normalized = _materialize_operation(
            omitted.unit,
            {
                "unit_id": omitted.unit.id,
                "action": "include_rows",
                "row_ranges": row_ranges_payload,
            },
        )
        reason = (
            f"chunk plan omitted table rows for unit {omitted.unit.id}: "
            f"{', '.join(str(index) for index in omitted.row_indexes)}"
        )
        repair = _chunk_from_items(
            len(repaired) + 1,
            [omitted.unit],
            [evidence_item],
            [source_text],
            {
                "summary": fallback_summary_from_text(source_text),
                "keywords": fallback_keywords_from_text(source_text),
                "questions": fallback_questions(source_text[:80]),
            },
            [normalized],
            [],
        )
        metadata = dict(repair.metadata)
        metadata["_fallback_reason"] = reason
        metadata["_rejected_plan"] = _debug_value(raw_plan)
        metadata["_needs_enrichment"] = True
        repair = RagChunk(
            id=repair.id,
            source=repair.source,
            evidence=repair.evidence,
            summary=repair.summary,
            keywords=list(repair.keywords),
            questions=list(repair.questions),
            metadata=metadata,
        )
        repaired = _insert_repair_chunk(
            repaired,
            repair,
            _chunk_sort_key_for_table_rows(omitted.unit, omitted.row_indexes, units),
            units,
        )
    return repaired


def _omitted_table_rows(
    units: list[EvidenceUnit],
    full_assigned: set[str],
    row_ranges_by_unit: dict[str, list[tuple[int, int]]],
) -> list[_OmittedTableRows]:
    by_id = {unit.id: unit for unit in units}
    omitted: list[_OmittedTableRows] = []
    for unit_id, ranges in row_ranges_by_unit.items():
        if unit_id in full_assigned:
            continue

        unit = by_id[unit_id]
        if unit.format != "structured_table" or not isinstance(unit.content, Mapping):
            continue

        rows = unit.content.get("rows", [])
        if not isinstance(rows, list):
            continue

        missing = [
            index
            for index in _table_row_indexes(rows)
            if not _row_selected(index, ranges)
        ]
        if missing:
            omitted.append(_OmittedTableRows(unit, missing))
    return omitted


def _operation_unit_ids(
    operations: list[Any],
    by_id: dict[str, EvidenceUnit],
) -> list[str]:
    unit_ids: list[str] = []
    for operation in operations:
        if not isinstance(operation, Mapping):
            raise ValueError("operation must be an object")
        unit_id = operation.get("unit_id")
        if not isinstance(unit_id, str) or unit_id not in by_id:
            raise ValueError(f"unknown unit id: {unit_id!r}")
        unit_ids.append(unit_id)
    return unit_ids


def _normalize_plan_operations(
    operations: list[Any],
    by_id: dict[str, EvidenceUnit],
) -> tuple[list[Any], list[dict[str, Any]]]:
    full_include_unit_ids = {
        operation.get("unit_id")
        for operation in operations
        if (
            isinstance(operation, Mapping)
            and operation.get("action", "include") == "include"
            and isinstance(operation.get("unit_id"), str)
            and _is_structured_table_unit(by_id.get(operation.get("unit_id")))
        )
    }
    if not full_include_unit_ids:
        return operations, []

    normalized: list[Any] = []
    ignored_row_unit_ids: list[str] = []
    for operation in operations:
        if (
            isinstance(operation, Mapping)
            and operation.get("action") == "include_rows"
            and operation.get("unit_id") in full_include_unit_ids
        ):
            ignored_row_unit_ids.append(operation["unit_id"])
            continue
        normalized.append(operation)

    if not ignored_row_unit_ids:
        return operations, []
    return normalized, [
        {
            "type": "agentic_plan_include_rows_ignored",
            "reason": "same plan item also fully included the table; full include was used",
            "unit_ids": unique_strings(ignored_row_unit_ids),
        }
    ]


def _is_structured_table_unit(unit: EvidenceUnit | None) -> bool:
    return (
        unit is not None
        and unit.format == "structured_table"
        and isinstance(unit.content, Mapping)
    )


def _plan_unit_id_warnings(
    item: dict[str, Any],
    by_id: dict[str, EvidenceUnit],
    operation_unit_ids: list[str],
) -> list[dict[str, Any]]:
    if "unit_ids" not in item:
        return []
    unit_ids = item.get("unit_ids")
    if not isinstance(unit_ids, list):
        return [
            {
                "type": "agentic_plan_unit_ids_ignored",
                "reason": "unit_ids must be a list",
                "operation_unit_ids": list(operation_unit_ids),
            }
        ]

    invalid_unit_ids = [
        unit_id
        for unit_id in unit_ids
        if not isinstance(unit_id, str) or unit_id not in by_id
    ]
    if invalid_unit_ids:
        return [
            {
                "type": "agentic_plan_unit_ids_ignored",
                "reason": "unit_ids contains unknown or invalid ids",
                "unit_ids": _debug_value(unit_ids),
                "operation_unit_ids": list(operation_unit_ids),
            }
        ]

    if unit_ids != operation_unit_ids:
        return [
            {
                "type": "agentic_plan_unit_ids_mismatch",
                "reason": "unit_ids did not match operations; operations were used",
                "unit_ids": list(unit_ids),
                "operation_unit_ids": list(operation_unit_ids),
            }
        ]
    return []


def _materialize_operation_with_repair(
    unit: EvidenceUnit,
    operation: dict[str, Any],
) -> tuple[EvidenceItem, str, dict[str, Any], list[dict[str, Any]]]:
    try:
        evidence_item, source_text, normalized = _materialize_operation(unit, operation)
    except ValueError as exc:
        if operation.get("action") != "include_rows":
            raise
        evidence_item, source_text, normalized = _materialize_operation(
            unit,
            {"unit_id": unit.id, "action": "include"},
        )
        return (
            evidence_item,
            source_text,
            normalized,
            [
                {
                    "type": "agentic_include_rows_repaired_to_full_include",
                    "unit_id": unit.id,
                    "reason": str(exc),
                    "operation": _debug_value(operation),
                }
            ],
        )
    return evidence_item, source_text, normalized, []


def _materialize_operation(
    unit: EvidenceUnit,
    operation: dict[str, Any],
) -> tuple[EvidenceItem, str, dict[str, Any]]:
    action = operation.get("action", "include")
    if action == "include":
        return (
            EvidenceItem(
                type=unit.type,
                format=unit.format,
                content=unit.content,
                source_unit_ids=[unit.id],
                metadata=dict(unit.metadata),
            ),
            unit.source.text,
            {"unit_id": unit.id, "action": "include"},
        )

    if action == "include_rows":
        ranges = _normalize_row_ranges(operation.get("row_ranges"))
        if unit.format != "structured_table" or not isinstance(unit.content, Mapping):
            raise ValueError(f"include_rows requires structured_table unit: {unit.id}")

        subset = table_subset(unit.content, ranges)
        row_ranges = [[start, end] for start, end in ranges]
        return (
            EvidenceItem(
                type=unit.type,
                format=unit.format,
                content=subset,
                source_unit_ids=[unit.id],
                metadata={**dict(unit.metadata), "row_ranges": row_ranges},
            ),
            table_source_text(subset),
            {"unit_id": unit.id, "action": "include_rows", "row_ranges": row_ranges},
        )

    raise ValueError(f"unsupported action: {action!r}")


def _normalize_row_ranges(value: Any) -> list[tuple[int, int]]:
    if not isinstance(value, list):
        raise ValueError("include_rows requires row_ranges")

    ranges: list[tuple[int, int]] = []
    for item in value:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or type(item[0]) is not int
            or type(item[1]) is not int
            or item[0] > item[1]
        ):
            raise ValueError("row range must be [start, end] ints with start <= end")
        ranges.append((item[0], item[1]))
    return ranges


def table_subset(table: dict[str, Any], ranges: list[tuple[int, int]]) -> dict[str, Any]:
    rows = table.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError("structured_table content requires rows")

    _validate_row_range_bounds(ranges, _table_row_indexes(rows))

    selected_indexes = {
        index
        for row in rows
        if (
            isinstance(row, Mapping)
            and type(row.get("index")) is int
            and _row_selected(row["index"], ranges)
        )
        for index in [row["index"]]
    }
    carried_by_row = _rowspan_context_cells_by_row(table, rows, selected_indexes)
    selected: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        index = row.get("index")
        if type(index) is int and _row_selected(index, ranges):
            selected.append(_row_with_carried_cells(row, carried_by_row.get(index, []), table))

    if not selected:
        raise ValueError("row_ranges selected no rows")

    subset = dict(table)
    subset["rows"] = selected
    return subset


def _rowspan_context_cells_by_row(
    table: dict[str, Any],
    rows: list[Any],
    selected_indexes: set[int],
) -> dict[int, list[dict[str, Any]]]:
    if not selected_indexes:
        return {}

    result: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        start_index = row.get("index")
        if type(start_index) is not int:
            continue
        cells = row.get("cells", [])
        if not isinstance(cells, list):
            continue
        for cell in cells:
            if not isinstance(cell, Mapping):
                continue
            rowspan = _positive_int(cell.get("rowspan"))
            if rowspan <= 1:
                continue
            covered = [
                index
                for index in sorted(selected_indexes)
                if start_index < index < start_index + rowspan
            ]
            for group in contiguous_ranges(covered):
                carried = dict(cell)
                carried["rowspan"] = group[1] - group[0] + 1
                metadata = (
                    dict(carried.get("metadata"))
                    if isinstance(carried.get("metadata"), Mapping)
                    else {}
                )
                metadata["rowspan_context"] = {
                    "source_row_index": start_index,
                    "source_rowspan": rowspan,
                }
                carried["metadata"] = metadata
                result.setdefault(group[0], []).append(carried)
    return result


def _row_with_carried_cells(
    row: dict[str, Any],
    carried_cells: list[dict[str, Any]],
    table: dict[str, Any],
) -> dict[str, Any]:
    if not carried_cells:
        return row

    copied = dict(row)
    existing_cells = [
        dict(cell)
        for cell in row.get("cells", [])
        if isinstance(cell, Mapping)
    ]
    columns = table.get("columns", [])
    if not isinstance(columns, list):
        columns = []
    cells = list(existing_cells)
    for carried in carried_cells:
        if _cell_overlaps_any(carried, cells, columns):
            continue
        cells.append(carried)
    copied["cells"] = sorted(cells, key=lambda cell: _cell_sort_key(cell, columns))
    return copied


def _cell_overlaps_any(
    cell: dict[str, Any],
    cells: list[dict[str, Any]],
    columns: list[Any],
) -> bool:
    span = _cell_column_span(cell, columns)
    if span is None:
        return False
    for other in cells:
        other_span = _cell_column_span(other, columns)
        if other_span is None:
            continue
        if span[0] < other_span[1] and other_span[0] < span[1]:
            return True
    return False


def _cell_sort_key(cell: dict[str, Any], columns: list[Any]) -> tuple[int, str]:
    span = _cell_column_span(cell, columns)
    if span is None:
        return (10_000, str(cell.get("column_id", "")))
    return (span[0], str(cell.get("column_id", "")))


def _cell_column_span(cell: dict[str, Any], columns: list[Any]) -> tuple[int, int] | None:
    column_id = cell.get("column_id")
    start = _column_index(columns, column_id)
    if start is None:
        return None
    colspan = _positive_int(cell.get("colspan"))
    return (start, start + colspan)


def _column_index(columns: list[Any], column_id: Any) -> int | None:
    for index, column in enumerate(columns):
        if isinstance(column, Mapping) and column.get("id") == column_id:
            return index
    if isinstance(column_id, str):
        match = re.fullmatch(r"c([1-9][0-9]*)", column_id)
        if match:
            index = int(match.group(1)) - 1
            if index >= 0:
                return index
    return None


def _positive_int(value: Any) -> int:
    return value if type(value) is int and value > 0 else 1


def _table_row_indexes(rows: list[Any]) -> list[int]:
    indexes: list[int] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        index = row.get("index")
        if type(index) is int:
            indexes.append(index)
    return indexes


def _validate_row_range_bounds(
    ranges: list[tuple[int, int]],
    row_indexes: list[int],
) -> None:
    if not row_indexes:
        return

    min_index = min(row_indexes)
    max_index = max(row_indexes)
    for start, end in ranges:
        if start < min_index or end > max_index:
            raise ValueError("row range is outside table rows")


def _row_selected(index: int, ranges: list[tuple[int, int]]) -> bool:
    for start, end in ranges:
        if start <= index <= end:
            return True
    return False


def _register_assignment(
    unit: EvidenceUnit,
    operation: dict[str, Any],
    full_assigned: set[str],
    row_ranges_by_unit: dict[str, list[tuple[int, int]]],
) -> None:
    action = operation["action"]
    if action == "include":
        if unit.id in full_assigned:
            raise ValueError(f"duplicate unit id: {unit.id}")
        if unit.id in row_ranges_by_unit:
            raise ValueError(f"full include conflicts with include_rows for unit: {unit.id}")
        full_assigned.add(unit.id)
        return

    if action == "include_rows":
        if unit.id in full_assigned:
            raise ValueError(f"full include conflicts with include_rows for unit: {unit.id}")

        existing = list(row_ranges_by_unit.get(unit.id, []))
        ranges = [(item[0], item[1]) for item in operation["row_ranges"]]
        for candidate in ranges:
            if any(_ranges_overlap(candidate, current) for current in existing):
                raise ValueError(f"include_rows ranges overlap for unit: {unit.id}")
            existing.append(candidate)
        row_ranges_by_unit[unit.id] = existing
        return

    raise ValueError(f"unsupported action: {action!r}")


def _ranges_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] <= right[1] and right[0] <= left[1]


def _covered_unit_ids(
    full_assigned: set[str],
    row_ranges_by_unit: dict[str, list[tuple[int, int]]],
) -> set[str]:
    return set(full_assigned) | set(row_ranges_by_unit)


def table_source_text(table: dict[str, Any]) -> str:
    columns = table.get("columns", [])
    if not isinstance(columns, list):
        columns = []

    lines = [f"table: {len(columns)} columns"]
    labels = [
        str(column.get("text", "")).strip()
        for column in columns
        if isinstance(column, Mapping)
    ]
    if labels:
        lines.append("columns: " + " | ".join(labels))

    rows = table.get("rows", [])
    if not isinstance(rows, list):
        return "\n".join(lines)

    for row in rows:
        if not isinstance(row, Mapping):
            continue
        values: list[str] = []
        cells = row.get("cells", [])
        if isinstance(cells, list):
            for cell in cells:
                if not isinstance(cell, Mapping):
                    continue
                label = column_label(columns, cell.get("column_id"))
                text = str(cell.get("text", "")).strip()
                if text:
                    values.append(f"{label}={text}")
        lines.append(f"row {row.get('index', '?')}: " + "; ".join(values))

    return "\n".join(lines)


def column_label(columns: list[Any], column_id: Any) -> str:
    for column in columns:
        if (
            isinstance(column, Mapping)
            and column.get("id") == column_id
            and isinstance(column.get("text"), str)
            and column["text"].strip()
        ):
            return column["text"].strip()

    if isinstance(column_id, str):
        match = re.fullmatch(r"c([1-9][0-9]*)", column_id)
        if match:
            index = int(match.group(1)) - 1
            if 0 <= index < len(columns):
                column = columns[index]
                if isinstance(column, Mapping) and isinstance(column.get("text"), str):
                    text = column["text"].strip()
                    if text:
                        return text
        return column_id

    return "col"


def _chunk_from_items(
    index: int,
    units: list[EvidenceUnit],
    evidence_items: list[EvidenceItem],
    source_parts: list[str],
    plan: dict[str, Any],
    operations: list[dict[str, Any]],
    context_unit_ids: list[str],
    max_units_per_chunk: int | None = None,
    *,
    plan_warnings: list[dict[str, Any]] | None = None,
) -> RagChunk:
    source_text = "\n\n".join(source_parts)
    source_unit_ids = unique_strings([unit.id for unit in units])
    title = plan.get("title") if isinstance(plan.get("title"), str) else ""
    summary = plan.get("summary") if isinstance(plan.get("summary"), str) else ""
    if not summary:
        summary = fallback_summary_from_text(source_text)

    keywords = string_items(plan.get("keywords")) or fallback_keywords_from_text(source_text)
    question_topic = source_text.strip().replace("\n", " ")[:160] or title or summary
    questions = string_items(plan.get("questions")) or fallback_questions(question_topic)
    metadata = {
        "source_unit_ids": source_unit_ids,
        "source_units": source_unit_records(units),
        "context_unit_ids": context_unit_ids,
        "operations": operations,
        "title": title,
        "common": {
            "unit_types": unique_strings([unit.type for unit in units]),
            "display_format": "composite",
        },
    }
    warnings = list(plan_warnings or [])
    if max_units_per_chunk is not None and len(source_unit_ids) > max_units_per_chunk:
        warnings.append(
            {
                "type": "agentic_chunk_exceeds_max_units",
                "source_unit_count": len(source_unit_ids),
                "max_units_per_chunk": max_units_per_chunk,
            }
        )
    if warnings:
        metadata["_warnings"] = warnings

    return RagChunk(
        id=f"chunk-{index}",
        source=SourceEvidence(kind="chunk", text=source_text),
        evidence=Evidence(items=evidence_items),
        summary=summary,
        keywords=keywords,
        questions=questions,
        metadata=metadata,
    )


def string_items(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def dict_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _context_unit_ids(
    value: Any,
    by_id: dict[str, EvidenceUnit],
    prior_assigned: set[str],
) -> list[str]:
    result: list[str] = []
    for unit_id in string_items(value):
        if unit_id not in by_id:
            raise ValueError(f"unknown context unit id: {unit_id!r}")
        if unit_id not in prior_assigned:
            raise ValueError(f"context unit id must refer to a prior assigned unit: {unit_id}")
        if unit_id not in result:
            result.append(unit_id)
    return result


def _insert_repair_chunk(
    chunks: list[RagChunk],
    repair: RagChunk,
    repair_key: tuple[int, int],
    units: list[EvidenceUnit],
) -> list[RagChunk]:
    result = list(chunks)
    for index, chunk in enumerate(result):
        if _chunk_sort_key(chunk, units) > repair_key:
            result.insert(index, repair)
            return result
    result.append(repair)
    return result


def _chunk_sort_key(chunk: RagChunk, units: list[EvidenceUnit]) -> tuple[int, int]:
    order = _unit_order(units)
    operations = dict_items(chunk.metadata.get("operations"))
    keys: list[tuple[int, int]] = []
    for operation in operations:
        unit_id = operation.get("unit_id")
        if not isinstance(unit_id, str) or unit_id not in order:
            continue
        row_key = 0
        if operation.get("action") == "include_rows":
            row_key = _first_row_range_start(operation.get("row_ranges"))
        keys.append((order[unit_id], row_key))
    if keys:
        return min(keys)

    source_unit_ids = string_items(chunk.metadata.get("source_unit_ids"))
    indexes = [order[unit_id] for unit_id in source_unit_ids if unit_id in order]
    if indexes:
        return (min(indexes), 0)
    return (len(units), 0)


def _first_row_range_start(row_ranges: Any) -> int:
    if not isinstance(row_ranges, list):
        return 0
    starts = [
        row_range[0]
        for row_range in row_ranges
        if (
            isinstance(row_range, list)
            and len(row_range) == 2
            and type(row_range[0]) is int
        )
    ]
    return min(starts) if starts else 0


def _chunk_sort_key_for_unit(
    unit: EvidenceUnit,
    units: list[EvidenceUnit],
) -> tuple[int, int]:
    return (_unit_order(units).get(unit.id, len(units)), 0)


def _chunk_sort_key_for_table_rows(
    unit: EvidenceUnit,
    row_indexes: list[int],
    units: list[EvidenceUnit],
) -> tuple[int, int]:
    row_key = min(row_indexes) if row_indexes else 0
    return (_unit_order(units).get(unit.id, len(units)), row_key)


def _unit_order(units: list[EvidenceUnit]) -> dict[str, int]:
    return {unit.id: index for index, unit in enumerate(units)}


def contiguous_ranges(indexes: list[int]) -> list[tuple[int, int]]:
    if not indexes:
        return []
    sorted_indexes = sorted(indexes)
    ranges: list[tuple[int, int]] = []
    start = previous = sorted_indexes[0]
    for index in sorted_indexes[1:]:
        if index == previous + 1:
            previous = index
            continue
        ranges.append((start, previous))
        start = previous = index
    ranges.append((start, previous))
    return ranges


def unique_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def source_unit_records(units: list[EvidenceUnit]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for unit in units:
        if unit.id in seen:
            continue
        seen.add(unit.id)
        records.append(
            {
                "id": unit.id,
                "type": unit.type,
                "format": unit.format,
                "metadata": dict(unit.metadata),
            }
        )
    return records


def _fallback_summary(units: list[EvidenceUnit]) -> str:
    parts = [
        unit.source.text.strip().replace("\n", " ")[:160]
        for unit in units
        if unit.source.text.strip()
    ]
    return " / ".join(parts)[:500]


def fallback_summary_from_text(text: str) -> str:
    return text.strip().replace("\n", " ")[:500]


def _fallback_keywords(units: list[EvidenceUnit]) -> list[str]:
    return fallback_keywords_from_text("\n\n".join(unit.source.text for unit in units))


def fallback_keywords_from_text(text: str) -> list[str]:
    words: list[str] = []
    for token in re.findall(r"[0-9A-Za-z가-힣]{2,}", text):
        if token not in words:
            words.append(token)
        if len(words) >= 8:
            return words
    return words


def fallback_questions(topic: str) -> list[str]:
    base = topic.strip() or "이 청크"
    return [f"{base}에 대해 무엇을 알 수 있나요?"]


def _debug_value(value: Any, limit: int = 8000) -> Any:
    try:
        text = json.dumps(value, ensure_ascii=False)
    except TypeError:
        text = repr(value)
        if len(text) > limit:
            return {"truncated": True, "preview": text[:limit]}
        return text

    if len(text) > limit:
        return {"truncated": True, "preview": text[:limit]}
    return json.loads(text)


def _fallback_chunks(
    units: list[EvidenceUnit],
    reason: str,
    raw_plan: Any | None = None,
) -> list[RagChunk]:
    chunks: list[RagChunk] = []
    rejected_plan = _debug_value(raw_plan) if raw_plan is not None else None
    for index, unit in enumerate(units, start=1):
        item = EvidenceItem(
            type=unit.type,
            format=unit.format,
            content=unit.content,
            source_unit_ids=[unit.id],
            metadata=dict(unit.metadata),
        )
        common = {}
        if isinstance(unit.metadata.get("common"), Mapping):
            common.update(unit.metadata["common"])
        common["unit_types"] = [unit.type]
        common["display_format"] = "composite"
        metadata = {
            **dict(unit.metadata),
            "source_unit_ids": [unit.id],
            "source_units": source_unit_records([unit]),
            "context_unit_ids": [],
            "_fallback_reason": reason,
            "_needs_enrichment": True,
            "common": common,
        }
        if rejected_plan is not None:
            metadata["_rejected_plan"] = rejected_plan
        chunks.append(
            RagChunk(
                id=f"chunk-{index}",
                source=SourceEvidence(kind="chunk", text=unit.source.text),
                evidence=Evidence(items=[item]),
                summary=_fallback_summary([unit]),
                keywords=_fallback_keywords([unit]),
                questions=fallback_questions(unit.source.text[:80]),
                metadata=metadata,
            )
        )
    return chunks
