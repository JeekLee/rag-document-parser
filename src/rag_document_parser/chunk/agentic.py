from __future__ import annotations

import json
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from ..enrichment.llm import LlmConfig, chat_json
from ..models import Evidence, EvidenceItem, EvidenceUnit, RagChunk, SourceEvidence

PlanFn = Callable[[list[EvidenceUnit], LlmConfig | None, int], Any]


@dataclass(frozen=True)
class _WindowResult:
    chunks: list[RagChunk]
    fallback_reason: str | None = None


class EvidenceUnitAgenticChunker:
    def __init__(
        self,
        *,
        llm: LlmConfig | None,
        max_units_per_chunk: int = 10,
        window_size: int = 40,
        max_concurrency: int = 4,
        plan_fn: PlanFn | None = None,
    ) -> None:
        self._llm = llm
        self._max_units = max(1, max_units_per_chunk)
        self._window_size = max(1, window_size)
        self._concurrency = max(1, max_concurrency)
        self._plan_fn = plan_fn or self._default_plan

    def chunk(self, units: list[EvidenceUnit]) -> list[RagChunk]:
        if not units:
            return []

        windows = _windows(units, self._window_size)
        with ThreadPoolExecutor(max_workers=self._concurrency) as executor:
            results = list(executor.map(self._chunk_window, windows))

        chunks: list[RagChunk] = []
        for result in results:
            chunks.extend(result.chunks)

        return [_reindex_chunk(index, chunk) for index, chunk in enumerate(chunks, start=1)]

    def _chunk_window(self, window: list[EvidenceUnit]) -> _WindowResult:
        try:
            raw_plan = self._plan_fn(window, self._llm, self._max_units)
            chunks = _materialize_window(window, raw_plan)
        except ValueError as exc:
            return _WindowResult(_fallback_chunks(window, str(exc)), str(exc))
        return _WindowResult(chunks)

    def _default_plan(
        self,
        window: list[EvidenceUnit],
        cfg: LlmConfig | None,
        max_units: int,
    ) -> Any:
        if cfg is None:
            return None
        return chat_json(_plan_prompt(window, max_units), cfg)


def _windows(units: list[EvidenceUnit], size: int) -> list[list[EvidenceUnit]]:
    return [units[index : index + size] for index in range(0, len(units), size)]


def _materialize_window(units: list[EvidenceUnit], raw_plan: Any) -> list[RagChunk]:
    if not isinstance(raw_plan, list):
        raise ValueError("chunk plan must be a list")

    by_id = {unit.id: unit for unit in units}
    full_assigned: set[str] = set()
    row_ranges_by_unit: dict[str, list[tuple[int, int]]] = {}
    chunks: list[RagChunk] = []

    for item in raw_plan:
        if not isinstance(item, dict):
            raise ValueError("chunk plan item must be an object")

        operations = item.get("operations")
        if not isinstance(operations, list) or not operations:
            raise ValueError("chunk plan item requires operations")
        operation_unit_ids = _operation_unit_ids(operations, by_id)
        _validate_plan_unit_ids(item, by_id, operation_unit_ids)

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
            if not isinstance(operation, dict):
                raise ValueError("operation must be an object")

            unit_id = operation.get("unit_id")
            if not isinstance(unit_id, str) or unit_id not in by_id:
                raise ValueError(f"unknown unit id: {unit_id!r}")

            unit = by_id[unit_id]
            evidence_item, source_text, normalized = _materialize_operation(unit, operation)
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
            )
        )
        full_assigned = next_full_assigned
        row_ranges_by_unit = next_row_ranges

    covered = _covered_unit_ids(full_assigned, row_ranges_by_unit)
    missing = [unit.id for unit in units if unit.id not in covered]
    if missing:
        raise ValueError(f"chunk plan omitted units: {', '.join(missing)}")
    return chunks


def _operation_unit_ids(
    operations: list[Any],
    by_id: dict[str, EvidenceUnit],
) -> list[str]:
    unit_ids: list[str] = []
    for operation in operations:
        if not isinstance(operation, dict):
            raise ValueError("operation must be an object")
        unit_id = operation.get("unit_id")
        if not isinstance(unit_id, str) or unit_id not in by_id:
            raise ValueError(f"unknown unit id: {unit_id!r}")
        unit_ids.append(unit_id)
    return unit_ids


def _validate_plan_unit_ids(
    item: dict[str, Any],
    by_id: dict[str, EvidenceUnit],
    operation_unit_ids: list[str],
) -> None:
    if "unit_ids" not in item:
        return
    unit_ids = item.get("unit_ids")
    if not isinstance(unit_ids, list):
        raise ValueError("unit_ids must be a list")
    for unit_id in unit_ids:
        if not isinstance(unit_id, str) or unit_id not in by_id:
            raise ValueError(f"unknown unit id: {unit_id!r}")
    if unit_ids != operation_unit_ids:
        raise ValueError("unit_ids must match operation unit_ids")


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
        if unit.format != "structured_table" or not isinstance(unit.content, dict):
            raise ValueError(f"include_rows requires structured_table unit: {unit.id}")

        subset = _table_subset(unit.content, ranges)
        row_ranges = [[start, end] for start, end in ranges]
        return (
            EvidenceItem(
                type=unit.type,
                format=unit.format,
                content=subset,
                source_unit_ids=[unit.id],
                metadata={**dict(unit.metadata), "row_ranges": row_ranges},
            ),
            _table_source_text(subset),
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


def _table_subset(table: dict[str, Any], ranges: list[tuple[int, int]]) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    rows = table.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError("structured_table content requires rows")

    _validate_row_range_bounds(ranges, _table_row_indexes(rows))

    for row in rows:
        if not isinstance(row, dict):
            continue
        index = row.get("index")
        if type(index) is int and _row_selected(index, ranges):
            selected.append(row)

    if not selected:
        raise ValueError("row_ranges selected no rows")

    subset = dict(table)
    subset["rows"] = selected
    return subset


def _table_row_indexes(rows: list[Any]) -> list[int]:
    indexes: list[int] = []
    for row in rows:
        if not isinstance(row, dict):
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


def _table_source_text(table: dict[str, Any]) -> str:
    columns = table.get("columns", [])
    if not isinstance(columns, list):
        columns = []

    lines = [f"table: {len(columns)} columns"]
    labels = [
        str(column.get("text", "")).strip()
        for column in columns
        if isinstance(column, dict)
    ]
    if labels:
        lines.append("columns: " + " | ".join(labels))

    rows = table.get("rows", [])
    if not isinstance(rows, list):
        return "\n".join(lines)

    for row in rows:
        if not isinstance(row, dict):
            continue
        values: list[str] = []
        cells = row.get("cells", [])
        if isinstance(cells, list):
            for cell in cells:
                if not isinstance(cell, dict):
                    continue
                label = _column_label(columns, cell.get("column_id"))
                text = str(cell.get("text", "")).strip()
                if text:
                    values.append(f"{label}={text}")
        lines.append(f"row {row.get('index', '?')}: " + "; ".join(values))

    return "\n".join(lines)


def _column_label(columns: list[Any], column_id: Any) -> str:
    for column in columns:
        if (
            isinstance(column, dict)
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
                if isinstance(column, dict) and isinstance(column.get("text"), str):
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
) -> RagChunk:
    source_unit_ids = _unique([unit.id for unit in units])
    title = plan.get("title") if isinstance(plan.get("title"), str) else ""
    summary = plan.get("summary") if isinstance(plan.get("summary"), str) else ""
    if not summary:
        summary = _fallback_summary(units)

    keywords = _strings(plan.get("keywords")) or _fallback_keywords(units)
    questions = _strings(plan.get("questions")) or _fallback_questions(title or summary)
    chunk_type = _chunk_type(evidence_items)

    return RagChunk(
        id=f"chunk-{index}",
        type=chunk_type,
        source=SourceEvidence(kind=chunk_type, text="\n\n".join(source_parts)),
        evidence=Evidence(items=evidence_items),
        summary=summary,
        keywords=keywords,
        questions=questions,
        metadata={
            "source_unit_ids": source_unit_ids,
            "source_units": _source_units(units),
            "context_unit_ids": context_unit_ids,
            "operations": operations,
            "title": title,
            "common": {
                "unit_types": _unique([unit.type for unit in units]),
                "display_format": "composite",
            },
        },
    )


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _context_unit_ids(
    value: Any,
    by_id: dict[str, EvidenceUnit],
    prior_assigned: set[str],
) -> list[str]:
    result: list[str] = []
    for unit_id in _strings(value):
        if unit_id not in by_id:
            raise ValueError(f"unknown context unit id: {unit_id!r}")
        if unit_id not in prior_assigned:
            raise ValueError(f"context unit id must refer to a prior assigned unit: {unit_id}")
        if unit_id not in result:
            result.append(unit_id)
    return result


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _source_units(units: list[EvidenceUnit]) -> list[dict[str, Any]]:
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


def _chunk_type(items: list[EvidenceItem]) -> str:
    types = _unique([item.type for item in items])
    return types[0] if len(types) == 1 else "mixed"


def _fallback_summary(units: list[EvidenceUnit]) -> str:
    parts = [
        unit.source.text.strip().replace("\n", " ")[:160]
        for unit in units
        if unit.source.text.strip()
    ]
    return " / ".join(parts)[:500]


def _fallback_keywords(units: list[EvidenceUnit]) -> list[str]:
    words: list[str] = []
    for unit in units:
        for token in re.findall(r"[0-9A-Za-z가-힣]{2,}", unit.source.text):
            if token not in words:
                words.append(token)
            if len(words) >= 8:
                return words
    return words


def _fallback_questions(topic: str) -> list[str]:
    base = topic.strip() or "이 청크"
    return [f"{base}에 대해 무엇을 알 수 있나요?"]


def _fallback_chunks(units: list[EvidenceUnit], reason: str) -> list[RagChunk]:
    chunks: list[RagChunk] = []
    for index, unit in enumerate(units, start=1):
        item = EvidenceItem(
            type=unit.type,
            format=unit.format,
            content=unit.content,
            source_unit_ids=[unit.id],
            metadata=dict(unit.metadata),
        )
        common = {}
        if isinstance(unit.metadata.get("common"), dict):
            common.update(unit.metadata["common"])
        common["unit_types"] = [unit.type]
        common["display_format"] = "composite"
        metadata = {
            **dict(unit.metadata),
            "source_unit_ids": [unit.id],
            "source_units": _source_units([unit]),
            "context_unit_ids": [],
            "_fallback_reason": reason,
            "common": common,
        }
        chunks.append(
            RagChunk(
                id=f"chunk-{index}",
                type=unit.type,
                source=SourceEvidence(kind=unit.type, text=unit.source.text),
                evidence=Evidence(items=[item]),
                summary=_fallback_summary([unit]),
                keywords=_fallback_keywords([unit]),
                questions=_fallback_questions(unit.source.text[:80]),
                metadata=metadata,
            )
        )
    return chunks


def _reindex_chunk(index: int, chunk: RagChunk) -> RagChunk:
    return RagChunk(
        id=f"chunk-{index}",
        type=chunk.type,
        source=chunk.source,
        evidence=chunk.evidence,
        summary=chunk.summary,
        keywords=list(chunk.keywords),
        questions=list(chunk.questions),
        metadata=dict(chunk.metadata),
    )


def _plan_prompt(window: list[EvidenceUnit], max_units: int) -> str:
    payload = {
        "max_units_per_chunk": max_units,
        "units": [
            {
                "id": unit.id,
                "type": unit.type,
                "format": unit.format,
                "source_text": unit.source.text[:2000],
                "metadata": unit.metadata,
            }
            for unit in window
        ],
    }
    return (
        "Return a JSON array of chunk plan objects. Each object must include "
        "unit_ids, non-empty operations, optional context_unit_ids, title, summary, "
        "keywords, and questions. Use operations with action include or include_rows; "
        "for include_rows provide row_ranges as inclusive [start, end] pairs. "
        "Do not generate final evidence content; it will be copied from source units.\n\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )
