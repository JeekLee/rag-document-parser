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
BoundaryMergeFn = Callable[[RagChunk, RagChunk, LlmConfig | None, int], Any]


_PROMPT = """\
당신은 RAG 인덱싱용 EvidenceUnit chunk planner입니다.
아래 unit 목록을 의미적으로 일관된 chunk plan으로 묶어 주세요.

규칙:
- evidence content는 작성하지 않습니다. unit_id와 operation만 작성합니다.
- evidence content는 unit에서 복사됩니다.
- 모든 unit은 누락 없이 evidence로 포함되어야 합니다.
- structured_table은 include_rows로 여러 chunk에 분해할 수 있습니다.
- include_rows row range는 겹치지 않아야 하며 table row 범위 안에 있어야 합니다.
- include_rows의 row_ranges는 양 끝을 포함하는 inclusive [start, end] 쌍 목록입니다.
- include_rows를 사용하면 해당 table의 모든 실제 row index를 빠짐없이, 겹치지 않게 포함해야 합니다.
- table row coverage에 확신이 없으면 action "include"로 전체 table을 포함합니다.
- context_unit_ids는 선택 사항이며 이미 이전 chunk에서 evidence로 포함된 unit id만 작성합니다.
- text, table, image를 같은 chunk에 묶을 수 있습니다.
- 원문에 없는 사실을 summary, keywords, questions에 추가하지 않습니다.
- 한 chunk는 가능하면 unit {max_units}개 이하로 유지합니다.

Unit 목록:
{units}

JSON 배열만 출력하세요:
[
  {
    "unit_ids": [{example_unit_id}],
    "operations": [
      {"unit_id": {example_unit_id}, "action": "include"}
    ],
    "context_unit_ids": [],
    "title": "제목",
    "summary": "요약",
    "keywords": ["키워드"],
    "questions": ["이 chunk로 답할 수 있는 질문"]
  }
]
{include_rows_example}
"""


_BOUNDARY_PROMPT = """\
You are a RAG window boundary merge planner.
Decide whether two adjacent chunks from neighboring EvidenceUnit windows are one continuous semantic unit.

Rules:
- Return only a JSON object.
- Use action "merge" only when the right chunk directly continues the same topic, table, section, or Q&A block.
- Use action "keep" when the chunks are merely related but answer different retrieval questions.
- Do not invent evidence content. Evidence content will be copied from the existing chunks.
- Keep merged chunks reasonably close to {max_units} source units when possible.

Boundary payload:
{boundary}

JSON object only:
{{
  "action": "merge",
  "reason": "short reason",
  "title": "optional merged title",
  "summary": "optional merged summary",
  "keywords": ["optional", "keywords"],
  "questions": ["optional retrieval question"]
}}
"""


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
        boundary_merge_fn: BoundaryMergeFn | None = None,
    ) -> None:
        self._llm = llm
        self._max_units = max(1, max_units_per_chunk)
        self._window_size = max(1, window_size)
        self._concurrency = max(1, max_concurrency)
        self._plan_fn = plan_fn or self._default_plan
        self._boundary_merge_fn = boundary_merge_fn

    def chunk(self, units: list[EvidenceUnit]) -> list[RagChunk]:
        if not units:
            return []

        windows = _windows(units, self._window_size)
        with ThreadPoolExecutor(max_workers=self._concurrency) as executor:
            results = list(executor.map(self._chunk_window, windows))

        chunks = self._merge_window_boundaries(results)

        return [_reindex_chunk(index, chunk) for index, chunk in enumerate(chunks, start=1)]

    def _chunk_window(self, window: list[EvidenceUnit]) -> _WindowResult:
        raw_plan: Any | None = None
        try:
            _validate_unique_unit_ids(window)
            raw_plan = self._plan_fn(window, self._llm, self._max_units)
            chunks = _materialize_window(window, raw_plan, self._max_units)
        except Exception as exc:
            reason = _fallback_reason(exc)
            return _WindowResult(_fallback_chunks(window, reason, raw_plan), reason)
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

    def _merge_window_boundaries(self, results: list[_WindowResult]) -> list[RagChunk]:
        if not results:
            return []
        if len(results) == 1 or (self._boundary_merge_fn is None and self._llm is None):
            return [chunk for result in results for chunk in result.chunks]

        chunks = list(results[0].chunks)
        previous_result = results[0]
        for result in results[1:]:
            next_chunks = list(result.chunks)
            if (
                chunks
                and next_chunks
                and previous_result.fallback_reason is None
                and result.fallback_reason is None
            ):
                try:
                    decision = self._plan_boundary_merge(chunks[-1], next_chunks[0])
                except Exception as exc:
                    chunks[-1] = _chunk_with_boundary_merge_warning(
                        chunks[-1],
                        next_chunks[0],
                        _fallback_reason(exc),
                    )
                    decision = {"action": "keep"}
                if _boundary_action(decision) == "merge":
                    chunks[-1] = _merge_adjacent_chunks(
                        chunks[-1],
                        next_chunks.pop(0),
                        decision,
                        self._max_units,
                    )
            chunks.extend(next_chunks)
            previous_result = result
        return chunks

    def _plan_boundary_merge(self, left: RagChunk, right: RagChunk) -> Any:
        if self._boundary_merge_fn is not None:
            return self._boundary_merge_fn(left, right, self._llm, self._max_units)
        return self._default_boundary_merge(left, right, self._llm, self._max_units)

    def _default_boundary_merge(
        self,
        left: RagChunk,
        right: RagChunk,
        cfg: LlmConfig | None,
        max_units: int,
    ) -> Any:
        if cfg is None:
            return {"action": "keep", "reason": "llm is not configured"}
        return chat_json(_boundary_prompt(left, right, max_units), cfg)


def _windows(units: list[EvidenceUnit], size: int) -> list[list[EvidenceUnit]]:
    return [units[index : index + size] for index in range(0, len(units), size)]


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
                max_units_per_chunk,
            )
        )
        full_assigned = next_full_assigned
        row_ranges_by_unit = next_row_ranges

    _validate_row_coverage(units, full_assigned, row_ranges_by_unit)
    covered = _covered_unit_ids(full_assigned, row_ranges_by_unit)
    missing = [unit.id for unit in units if unit.id not in covered]
    if missing:
        raise ValueError(f"chunk plan omitted units: {', '.join(missing)}")
    return chunks


def _fallback_reason(exc: Exception) -> str:
    reason = str(exc)
    if reason:
        return reason
    return exc.__class__.__name__


def _boundary_action(decision: Any) -> str:
    if not isinstance(decision, dict):
        return "keep"
    action = decision.get("action")
    return action if action in {"merge", "keep"} else "keep"


def _merge_adjacent_chunks(
    left: RagChunk,
    right: RagChunk,
    decision: Any,
    max_units_per_chunk: int,
) -> RagChunk:
    decision = decision if isinstance(decision, dict) else {}
    evidence_items = [*left.evidence.items, *right.evidence.items]
    source_text = "\n\n".join(
        text for text in [left.source.text, right.source.text] if text
    )
    source_unit_ids = _unique(
        [
            *_strings(left.metadata.get("source_unit_ids")),
            *_strings(right.metadata.get("source_unit_ids")),
        ]
    )
    context_unit_ids = [
        unit_id
        for unit_id in _unique(
            [
                *_strings(left.metadata.get("context_unit_ids")),
                *_strings(right.metadata.get("context_unit_ids")),
            ]
        )
        if unit_id not in source_unit_ids
    ]
    operations = [
        *_dicts(left.metadata.get("operations")),
        *_dicts(right.metadata.get("operations")),
    ]
    title = decision.get("title") if isinstance(decision.get("title"), str) else ""
    summary = decision.get("summary") if isinstance(decision.get("summary"), str) else ""
    if not summary:
        summary = _fallback_summary_from_text(source_text)

    keywords = _strings(decision.get("keywords")) or _unique(
        [*left.keywords, *right.keywords]
    )
    question_topic = source_text.strip().replace("\n", " ")[:160] or title or summary
    questions = _strings(decision.get("questions")) or _unique(
        [*left.questions, *right.questions]
    ) or _fallback_questions(question_topic)
    metadata = {
        "source_unit_ids": source_unit_ids,
        "source_units": _merge_source_units(
            left.metadata.get("source_units"),
            right.metadata.get("source_units"),
        ),
        "context_unit_ids": context_unit_ids,
        "operations": operations,
        "title": title,
        "common": {
            "unit_types": _unique([item.type for item in evidence_items]),
            "display_format": "composite",
        },
        "_boundary_merges": [
            *_dicts(left.metadata.get("_boundary_merges")),
            *_dicts(right.metadata.get("_boundary_merges")),
            {
                "left_source_unit_ids": _strings(left.metadata.get("source_unit_ids")),
                "right_source_unit_ids": _strings(right.metadata.get("source_unit_ids")),
                "reason": decision.get("reason") if isinstance(decision.get("reason"), str) else "",
            },
        ],
    }
    warnings = [
        *_dicts(left.metadata.get("_warnings")),
        *_dicts(right.metadata.get("_warnings")),
    ]
    if len(source_unit_ids) > max_units_per_chunk:
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
        id=left.id,
        source=SourceEvidence(kind="chunk", text=source_text),
        evidence=Evidence(items=evidence_items),
        summary=summary,
        keywords=keywords,
        questions=questions,
        metadata=metadata,
    )


def _chunk_with_boundary_merge_warning(
    left: RagChunk,
    right: RagChunk,
    reason: str,
) -> RagChunk:
    metadata = dict(left.metadata)
    warnings = _dicts(metadata.get("_warnings"))
    warnings.append(
        {
            "type": "agentic_boundary_merge_failed",
            "reason": reason,
            "right_source_unit_ids": _strings(right.metadata.get("source_unit_ids")),
        }
    )
    metadata["_warnings"] = warnings
    return RagChunk(
        id=left.id,
        source=left.source,
        evidence=left.evidence,
        summary=left.summary,
        keywords=list(left.keywords),
        questions=list(left.questions),
        metadata=metadata,
    )


def _validate_unique_unit_ids(units: list[EvidenceUnit]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for unit in units:
        if unit.id in seen and unit.id not in duplicates:
            duplicates.append(unit.id)
        seen.add(unit.id)

    if duplicates:
        raise ValueError(f"duplicate unit id: {', '.join(duplicates)}")


def _validate_row_coverage(
    units: list[EvidenceUnit],
    full_assigned: set[str],
    row_ranges_by_unit: dict[str, list[tuple[int, int]]],
) -> None:
    by_id = {unit.id: unit for unit in units}
    for unit_id, ranges in row_ranges_by_unit.items():
        if unit_id in full_assigned:
            continue

        unit = by_id[unit_id]
        if unit.format != "structured_table" or not isinstance(unit.content, dict):
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
            omitted = ", ".join(str(index) for index in missing)
            raise ValueError(f"chunk plan omitted table rows for unit {unit.id}: {omitted}")


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
    max_units_per_chunk: int | None = None,
) -> RagChunk:
    source_text = "\n\n".join(source_parts)
    source_unit_ids = _unique([unit.id for unit in units])
    title = plan.get("title") if isinstance(plan.get("title"), str) else ""
    summary = plan.get("summary") if isinstance(plan.get("summary"), str) else ""
    if not summary:
        summary = _fallback_summary_from_text(source_text)

    keywords = _strings(plan.get("keywords")) or _fallback_keywords_from_text(source_text)
    question_topic = source_text.strip().replace("\n", " ")[:160] or title or summary
    questions = _strings(plan.get("questions")) or _fallback_questions(question_topic)
    metadata = {
        "source_unit_ids": source_unit_ids,
        "source_units": _source_units(units),
        "context_unit_ids": context_unit_ids,
        "operations": operations,
        "title": title,
        "common": {
            "unit_types": _unique([unit.type for unit in units]),
            "display_format": "composite",
        },
    }
    if max_units_per_chunk is not None and len(source_unit_ids) > max_units_per_chunk:
        metadata["_warnings"] = [
            {
                "type": "agentic_chunk_exceeds_max_units",
                "source_unit_count": len(source_unit_ids),
                "max_units_per_chunk": max_units_per_chunk,
            }
        ]

    return RagChunk(
        id=f"chunk-{index}",
        source=SourceEvidence(kind="chunk", text=source_text),
        evidence=Evidence(items=evidence_items),
        summary=summary,
        keywords=keywords,
        questions=questions,
        metadata=metadata,
    )


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


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


def _merge_source_units(left: Any, right: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in [*_dicts(left), *_dicts(right)]:
        unit_id = item.get("id")
        if not isinstance(unit_id, str) or unit_id in seen:
            continue
        seen.add(unit_id)
        records.append(dict(item))
    return records


def _fallback_summary(units: list[EvidenceUnit]) -> str:
    parts = [
        unit.source.text.strip().replace("\n", " ")[:160]
        for unit in units
        if unit.source.text.strip()
    ]
    return " / ".join(parts)[:500]


def _fallback_summary_from_text(text: str) -> str:
    return text.strip().replace("\n", " ")[:500]


def _fallback_keywords(units: list[EvidenceUnit]) -> list[str]:
    return _fallback_keywords_from_text("\n\n".join(unit.source.text for unit in units))


def _fallback_keywords_from_text(text: str) -> list[str]:
    words: list[str] = []
    for token in re.findall(r"[0-9A-Za-z가-힣]{2,}", text):
        if token not in words:
            words.append(token)
        if len(words) >= 8:
            return words
    return words


def _fallback_questions(topic: str) -> list[str]:
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
        if rejected_plan is not None:
            metadata["_rejected_plan"] = rejected_plan
        chunks.append(
            RagChunk(
                id=f"chunk-{index}",
                source=SourceEvidence(kind="chunk", text=unit.source.text),
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
        "units": [_unit_payload(index, unit) for index, unit in enumerate(window)],
    }
    example_unit_id = json.dumps(window[0].id if window else "unit", ensure_ascii=False)
    table_unit = next((unit for unit in window if unit.format == "structured_table"), None)
    include_rows_example = ""
    row_range = _example_row_range(table_unit) if table_unit is not None else None
    if table_unit is not None and row_range is not None:
        table_unit_id = json.dumps(table_unit.id, ensure_ascii=False)
        include_rows_example = (
            "\ninclude_rows operation 예시:\n"
            f'{{"unit_id": {table_unit_id}, "action": "include_rows", "row_ranges": [{row_range}]}}\n'
        )
    return _PROMPT.replace("{max_units}", str(max_units)).replace(
        "{example_unit_id}", example_unit_id
    ).replace(
        "{include_rows_example}", include_rows_example
    ).replace(
        "{units}", json.dumps(payload, ensure_ascii=False, indent=2)
    )


def _boundary_prompt(left: RagChunk, right: RagChunk, max_units: int) -> str:
    payload = {
        "max_units_per_chunk": max_units,
        "left_chunk": _boundary_chunk_payload(left),
        "right_chunk": _boundary_chunk_payload(right),
    }
    return _BOUNDARY_PROMPT.replace("{max_units}", str(max_units)).replace(
        "{boundary}",
        json.dumps(payload, ensure_ascii=False),
    )


def _boundary_chunk_payload(chunk: RagChunk) -> dict[str, Any]:
    return {
        "id": chunk.id,
        "source_unit_ids": _strings(chunk.metadata.get("source_unit_ids")),
        "context_unit_ids": _strings(chunk.metadata.get("context_unit_ids")),
        "unit_types": _strings(
            chunk.metadata.get("common", {}).get("unit_types")
            if isinstance(chunk.metadata.get("common"), dict)
            else None
        ),
        "summary": chunk.summary,
        "keywords": list(chunk.keywords),
        "questions": list(chunk.questions),
        "source_preview": _truncate(chunk.source.text, 900),
        "evidence_items": [
            {
                "type": item.type,
                "format": item.format,
                "source_unit_ids": list(item.source_unit_ids),
            }
            for item in chunk.evidence.items
        ],
    }


def _example_row_range(unit: EvidenceUnit) -> list[int] | None:
    if not isinstance(unit.content, dict):
        return None
    rows = unit.content.get("rows")
    if not isinstance(rows, list):
        return None
    for row in rows:
        if not isinstance(row, dict):
            continue
        index = row.get("index")
        if type(index) is int:
            return [index, index]
    return None


def _unit_payload(index: int, unit: EvidenceUnit) -> dict[str, Any]:
    common = unit.metadata.get("common", {})
    table = unit.metadata.get("table", {})
    asset = unit.metadata.get("asset", {})
    return {
        "id": unit.id,
        "index": index,
        "type": unit.type,
        "format": unit.format,
        "section_path": common.get("section_path", []) if isinstance(common, dict) else [],
        "source_preview": _truncate(unit.source.text, 900),
        "table": _compact_table(table),
        "asset": _compact_asset(asset),
    }


def _compact_table(table: Any) -> dict[str, Any]:
    if not isinstance(table, dict):
        return {}
    result: dict[str, Any] = {}
    if "table_id" in table:
        result["table_id"] = table["table_id"]
    if "headers" in table and isinstance(table["headers"], list):
        result["headers"] = [str(header)[:80] for header in table["headers"][:12]]
    if "row_count" in table:
        result["row_count"] = table["row_count"]
    return result


def _compact_asset(asset: Any) -> dict[str, Any]:
    if not isinstance(asset, dict):
        return {}
    result: dict[str, Any] = {}
    for key in ("asset_id", "kind", "mime", "ext", "uri", "public_url", "alt", "caption"):
        if key not in asset:
            continue
        value = asset[key]
        if value is None:
            result[key] = None
        elif isinstance(value, str):
            result[key] = _truncate(value, 300)
        elif type(value) in (int, float, bool):
            result[key] = value
    return result


def _truncate(value: str, limit: int) -> str:
    text = value.strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"
