from __future__ import annotations

import json
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from ..models import Evidence, EvidenceItem, EvidenceUnit, RagChunk, SourceEvidence
from .enrichment import Enricher, RagChunkEnricher
from .llm import LlmConfig, chat_json

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
- summary, keywords, questions는 최종 RagChunk 확정 후 별도 enrichment 단계에서 생성합니다.
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
    "title": "제목"
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
- Summary, keywords, and questions are generated after final chunks are fixed.
- Keep merged chunks reasonably close to {max_units} source units when possible.

Boundary payload:
{boundary}

JSON object only:
{{
  "action": "merge",
  "reason": "short reason",
  "title": "optional merged title"
}}
"""


_FORM_MARKER_PATTERN = (
    r"[\[\(][^\]\)]*별지\s*제?\s*[0-9]+(?:[-‐‑–—][0-9]+)?\s*호"
    r"(?:\s*서식)?[^\]\)]*[\]\)]"
)
_DEFAULT_TARGET_CHUNK_TOKENS = 800
_DEFAULT_MAX_CHUNK_TOKENS = 2048
_TOKEN_ENCODER: Any | None = None
_TOKEN_ENCODER_UNAVAILABLE = False


@dataclass(frozen=True)
class _WindowResult:
    chunks: list[RagChunk]
    fallback_reason: str | None = None


@dataclass(frozen=True)
class _OmittedTableRows:
    unit: EvidenceUnit
    row_indexes: list[int]


class EvidenceUnitAgenticChunker:
    def __init__(
        self,
        *,
        llm: LlmConfig | None,
        max_units_per_chunk: int = 10,
        target_tokens_per_chunk: int = _DEFAULT_TARGET_CHUNK_TOKENS,
        max_tokens_per_chunk: int = _DEFAULT_MAX_CHUNK_TOKENS,
        window_size: int = 40,
        max_concurrency: int = 4,
        plan_fn: PlanFn | None = None,
        boundary_merge_fn: BoundaryMergeFn | None = None,
        final_enricher: Enricher | None = None,
        enrich_final_chunks: bool = True,
    ) -> None:
        self._llm = llm
        self._max_units = max(1, max_units_per_chunk)
        self._target_tokens = max(1, target_tokens_per_chunk)
        self._max_tokens = max(self._target_tokens, max_tokens_per_chunk)
        self._window_size = max(1, window_size)
        self._concurrency = max(1, max_concurrency)
        self._plan_fn = plan_fn or self._default_plan
        self._boundary_merge_fn = boundary_merge_fn
        self._final_enricher = (
            final_enricher
            if final_enricher is not None
            else (
                RagChunkEnricher(
                    llm=llm,
                    chat_fn=chat_json,
                    max_concurrency=self._concurrency,
                )
                if enrich_final_chunks
                else None
            )
        )

    def chunk(self, units: list[EvidenceUnit]) -> list[RagChunk]:
        if not units:
            return []

        windows = _windows(units, self._window_size)
        with ThreadPoolExecutor(max_workers=self._concurrency) as executor:
            results = list(executor.map(self._chunk_window, windows))

        chunks = self._merge_window_boundaries(results)
        chunks = _improve_chunk_boundaries(
            chunks,
            units,
            self._max_units,
            self._target_tokens,
            self._max_tokens,
        )

        final_chunks = [
            _reindex_chunk(index, chunk)
            for index, chunk in enumerate(chunks, start=1)
        ]
        if self._final_enricher is not None:
            return self._final_enricher.enrich(final_chunks)
        return final_chunks

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
                    right = next_chunks[0]
                    source_unit_count = _combined_source_unit_count(chunks[-1], right)
                    if source_unit_count > self._max_units:
                        chunks[-1] = _chunk_with_boundary_merge_warning(
                            chunks[-1],
                            right,
                            f"boundary merge would exceed max_units_per_chunk: "
                            f"{source_unit_count} > {self._max_units}",
                            warning_type="agentic_boundary_merge_exceeds_max_units",
                        )
                    else:
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


def _combined_source_unit_count(left: RagChunk, right: RagChunk) -> int:
    return len(
        _unique(
            [
                *_strings(left.metadata.get("source_unit_ids")),
                *_strings(right.metadata.get("source_unit_ids")),
            ]
        )
    )


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
    *,
    warning_type: str = "agentic_boundary_merge_failed",
) -> RagChunk:
    metadata = dict(left.metadata)
    warnings = _dicts(metadata.get("_warnings"))
    warnings.append(
        {
            "type": warning_type,
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


def _improve_chunk_boundaries(
    chunks: list[RagChunk],
    units: list[EvidenceUnit],
    max_units_per_chunk: int,
    target_tokens_per_chunk: int,
    max_tokens_per_chunk: int,
) -> list[RagChunk]:
    if not chunks:
        return []

    units_by_id = {unit.id: unit for unit in units}
    improved = _move_trailing_heading_units_forward(chunks, units_by_id, max_units_per_chunk)
    improved = _split_independent_form_chunks(improved, units_by_id, max_units_per_chunk)
    return _split_large_table_chunks(
        improved,
        units_by_id,
        max_units_per_chunk,
        target_tokens_per_chunk,
        max_tokens_per_chunk,
    )


def _move_trailing_heading_units_forward(
    chunks: list[RagChunk],
    units_by_id: dict[str, EvidenceUnit],
    max_units_per_chunk: int,
) -> list[RagChunk]:
    result = list(chunks)
    index = 0
    while index < len(result) - 1:
        left = result[index]
        right = result[index + 1]
        left_items = list(left.evidence.items)
        right_items = list(right.evidence.items)
        moving: list[EvidenceItem] = []

        while left_items and _is_forward_heading_item(left_items[-1], units_by_id):
            moving.insert(0, left_items.pop())

        if not moving:
            index += 1
            continue

        moved_source_unit_ids = _item_source_unit_ids(moving)
        right_warning = {
            "type": "agentic_heading_moved_forward",
            "source_unit_ids": moved_source_unit_ids,
        }
        result[index + 1] = _rebuild_chunk_from_items(
            right,
            [*moving, *right_items],
            units_by_id,
            max_units_per_chunk=max_units_per_chunk,
            warning=right_warning,
        )

        if left_items:
            result[index] = _rebuild_chunk_from_items(
                left,
                left_items,
                units_by_id,
                max_units_per_chunk=max_units_per_chunk,
                warning={
                    "type": "agentic_trailing_heading_removed",
                    "source_unit_ids": moved_source_unit_ids,
                },
            )
            index += 1
            continue

        del result[index]

    return result


def _split_independent_form_chunks(
    chunks: list[RagChunk],
    units_by_id: dict[str, EvidenceUnit],
    max_units_per_chunk: int,
) -> list[RagChunk]:
    result: list[RagChunk] = []
    for chunk in chunks:
        source_unit_ids = _strings(chunk.metadata.get("source_unit_ids"))
        if len(source_unit_ids) <= 1:
            result.append(chunk)
            continue

        groups = _split_form_item_groups(list(chunk.evidence.items), units_by_id)
        if len(groups) <= 1:
            result.append(chunk)
            continue

        for group_index, group in enumerate(groups, start=1):
            result.append(
                _rebuild_chunk_from_items(
                    chunk,
                    group,
                    units_by_id,
                    max_units_per_chunk=max_units_per_chunk,
                    regenerate_text_fields=True,
                    warning={
                        "type": "agentic_independent_form_boundary_split",
                        "original_source_unit_count": len(source_unit_ids),
                        "split_group_index": group_index,
                        "split_group_count": len(groups),
                    },
                )
            )
    return result


def _split_large_table_chunks(
    chunks: list[RagChunk],
    units_by_id: dict[str, EvidenceUnit],
    max_units_per_chunk: int,
    target_tokens_per_chunk: int,
    max_tokens_per_chunk: int,
) -> list[RagChunk]:
    result: list[RagChunk] = []
    for chunk in chunks:
        if _llm_token_count(chunk.source.text) <= max_tokens_per_chunk:
            result.append(chunk)
            continue

        split_chunks = _split_large_table_chunk(
            chunk,
            units_by_id,
            max_units_per_chunk,
            target_tokens_per_chunk,
            max_tokens_per_chunk,
        )
        result.extend(split_chunks)
    return result


def _split_large_table_chunk(
    chunk: RagChunk,
    units_by_id: dict[str, EvidenceUnit],
    max_units_per_chunk: int,
    target_tokens_per_chunk: int,
    max_tokens_per_chunk: int,
) -> list[RagChunk]:
    result: list[RagChunk] = []
    pending_context_items: list[EvidenceItem] = []
    original_token_count = _llm_token_count(chunk.source.text)
    did_split = False

    for item in chunk.evidence.items:
        if not _is_splittable_table_item(item):
            pending_context_items.append(item)
            continue

        table_splits = _split_table_item_by_rows(
            item,
            pending_context_items,
            units_by_id,
            target_tokens_per_chunk,
            max_tokens_per_chunk,
        )
        if len(table_splits) <= 1:
            pending_context_items.append(item)
            continue

        did_split = True
        for split_index, split in enumerate(table_splits, start=1):
            evidence_items = [*pending_context_items, split.item] if split_index == 1 else [split.item]
            source_text = _table_split_source_text(
                split.item,
                split.context_text,
                split.rows,
            )
            result.append(
                _rebuild_chunk_from_items(
                    chunk,
                    evidence_items,
                    units_by_id,
                    max_units_per_chunk=max_units_per_chunk,
                    max_tokens_per_chunk=max_tokens_per_chunk,
                    regenerate_text_fields=True,
                    source_text_override=source_text,
                    warning={
                        "type": "agentic_table_split_by_token_budget",
                        "parent_source_unit_ids": list(split.item.source_unit_ids),
                        "row_ranges": split.row_ranges,
                        "original_token_count": original_token_count,
                        "target_tokens_per_chunk": target_tokens_per_chunk,
                        "max_tokens_per_chunk": max_tokens_per_chunk,
                    },
                )
            )
        pending_context_items = []

    if pending_context_items:
        rebuilt = _rebuild_chunk_from_items(
            chunk,
            pending_context_items,
            units_by_id,
            max_units_per_chunk=max_units_per_chunk,
            max_tokens_per_chunk=max_tokens_per_chunk,
            regenerate_text_fields=did_split,
        )
        result.append(rebuilt)

    return result if did_split else [chunk]


@dataclass(frozen=True)
class _TableRowSplit:
    item: EvidenceItem
    rows: list[dict[str, Any]]
    row_ranges: list[list[int]]
    context_text: str


def _is_splittable_table_item(item: EvidenceItem) -> bool:
    return (
        item.type == "table"
        and item.format == "structured_table"
        and isinstance(item.content, dict)
        and isinstance(item.content.get("rows"), list)
        and len(item.content.get("rows", [])) > 1
    )


def _split_table_item_by_rows(
    item: EvidenceItem,
    context_items: list[EvidenceItem],
    units_by_id: dict[str, EvidenceUnit],
    target_tokens_per_chunk: int,
    max_tokens_per_chunk: int,
) -> list[_TableRowSplit]:
    table = item.content if isinstance(item.content, dict) else {}
    rows = [row for row in table.get("rows", []) if isinstance(row, dict)]
    if len(rows) <= 1:
        return [
            _TableRowSplit(
                item=item,
                rows=rows,
                row_ranges=_row_ranges_from_rows(rows),
                context_text=_table_context_text(item, context_items, units_by_id),
            )
        ]

    context_text = _table_context_text(item, context_items, units_by_id)
    current_rows: list[dict[str, Any]] = []
    splits: list[_TableRowSplit] = []

    for row in rows:
        candidate_rows = [*current_rows, row]
        candidate_text = _table_split_source_text(item, context_text, candidate_rows)
        if current_rows and _llm_token_count(candidate_text) > target_tokens_per_chunk:
            splits.append(_table_row_split(item, current_rows, context_text))
            current_rows = [row]
            continue
        current_rows = candidate_rows

    if current_rows:
        splits.append(_table_row_split(item, current_rows, context_text))

    if len(splits) <= 1:
        return splits

    return [
        _with_large_row_warning(split, max_tokens_per_chunk)
        for split in splits
    ]


def _table_row_split(
    item: EvidenceItem,
    rows: list[dict[str, Any]],
    context_text: str,
) -> _TableRowSplit:
    table = item.content if isinstance(item.content, dict) else {}
    row_ranges = _row_ranges_from_rows(rows)
    subset = _table_subset(
        table,
        [(start, end) for start, end in _row_range_tuples(row_ranges)],
    )
    subset_rows = [
        row
        for row in subset.get("rows", [])
        if isinstance(row, dict)
    ]
    metadata = dict(item.metadata)
    metadata["row_ranges"] = row_ranges
    metadata["agentic_table_split"] = {
        "strategy": "token_budget_rows",
        "row_ranges": row_ranges,
    }
    return _TableRowSplit(
        item=EvidenceItem(
            type=item.type,
            format=item.format,
            content=subset,
            source_unit_ids=list(item.source_unit_ids),
            metadata=metadata,
        ),
        rows=subset_rows,
        row_ranges=row_ranges,
        context_text=context_text,
    )


def _with_large_row_warning(
    split: _TableRowSplit,
    max_tokens_per_chunk: int,
) -> _TableRowSplit:
    token_count = _llm_token_count(_table_split_source_text(split.item, split.context_text, split.rows))
    if token_count <= max_tokens_per_chunk:
        return split

    metadata = dict(split.item.metadata)
    warnings = _dicts(metadata.get("_warnings"))
    warnings.append(
        {
            "type": "agentic_table_row_group_exceeds_max_tokens",
            "token_count": token_count,
            "max_tokens_per_chunk": max_tokens_per_chunk,
            "row_ranges": split.row_ranges,
        }
    )
    metadata["_warnings"] = warnings
    return _TableRowSplit(
        item=EvidenceItem(
            type=split.item.type,
            format=split.item.format,
            content=split.item.content,
            source_unit_ids=list(split.item.source_unit_ids),
            metadata=metadata,
        ),
        rows=split.rows,
        row_ranges=split.row_ranges,
        context_text=split.context_text,
    )


def _table_context_text(
    item: EvidenceItem,
    context_items: list[EvidenceItem],
    units_by_id: dict[str, EvidenceUnit],
) -> str:
    context_parts = [
        _normalize_space(_item_plain_text(context_item, units_by_id))
        for context_item in context_items
        if context_item.type == "text"
    ]
    source_unit_ids = list(item.source_unit_ids)
    if len(source_unit_ids) == 1 and source_unit_ids[0] in units_by_id:
        unit = units_by_id[source_unit_ids[0]]
        common = unit.metadata.get("common", {})
        if isinstance(common, dict):
            section_path = common.get("section_path")
            if isinstance(section_path, list):
                context_parts.extend(str(part).strip() for part in section_path if str(part).strip())

    table = item.content if isinstance(item.content, dict) else {}
    caption = table.get("caption")
    if isinstance(caption, str) and caption.strip():
        context_parts.append(caption.strip())

    compact_columns = _compact_table_columns(table.get("columns"))
    if compact_columns:
        context_parts.append(compact_columns)

    context = " | ".join(_unique([part for part in context_parts if part]))
    return _truncate(context, 1200)


def _compact_table_columns(columns: Any) -> str:
    if not isinstance(columns, list):
        return ""

    parts: list[str] = []
    previous = ""
    for index, column in enumerate(columns, start=1):
        if not isinstance(column, dict):
            continue
        label = _normalize_space(str(column.get("text", "")))
        if not label or label == previous:
            continue
        previous = label
        parts.append(f"c{index}: {_truncate(label, 48)}")
        if len("; ".join(parts)) > 900:
            break
    return "columns: " + "; ".join(parts) if parts else ""


def _table_split_source_text(
    item: EvidenceItem,
    context_text: str,
    rows: list[dict[str, Any]],
) -> str:
    table = item.content if isinstance(item.content, dict) else {}
    columns = table.get("columns", [])
    if not isinstance(columns, list):
        columns = []

    row_ranges = _row_ranges_from_rows(rows)
    lines = [
        f"table rows: {_row_ranges_label(row_ranges)}",
    ]
    if context_text:
        lines.append(f"context: {context_text}")
    for row in rows:
        lines.append(_compact_table_row_text(row, columns))
    return "\n".join(line for line in lines if line)


def _compact_table_row_text(row: dict[str, Any], columns: list[Any]) -> str:
    values: list[str] = []
    cells = row.get("cells", [])
    if isinstance(cells, list):
        for cell in cells:
            if not isinstance(cell, dict):
                continue
            text = _normalize_space(str(cell.get("text", "")))
            if not text:
                continue
            label = _compact_column_label(columns, cell.get("column_id"))
            if label and label != text:
                values.append(f"{label}: {_truncate(text, 120)}")
            else:
                values.append(_truncate(text, 120))
    return f"row {row.get('index', '?')}: " + "; ".join(values)


def _compact_column_label(columns: list[Any], column_id: Any) -> str:
    label = _column_label(columns, column_id)
    if label.startswith("col") or label == column_id:
        return label
    return _truncate(_normalize_space(label), 36)


def _row_ranges_from_rows(rows: list[dict[str, Any]]) -> list[list[int]]:
    indexes = [
        row.get("index")
        for row in rows
        if type(row.get("index")) is int
    ]
    return [[start, end] for start, end in _contiguous_ranges(indexes)]


def _row_ranges_label(row_ranges: list[list[int]]) -> str:
    return ", ".join(
        str(start) if start == end else f"{start}-{end}"
        for start, end in row_ranges
    )


def _split_form_item_groups(
    items: list[EvidenceItem],
    units_by_id: dict[str, EvidenceUnit],
) -> list[list[EvidenceItem]]:
    groups: list[list[EvidenceItem]] = []
    current: list[EvidenceItem] = []
    pending_form_end = False
    for item in items:
        if (
            current
            and (
                (
                    _item_starts_new_form(item, units_by_id)
                    and _group_has_form_body(current, units_by_id)
                )
                or (
                    pending_form_end
                    and _item_can_start_after_form_end(item, units_by_id)
                )
            )
        ):
            groups.append(current)
            current = []
            pending_form_end = False
        current.append(item)
        pending_form_end = _item_ends_form(item, units_by_id)

    if current:
        groups.append(current)
    return groups


def _group_has_form_body(
    items: list[EvidenceItem],
    units_by_id: dict[str, EvidenceUnit],
) -> bool:
    for item in items:
        if _is_deleted_form_item(item, units_by_id):
            return True
        if not _is_form_heading_only_item(item, units_by_id):
            return True
    return False


def _rebuild_chunk_from_items(
    original: RagChunk,
    evidence_items: list[EvidenceItem],
    units_by_id: dict[str, EvidenceUnit],
    *,
    max_units_per_chunk: int,
    max_tokens_per_chunk: int | None = None,
    regenerate_text_fields: bool = False,
    source_text_override: str | None = None,
    warning: dict[str, Any] | None = None,
) -> RagChunk:
    source_unit_ids = _item_source_unit_ids(evidence_items)
    chunk_units = [units_by_id[unit_id] for unit_id in source_unit_ids if unit_id in units_by_id]
    source_parts = [
        source_text
        for item in evidence_items
        if (source_text := _source_text_for_evidence_item(item, units_by_id))
    ]
    source_text = source_text_override if source_text_override is not None else "\n\n".join(source_parts)

    if regenerate_text_fields:
        title = _fallback_title_from_text(source_text)
        summary = _fallback_summary_from_text(source_text)
        keywords = _fallback_keywords_from_text(source_text)
        questions = _fallback_questions((source_text or title)[:160])
    else:
        title = original.metadata.get("title") if isinstance(original.metadata.get("title"), str) else ""
        summary = original.summary or _fallback_summary_from_text(source_text)
        keywords = list(original.keywords) or _fallback_keywords_from_text(source_text)
        questions = list(original.questions) or _fallback_questions((source_text or title)[:160])

    metadata = dict(original.metadata)
    metadata["source_unit_ids"] = source_unit_ids
    metadata["source_units"] = _source_units(chunk_units)
    metadata["context_unit_ids"] = [
        unit_id
        for unit_id in _strings(original.metadata.get("context_unit_ids"))
        if unit_id not in source_unit_ids
    ]
    metadata["operations"] = _operations_from_evidence_items(evidence_items)
    metadata["title"] = title
    metadata["common"] = {
        **(
            dict(original.metadata.get("common"))
            if isinstance(original.metadata.get("common"), dict)
            else {}
        ),
        "unit_types": _unique([item.type for item in evidence_items]),
        "display_format": "composite",
    }

    warnings = [
        item
        for item in _dicts(original.metadata.get("_warnings"))
        if item.get("type") != "agentic_chunk_exceeds_max_units"
    ]
    if warning is not None:
        warnings.append(warning)
    if len(source_unit_ids) > max_units_per_chunk:
        warnings.append(
            {
                "type": "agentic_chunk_exceeds_max_units",
                "source_unit_count": len(source_unit_ids),
                "max_units_per_chunk": max_units_per_chunk,
            }
        )
    if max_tokens_per_chunk is not None:
        token_count = _llm_token_count(source_text)
        if token_count > max_tokens_per_chunk:
            warnings.append(
                {
                    "type": "agentic_chunk_exceeds_max_tokens",
                    "token_count": token_count,
                    "max_tokens_per_chunk": max_tokens_per_chunk,
                }
            )
    if warnings:
        metadata["_warnings"] = warnings
    else:
        metadata.pop("_warnings", None)
    if (
        regenerate_text_fields
        or source_text_override is not None
        or source_unit_ids != _strings(original.metadata.get("source_unit_ids"))
    ):
        metadata["_needs_enrichment"] = True

    return RagChunk(
        id=original.id,
        source=SourceEvidence(kind="chunk", text=source_text),
        evidence=Evidence(items=evidence_items),
        summary=summary,
        keywords=keywords,
        questions=questions,
        metadata=metadata,
    )


def _operations_from_evidence_items(items: list[EvidenceItem]) -> list[dict[str, Any]]:
    operations: list[dict[str, Any]] = []
    for item in items:
        unit_ids = list(item.source_unit_ids)
        row_ranges = (
            item.metadata.get("row_ranges")
            if isinstance(item.metadata, dict)
            else None
        )
        if len(unit_ids) == 1 and isinstance(row_ranges, list):
            operations.append(
                {
                    "unit_id": unit_ids[0],
                    "action": "include_rows",
                    "row_ranges": row_ranges,
                }
            )
            continue
        for unit_id in unit_ids:
            operations.append({"unit_id": unit_id, "action": "include"})
    return operations


def _source_text_for_evidence_item(
    item: EvidenceItem,
    units_by_id: dict[str, EvidenceUnit],
) -> str:
    if item.format == "structured_table" and isinstance(item.content, dict):
        return _table_source_text(item.content)

    source_unit_ids = list(item.source_unit_ids)
    if len(source_unit_ids) == 1 and source_unit_ids[0] in units_by_id:
        return units_by_id[source_unit_ids[0]].source.text

    if isinstance(item.content, str):
        return item.content
    try:
        return json.dumps(item.content, ensure_ascii=False)
    except TypeError:
        return str(item.content)


def _item_source_unit_ids(items: list[EvidenceItem]) -> list[str]:
    return _unique(
        [
            unit_id
            for item in items
            for unit_id in item.source_unit_ids
            if isinstance(unit_id, str)
        ]
    )


def _is_forward_heading_item(
    item: EvidenceItem,
    units_by_id: dict[str, EvidenceUnit],
) -> bool:
    if item.type != "text":
        return False
    if _is_deleted_form_item(item, units_by_id):
        return False

    text = _item_plain_text(item, units_by_id)
    if not _looks_like_heading_text(text):
        return False
    return len(list(item.source_unit_ids)) == 1


def _looks_like_heading_text(text: str) -> bool:
    normalized = _normalize_space(text)
    if not normalized or len(normalized) > 120:
        return False
    if "\n" in text.strip():
        return False
    if re.match(r"^\d{1,2}\.\s+\S", normalized):
        return True
    if re.match(r"^[가-힣]\.\s+\S", normalized):
        return True
    if re.match(rf"^{_FORM_MARKER_PATTERN}\s*$", normalized):
        return True
    if re.match(rf"^{_FORM_MARKER_PATTERN}\s*[^.?!]{{1,60}}$", normalized):
        return True
    return False


def _is_form_heading_only_item(
    item: EvidenceItem,
    units_by_id: dict[str, EvidenceUnit],
) -> bool:
    if item.type != "text":
        return False
    text = _normalize_space(_item_plain_text(item, units_by_id))
    return bool(re.match(rf"^{_FORM_MARKER_PATTERN}", text)) and len(text) <= 120


def _is_deleted_form_item(
    item: EvidenceItem,
    units_by_id: dict[str, EvidenceUnit],
) -> bool:
    text = _normalize_space(_item_plain_text(item, units_by_id))
    return bool(re.match(rf"^{_FORM_MARKER_PATTERN}\s*<[^>]*삭\s*제[^>]*>$", text))


def _item_starts_new_form(
    item: EvidenceItem,
    units_by_id: dict[str, EvidenceUnit],
) -> bool:
    text = _normalize_space(_item_plain_text(item, units_by_id))
    if not text:
        return False
    if re.match(r"^\(?\s*(뒷면|뒤\s*쪽|을지|앞면|앞\s*쪽)\s*\)?$", text[:20]):
        return False
    return bool(re.search(_FORM_MARKER_PATTERN, text[:300]))


def _item_ends_form(
    item: EvidenceItem,
    units_by_id: dict[str, EvidenceUnit],
) -> bool:
    text = _normalize_space(_item_plain_text(item, units_by_id))
    if not re.search(r"[0-9]{2,3}\s*m{1,3}\s*[×xX]\s*[0-9]{2,3}\s*m{1,3}", text):
        return False
    return any(token in text for token in ("신문용지", "일반용지", "재활용품", "g/m", "g/㎡"))


def _item_can_start_after_form_end(
    item: EvidenceItem,
    units_by_id: dict[str, EvidenceUnit],
) -> bool:
    if item.type not in {"text", "table"}:
        return False
    text = _normalize_space(_item_plain_text(item, units_by_id))
    return not bool(re.match(r"^\(?\s*(뒷면|뒤\s*쪽|을지|앞면|앞\s*쪽)\s*\)?$", text[:20]))


def _item_plain_text(
    item: EvidenceItem,
    units_by_id: dict[str, EvidenceUnit],
) -> str:
    if isinstance(item.content, str):
        return item.content
    source_unit_ids = list(item.source_unit_ids)
    if len(source_unit_ids) == 1 and source_unit_ids[0] in units_by_id:
        return units_by_id[source_unit_ids[0]].source.text
    if isinstance(item.content, dict):
        return _plain_text_from_value(item.content)
    return str(item.content)


def _plain_text_from_value(value: Any) -> str:
    parts: list[str] = []

    def collect(node: Any) -> None:
        if isinstance(node, str):
            if node.strip():
                parts.append(node.strip())
            return
        if isinstance(node, dict):
            text = node.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
            for child in node.values():
                if isinstance(child, (dict, list)):
                    collect(child)
            return
        if isinstance(node, list):
            for child in node:
                collect(child)

    collect(value)
    return " ".join(parts)


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _fallback_title_from_text(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:100]
    return ""


def _llm_token_count(text: str) -> int:
    encoder = _token_encoder()
    if encoder is not None:
        return len(encoder.encode(text or ""))
    return _estimated_token_count(text)


def _token_encoder() -> Any | None:
    global _TOKEN_ENCODER, _TOKEN_ENCODER_UNAVAILABLE
    if _TOKEN_ENCODER is not None:
        return _TOKEN_ENCODER
    if _TOKEN_ENCODER_UNAVAILABLE:
        return None

    try:
        import tiktoken  # type: ignore[import-not-found]
    except Exception:
        _TOKEN_ENCODER_UNAVAILABLE = True
        return None

    try:
        _TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _TOKEN_ENCODER_UNAVAILABLE = True
        return None
    return _TOKEN_ENCODER


def _estimated_token_count(text: str) -> int:
    count = 0
    for token in re.findall(r"[A-Za-z0-9_]+|[가-힣]|[^\s]", text or ""):
        if re.fullmatch(r"[A-Za-z0-9_]+", token):
            count += max(1, (len(token) + 3) // 4)
        else:
            count += 1
    return count


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
        row_ranges = _contiguous_ranges(omitted.row_indexes)
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
                "summary": _fallback_summary_from_text(source_text),
                "keywords": _fallback_keywords_from_text(source_text),
                "questions": _fallback_questions(source_text[:80]),
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
            omitted.append(_OmittedTableRows(unit, missing))
    return omitted


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


def _normalize_plan_operations(
    operations: list[Any],
    by_id: dict[str, EvidenceUnit],
) -> tuple[list[Any], list[dict[str, Any]]]:
    full_include_unit_ids = {
        operation.get("unit_id")
        for operation in operations
        if (
            isinstance(operation, dict)
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
            isinstance(operation, dict)
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
            "unit_ids": _unique(ignored_row_unit_ids),
        }
    ]


def _is_structured_table_unit(unit: EvidenceUnit | None) -> bool:
    return (
        unit is not None
        and unit.format == "structured_table"
        and isinstance(unit.content, dict)
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
    rows = table.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError("structured_table content requires rows")

    _validate_row_range_bounds(ranges, _table_row_indexes(rows))

    selected_indexes = {
        index
        for row in rows
        if (
            isinstance(row, dict)
            and type(row.get("index")) is int
            and _row_selected(row["index"], ranges)
        )
        for index in [row["index"]]
    }
    carried_by_row = _rowspan_context_cells_by_row(table, rows, selected_indexes)
    selected: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        index = row.get("index")
        if type(index) is int and _row_selected(index, ranges):
            selected.append(_row_with_carried_cells(row, carried_by_row.get(index, []), table))

    if not selected:
        raise ValueError("row_ranges selected no rows")

    subset = dict(table)
    subset["rows"] = selected
    return subset


def _row_range_tuples(row_ranges: list[list[int]]) -> list[tuple[int, int]]:
    return [
        (row_range[0], row_range[1])
        for row_range in row_ranges
        if (
            isinstance(row_range, list)
            and len(row_range) == 2
            and type(row_range[0]) is int
            and type(row_range[1]) is int
        )
    ]


def _rowspan_context_cells_by_row(
    table: dict[str, Any],
    rows: list[Any],
    selected_indexes: set[int],
) -> dict[int, list[dict[str, Any]]]:
    if not selected_indexes:
        return {}

    result: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        start_index = row.get("index")
        if type(start_index) is not int:
            continue
        cells = row.get("cells", [])
        if not isinstance(cells, list):
            continue
        for cell in cells:
            if not isinstance(cell, dict):
                continue
            rowspan = _positive_int(cell.get("rowspan"))
            if rowspan <= 1:
                continue
            covered = [
                index
                for index in sorted(selected_indexes)
                if start_index < index < start_index + rowspan
            ]
            for group in _contiguous_ranges(covered):
                carried = dict(cell)
                carried["rowspan"] = group[1] - group[0] + 1
                metadata = (
                    dict(carried.get("metadata"))
                    if isinstance(carried.get("metadata"), dict)
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
        if isinstance(cell, dict)
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
        if isinstance(column, dict) and column.get("id") == column_id:
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
    *,
    plan_warnings: list[dict[str, Any]] | None = None,
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
    operations = _dicts(chunk.metadata.get("operations"))
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

    source_unit_ids = _strings(chunk.metadata.get("source_unit_ids"))
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


def _contiguous_ranges(indexes: list[int]) -> list[tuple[int, int]]:
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
