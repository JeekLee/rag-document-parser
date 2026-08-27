from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from ..llm import LlmConfig, as_llm_gateway
from ..models import Evidence, EvidenceItem, EvidenceUnit, RagChunk, SourceEvidence
from ..ports.llm import LlmGateway
from .contracts import BoundaryDecision, ChunkPlanResponse
from .enrichment import Enricher, RagChunkEnricher
from .materialization import (
    ChunkPlanMaterializer,
    column_label as _column_label,
    contiguous_ranges as _contiguous_ranges,
    dict_items as _dicts,
    fallback_keywords_from_text as _fallback_keywords_from_text,
    fallback_questions as _fallback_questions,
    fallback_summary_from_text as _fallback_summary_from_text,
    source_unit_records as _source_units,
    string_items as _strings,
    table_source_text as _table_source_text,
    table_subset as _table_subset,
    unique_strings as _unique,
)
from .planning_prompt import (
    build_boundary_prompt as _boundary_prompt,
    build_plan_prompt as _plan_prompt,
)
from .tokens import count_tokens as _llm_token_count

PlanFn = Callable[[list[EvidenceUnit], LlmConfig | None, int], Any]
BoundaryMergeFn = Callable[[RagChunk, RagChunk, LlmConfig | None, int], Any]


_FORM_MARKER_PATTERN = (
    r"[\[\(][^\]\)]*별지\s*제?\s*[0-9]+(?:[-‐‑–—][0-9]+)?\s*호"
    r"(?:\s*서식)?[^\]\)]*[\]\)]"
)
_DEFAULT_TARGET_CHUNK_TOKENS = 800
_DEFAULT_MAX_CHUNK_TOKENS = 2048


@dataclass(frozen=True)
class _WindowResult:
    chunks: list[RagChunk]
    fallback_reason: str | None = None


class EvidenceUnitAgenticChunker:
    def __init__(
        self,
        *,
        llm: LlmConfig | None = None,
        llm_gateway: LlmGateway | None = None,
        max_units_per_chunk: int = 10,
        target_tokens_per_chunk: int = _DEFAULT_TARGET_CHUNK_TOKENS,
        max_tokens_per_chunk: int = _DEFAULT_MAX_CHUNK_TOKENS,
        window_size: int = 40,
        max_concurrency: int = 4,
        plan_fn: PlanFn | None = None,
        boundary_merge_fn: BoundaryMergeFn | None = None,
        final_enricher: Enricher | None = None,
        enrich_final_chunks: bool = True,
        enrichment_batch_token_budget: int | None = None,
    ) -> None:
        if llm is not None and llm_gateway is not None:
            raise ValueError("configure either llm or llm_gateway, not both")
        self._llm = llm
        self._llm_gateway = (
            llm_gateway
            if llm_gateway is not None
            else (as_llm_gateway(llm) if llm is not None else None)
        )
        self._max_units = max(1, max_units_per_chunk)
        self._materializer = ChunkPlanMaterializer(self._max_units)
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
                    llm_gateway=self._llm_gateway,
                    max_concurrency=self._concurrency,
                    batch_token_budget=enrichment_batch_token_budget,
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
            self._materializer.validate(window)
            raw_plan = self._plan_fn(window, self._llm, self._max_units)
            chunks = self._materializer.materialize(window, raw_plan)
        except Exception as exc:
            reason = _fallback_reason(exc)
            return _WindowResult(
                self._materializer.fallback(window, reason, raw_plan),
                reason,
            )
        return _WindowResult(chunks)

    def _default_plan(
        self,
        window: list[EvidenceUnit],
        cfg: LlmConfig | None,
        max_units: int,
    ) -> Any:
        if self._llm_gateway is None:
            return None
        response = self._llm_gateway.complete_model(
            _json_messages(_plan_prompt(window, max_units)),
            ChunkPlanResponse,
            schema_name="chunk_plan",
        )
        return [item.model_dump(exclude_none=True) for item in response.chunks]

    def _merge_window_boundaries(self, results: list[_WindowResult]) -> list[RagChunk]:
        if not results:
            return []
        if len(results) == 1 or (
            self._boundary_merge_fn is None and self._llm_gateway is None
        ):
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
        if self._llm_gateway is None:
            return {"action": "keep", "reason": "llm is not configured"}
        response = self._llm_gateway.complete_model(
            _json_messages(_boundary_prompt(left, right, max_units)),
            BoundaryDecision,
            schema_name="chunk_boundary_decision",
        )
        return response.model_dump()


def _json_messages(prompt: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "Return a response matching the JSON schema."},
        {"role": "user", "content": prompt},
    ]


def _windows(units: list[EvidenceUnit], size: int) -> list[list[EvidenceUnit]]:
    return [units[index : index + size] for index in range(0, len(units), size)]


def _fallback_reason(exc: Exception) -> str:
    reason = str(exc)
    if reason:
        return reason
    return exc.__class__.__name__


def _boundary_action(decision: Any) -> str:
    if not isinstance(decision, Mapping):
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
    decision = decision if isinstance(decision, Mapping) else {}
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
    improved = _split_case_boundary_chunks(improved, units_by_id, max_units_per_chunk)
    improved = _split_independent_form_chunks(improved, units_by_id, max_units_per_chunk)
    improved = _split_chunks_exceeding_max_units(
        improved,
        units_by_id,
        max_units_per_chunk,
        max_tokens_per_chunk,
    )
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


def _split_case_boundary_chunks(
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

        groups = _split_case_boundary_groups(list(chunk.evidence.items), units_by_id)
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
                        "type": "agentic_case_boundary_split",
                        "original_source_unit_count": len(source_unit_ids),
                        "split_group_index": group_index,
                        "split_group_count": len(groups),
                    },
                )
            )
    return result


def _split_case_boundary_groups(
    items: list[EvidenceItem],
    units_by_id: dict[str, EvidenceUnit],
) -> list[list[EvidenceItem]]:
    groups: list[list[EvidenceItem]] = []
    current: list[EvidenceItem] = []
    for item in items:
        if current and _item_starts_new_case(item, units_by_id):
            groups.append(current)
            current = []
        current.append(item)

    if current:
        groups.append(current)
    return groups


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


def _split_chunks_exceeding_max_units(
    chunks: list[RagChunk],
    units_by_id: dict[str, EvidenceUnit],
    max_units_per_chunk: int,
    max_tokens_per_chunk: int,
) -> list[RagChunk]:
    result: list[RagChunk] = []
    for chunk in chunks:
        source_unit_ids = _strings(chunk.metadata.get("source_unit_ids"))
        if len(source_unit_ids) <= max_units_per_chunk:
            result.append(chunk)
            continue

        groups = _evidence_item_groups_by_max_units(
            list(chunk.evidence.items),
            max_units_per_chunk,
        )
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
                    max_tokens_per_chunk=max_tokens_per_chunk,
                    regenerate_text_fields=True,
                    warning={
                        "type": "agentic_chunk_split_by_max_units",
                        "original_source_unit_count": len(source_unit_ids),
                        "max_units_per_chunk": max_units_per_chunk,
                        "split_group_index": group_index,
                        "split_group_count": len(groups),
                    },
                )
            )
    return result


def _evidence_item_groups_by_max_units(
    items: list[EvidenceItem],
    max_units_per_chunk: int,
) -> list[list[EvidenceItem]]:
    groups: list[list[EvidenceItem]] = []
    current: list[EvidenceItem] = []
    current_source_unit_ids: list[str] = []
    for item in items:
        item_source_unit_ids = list(item.source_unit_ids)
        next_source_unit_ids = _unique([*current_source_unit_ids, *item_source_unit_ids])
        if current and len(next_source_unit_ids) > max_units_per_chunk:
            groups.append(current)
            current = [item]
            current_source_unit_ids = _unique(item_source_unit_ids)
            continue
        current.append(item)
        current_source_unit_ids = next_source_unit_ids

    if current:
        groups.append(current)
    return groups


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
                source_prefix_text=split.source_prefix_text,
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
    source_prefix_text: str = ""


def _is_splittable_table_item(item: EvidenceItem) -> bool:
    return (
        item.type == "table"
        and item.format == "structured_table"
        and isinstance(item.content, Mapping)
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
    table = item.content if isinstance(item.content, Mapping) else {}
    rows = [row for row in table.get("rows", []) if isinstance(row, Mapping)]
    if len(rows) <= 1:
        return [
            _TableRowSplit(
                item=item,
                rows=rows,
                row_ranges=_row_ranges_from_rows(rows),
                context_text=_table_context_text(item, context_items, units_by_id),
            )
        ]

    repeated_context_text = _table_context_text(item, context_items, units_by_id)
    first_context_text = _table_context_text(item, [], units_by_id)
    first_source_prefix_text = _table_pending_source_text(context_items, units_by_id)
    current_rows: list[dict[str, Any]] = []
    splits: list[_TableRowSplit] = []

    for row in rows:
        candidate_rows = [*current_rows, row]
        candidate_context_text = first_context_text if not splits else repeated_context_text
        candidate_source_prefix_text = first_source_prefix_text if not splits else ""
        candidate_text = _table_split_source_text(
            item,
            candidate_context_text,
            candidate_rows,
            source_prefix_text=candidate_source_prefix_text,
        )
        if current_rows and _llm_token_count(candidate_text) > target_tokens_per_chunk:
            splits.append(
                _table_row_split(
                    item,
                    current_rows,
                    first_context_text if not splits else repeated_context_text,
                    source_prefix_text=first_source_prefix_text if not splits else "",
                )
            )
            current_rows = [row]
            continue
        current_rows = candidate_rows

    if current_rows:
        splits.append(
            _table_row_split(
                item,
                current_rows,
                first_context_text if not splits else repeated_context_text,
                source_prefix_text=first_source_prefix_text if not splits else "",
            )
        )

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
    *,
    source_prefix_text: str = "",
) -> _TableRowSplit:
    table = item.content if isinstance(item.content, Mapping) else {}
    row_ranges = _row_ranges_from_rows(rows)
    subset = _table_subset(
        table,
        [(start, end) for start, end in _row_range_tuples(row_ranges)],
    )
    subset_rows = [
        row
        for row in subset.get("rows", [])
        if isinstance(row, Mapping)
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
        source_prefix_text=source_prefix_text,
    )


def _with_large_row_warning(
    split: _TableRowSplit,
    max_tokens_per_chunk: int,
) -> _TableRowSplit:
    token_count = _llm_token_count(
        _table_split_source_text(
            split.item,
            split.context_text,
            split.rows,
            source_prefix_text=split.source_prefix_text,
        )
    )
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
        source_prefix_text=split.source_prefix_text,
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
        if isinstance(common, Mapping):
            section_path = common.get("section_path")
            if isinstance(section_path, list):
                context_parts.extend(str(part).strip() for part in section_path if str(part).strip())

    table = item.content if isinstance(item.content, Mapping) else {}
    caption = table.get("caption")
    if isinstance(caption, str) and caption.strip():
        context_parts.append(caption.strip())

    compact_columns = _compact_table_columns(table.get("columns"))
    if compact_columns:
        context_parts.append(compact_columns)

    return " | ".join(_unique([part for part in context_parts if part]))


def _table_pending_source_text(
    context_items: list[EvidenceItem],
    units_by_id: dict[str, EvidenceUnit],
) -> str:
    source_parts = [
        source_text
        for context_item in context_items
        if (source_text := _source_text_for_evidence_item(context_item, units_by_id))
    ]
    return "\n\n".join(source_parts)


def _compact_table_columns(columns: Any) -> str:
    if not isinstance(columns, list):
        return ""

    parts: list[str] = []
    previous = ""
    for index, column in enumerate(columns, start=1):
        if not isinstance(column, Mapping):
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
    *,
    source_prefix_text: str = "",
) -> str:
    table = item.content if isinstance(item.content, Mapping) else {}
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
    table_text = "\n".join(line for line in lines if line)
    return "\n\n".join(
        part
        for part in (source_prefix_text, table_text)
        if part
    )


def _compact_table_row_text(row: dict[str, Any], columns: list[Any]) -> str:
    values: list[str] = []
    cells = row.get("cells", [])
    if isinstance(cells, list):
        for cell in cells:
            if not isinstance(cell, Mapping):
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
            if isinstance(original.metadata.get("common"), Mapping)
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
            if isinstance(item.metadata, Mapping)
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
    if item.format == "structured_table" and isinstance(item.content, Mapping):
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


def _item_starts_new_case(
    item: EvidenceItem,
    units_by_id: dict[str, EvidenceUnit],
) -> bool:
    if item.type != "text":
        return False
    text = _normalize_space(_item_plain_text(item, units_by_id))
    return bool(re.match(r"^[○●]?\s*사례\s*\d+(?:\D|$)", text))


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
    if isinstance(item.content, Mapping):
        return _plain_text_from_value(item.content)
    return str(item.content)


def _plain_text_from_value(value: Any) -> str:
    parts: list[str] = []

    def collect(node: Any) -> None:
        if isinstance(node, str):
            if node.strip():
                parts.append(node.strip())
            return
        if isinstance(node, Mapping):
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


def _truncate(value: str, limit: int) -> str:
    text = value.strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _fallback_title_from_text(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:100]
    return ""


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
