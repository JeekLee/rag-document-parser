"""Bound planning requests, retaining original IDs and row coordinates."""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from ..models import EvidenceUnit, SourceEvidence
from .contracts import ChunkPlanResponse
from .materialization import TableRowIndex, contiguous_ranges, table_source_text
from .planning_prompt import PLAN_ROW_RANGES, build_plan_prompt, json_messages


def planning_request_text(prompt: str, response_model: type[BaseModel]) -> str:
    """Canonical counting input, including system/user messages and JSON schema.

    Provider-specific framing and tokenization can differ. A caller may supply
    its own counter and should leave headroom for provider-added overhead.
    """
    return json.dumps(
        {"messages": json_messages(prompt), "response_schema": response_model.model_json_schema()},
        ensure_ascii=False, separators=(",", ":"),
    )


@dataclass(frozen=True)
class PlanningWindow:
    units: list[EvidenceUnit]
    input_tokens: int


@dataclass(frozen=True)
class PlanInputBudget:
    max_tokens: int
    max_units: int
    target_chunk_tokens: int
    max_chunk_tokens: int
    token_counter: Callable[[str], int]

    def count(self, prompt: str, response_model: type[BaseModel]) -> int:
        count = self.token_counter(planning_request_text(prompt, response_model))
        if type(count) is not int or count < 0:
            raise ValueError("plan_token_counter must return a nonnegative integer")
        return count

    def plan_tokens(self, units: list[EvidenceUnit]) -> int:
        return self.count(build_plan_prompt(
            units, self.max_units, self.target_chunk_tokens, self.max_chunk_tokens,
        ), ChunkPlanResponse)

    def windows(self, units: list[EvidenceUnit], window_size: int) -> list[PlanningWindow]:
        """Split unit windows first, then a single oversized table by rows.

        Every emitted window fits, except indivisible units/rows. The caller
        must route those to a local fallback without invoking the planner.
        No window contains two views with the same original unit ID.
        """
        result: list[PlanningWindow] = []

        def split_window(window: list[EvidenceUnit]) -> None:
            count = self.plan_tokens(window)
            if count <= self.max_tokens:
                result.append(PlanningWindow(window, count))
            elif len(window) > 1:
                middle = len(window) // 2
                split_window(window[:middle])
                split_window(window[middle:])
            else:
                unit = window[0]
                rows = unit.content.get("rows", []) if isinstance(unit.content, Mapping) else []
                if unit.format != "structured_table" or not isinstance(rows, list) or len(rows) <= 1:
                    result.append(PlanningWindow(window, count))
                    return
                index = TableRowIndex(unit.content)

                def split_rows(start: int, end: int) -> None:
                    view = _table_view(unit, rows[start:end], index)
                    tokens = self.plan_tokens([view])
                    if tokens <= self.max_tokens or end - start == 1:
                        result.append(PlanningWindow([view], tokens))
                    else:
                        middle = (start + end) // 2
                        split_rows(start, middle)
                        split_rows(middle, end)

                middle = len(rows) // 2
                split_rows(0, middle)
                split_rows(middle, len(rows))

        for start in range(0, len(units), window_size):
            split_window(units[start:start + window_size])
        return result


def _table_view(unit: EvidenceUnit, rows: list[Any], index: TableRowIndex) -> EvidenceUnit:
    ranges = contiguous_ranges([row["index"] for row in rows])
    table = index.subset(ranges)
    metadata = {**unit.metadata, PLAN_ROW_RANGES: [[start, end] for start, end in ranges]}
    if isinstance(metadata.get("table"), Mapping):
        metadata["table"] = {**metadata["table"], "row_count": len(rows)}
    return EvidenceUnit(
        id=unit.id, type=unit.type, format=unit.format, content=table,
        source=SourceEvidence(kind=unit.source.kind, text=table_source_text(table)),
        metadata=metadata,
    )
