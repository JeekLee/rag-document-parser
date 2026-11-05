from __future__ import annotations

import json

import pytest

from rag_document_parser import EvidenceUnit, EvidenceUnitAgenticChunker, SourceEvidence
from rag_document_parser.chunk.contracts import ChunkPlanResponse
from rag_document_parser.chunk.planning_prompt import build_plan_prompt
from rag_document_parser.chunk.tokens import count_tokens


def _table(size=40):
    rows = []
    for number in range(1, size + 1):
        # Sparse original indexes; a merged record crosses planning windows.
        index = number + (number > 20)
        cells = [{"column_id": "c2", "text": f"row {number}: " + "full source text. " * 12}]
        if number == 1:
            cells.insert(0, {"column_id": "c1", "text": "shared record", "rowspan": size + 1})
        if number % 7 == 0:
            cells[-1]["children"] = [{"type": "text", "format": "plain", "content": f"nested {number}"}]
        rows.append({"index": index, "cells": cells})
    return EvidenceUnit(
        id="original-table", type="table", format="structured_table",
        source=SourceEvidence(kind="table", text="Original table overview"),
        content={
            "caption": "Original caption",
            "columns": [{"id": "c1", "text": "Key"}, {"id": "c2", "text": "Body"}],
            "header_rows": [{"index": 1, "cells": [{"column_id": "c1", "text": "Both columns", "colspan": 2}]}],
            "rows": rows,
        },
        metadata={"table": {"table_id": "original", "headers": ["Key", "Body"], "row_count": size}},
    )


def _text(id):
    return EvidenceUnit(id=id, type="text", format="plain", content=f"Text {id}.",
                        source=SourceEvidence(kind="text", text=f"Text {id}."))


class Gateway:
    def __init__(self, *, partial=False, fail_second=False):
        self.requests = []
        self.plans = []
        self.partial = partial
        self.fail_second = fail_second

    def complete_model(self, messages, response_model, *, schema_name):
        self.requests.append((messages, response_model))
        if schema_name == "chunk_boundary_decision":
            return response_model(action="keep", reason="test boundary", title="")
        payload = json.JSONDecoder().raw_decode(messages[-1]["content"].split("Unit 목록:\n", 1)[1])[0]
        self.plans.append(payload)
        if self.fail_second and len(self.plans) == 2:
            raise RuntimeError("simulated planner failure")
        operations = []
        for unit in payload["units"]:
            if self.partial and unit["format"] == "structured_table":
                rows = unit["table"]["rows"]
                operations.append({"unit_id": unit["id"], "action": "include_rows",
                                   "row_ranges": [[rows[0]["index"], rows[0]["index"]]]})
            else:
                operations.append({"unit_id": unit["id"], "action": "include", "row_ranges": None})
        return response_model.model_validate({"chunks": [{
            "unit_ids": [u["id"] for u in payload["units"]], "operations": operations,
            "context_unit_ids": [], "title": "Test plan",
        }]})


def _assert_budget(gateway, maximum, counter):
    for messages, model in gateway.requests:
        actual_request = json.dumps(
            {"messages": messages, "response_schema": model.model_json_schema()},
            ensure_ascii=False, separators=(",", ":"),
        )
        assert counter(actual_request) <= maximum


def _assert_coverage(chunks, original):
    rows = [row for chunk in chunks for item in chunk.evidence.items
            if item.source_unit_ids == [original.id] for row in item.content.rows]
    assert [row.index for row in rows] == [row.index for row in original.content.rows]
    for expected, actual in zip(original.content.rows, rows):
        # Carried merge origins are context; all primary text/children remain.
        actual_cells = [cell for cell in actual.cells if not cell.metadata.get("rowspan_context")]
        assert [(c.column_id, c.text, c.children) for c in actual_cells] == [
            (c.column_id, c.text, c.children) for c in expected.cells
        ]
    for chunk in chunks:
        for item in chunk.evidence.items:
            if item.source_unit_ids == [original.id]:
                assert item.content.columns == original.content.columns
                assert item.content.header_rows == original.content.header_rows
                for row in item.content.rows:
                    for cell in row.cells:
                        assert cell.text.strip() in chunk.source.text
                        for child in cell.children:
                            if isinstance(child.content, str):
                                assert child.content.strip() in chunk.source.text
                assert item.metadata["row_ranges"] == next(
                    op["row_ranges"] for op in chunk.metadata["operations"] if op["unit_id"] == original.id
                )


@pytest.mark.parametrize(("counter", "maximum"), [(len, 9000), (count_tokens, 5000)])
@pytest.mark.parametrize("mode", ["include", "partial", "failure"])
def test_budgeted_requests_preserve_original_rows_and_spans(counter, maximum, mode):
    table = _table()
    before = table.to_dict()
    gateway = Gateway(partial=mode == "partial", fail_second=mode == "failure")
    chunks = EvidenceUnitAgenticChunker(
        llm_gateway=gateway, max_plan_input_tokens=maximum, plan_token_counter=counter,
        max_concurrency=1, target_tokens_per_chunk=800, max_tokens_per_chunk=2048,
        enrich_final_chunks=False,
    ).chunk([table])
    _assert_budget(gateway, maximum, counter)
    assert len(gateway.plans) > 1
    planned_rows = []
    for plan in gateway.plans:
        assert len(plan["units"]) == 1
        view = plan["units"][0]
        assert view["id"] == table.id
        assert view["table"]["row_ranges"]
        assert view["table"]["columns"][0]["id"] == "c1"
        assert view["table"]["header_rows"][0]["cells"][0]["text_preview"] == "Both columns"
        assert view["table"]["header_rows"][0]["cells"][0]["colspan"] == 2
        assert view["table"]["rows"][0]["cells"][0]["text_preview"] == "shared record"
        planned_rows.extend(row["index"] for row in view["table"]["rows"])
    assert planned_rows == [row.index for row in table.content.rows]
    _assert_coverage(chunks, table)
    assert table.to_dict() == before
    if mode == "failure":
        assert any(c.metadata.get("_fallback_reason") == "simulated planner failure" for c in chunks)


def test_custom_planner_receives_bounded_typed_views_with_original_ids():
    table = _table()
    windows = []

    def plan(window, cfg, max_units):
        windows.append(window)
        assert all(u.id == table.id and u.content.rows for u in window)
        assert all("plan_row_ranges" in u.metadata for u in window)
        return [{"operations": [{"unit_id": u.id, "action": "include"} for u in window]}]

    chunks = EvidenceUnitAgenticChunker(
        plan_fn=plan, max_plan_input_tokens=9000, plan_token_counter=len,
        max_concurrency=1, enrich_final_chunks=False,
    ).chunk([table])
    assert len(windows) > 1
    _assert_coverage(chunks, table)


def test_repeated_row_splitting_retains_original_span_provenance():
    table = _table()
    chunks = EvidenceUnitAgenticChunker(
        llm_gateway=Gateway(), max_plan_input_tokens=9000, plan_token_counter=len,
        target_tokens_per_chunk=150, max_tokens_per_chunk=300,
        enrich_final_chunks=False,
    ).chunk([table])
    contexts = [cell.metadata["rowspan_context"] for c in chunks for item in c.evidence.items
                for row in item.content.rows for cell in row.cells if "rowspan_context" in cell.metadata]
    assert len(contexts) > 10
    assert all(context == {"source_row_index": 1, "source_rowspan": 41} for context in contexts)
    _assert_coverage(chunks, table)


def test_indivisible_input_skips_planner_and_preserves_evidence():
    unit = _table(1)
    calls = []
    chunks = EvidenceUnitAgenticChunker(
        plan_fn=lambda *args: calls.append(args), max_plan_input_tokens=1,
        enrich_final_chunks=False,
    ).chunk([unit])
    assert not calls
    assert [item.content for c in chunks for item in c.evidence.items] == [unit.content]
    assert any(w["type"] == "agentic_plan_input_budget_exceeded"
               for c in chunks for w in c.metadata["_warnings"])


def test_budget_splits_unit_windows_in_source_order():
    units = [_text(str(index)) for index in range(5)]
    prompt = build_plan_prompt(units[:1], 10)
    maximum = len(json.dumps({
        "messages": [{"role": "system", "content": "Return a response matching the JSON schema."},
                     {"role": "user", "content": prompt}],
        "response_schema": ChunkPlanResponse.model_json_schema(),
    }, ensure_ascii=False, separators=(",", ":"))) + 50
    gateway = Gateway()
    chunks = EvidenceUnitAgenticChunker(
        llm_gateway=gateway, window_size=40, max_concurrency=1,
        max_plan_input_tokens=maximum, plan_token_counter=len, enrich_final_chunks=False,
    ).chunk(units)
    assert [[unit["id"] for unit in plan["units"]] for plan in gateway.plans] == [[u.id] for u in units]
    assert [item.content for c in chunks for item in c.evidence.items] == [u.content for u in units]
    _assert_budget(gateway, maximum, len)


def test_only_an_oversized_row_falls_back_while_neighboring_rows_are_planned():
    table = _table(3)
    table.content.columns = [{"id": f"c{i}", "text": f"Column {i}"} for i in range(1, 121)]
    table.content.rows[1].cells = [{"column_id": f"c{i}", "text": "wide row content " * 10}
                                   for i in range(2, 121)]
    gateway = Gateway()
    chunks = EvidenceUnitAgenticChunker(
        llm_gateway=gateway, max_plan_input_tokens=20000, plan_token_counter=len,
        target_tokens_per_chunk=25000, max_tokens_per_chunk=50000,
        max_concurrency=1, enrich_final_chunks=False,
    ).chunk([table])
    assert [row["index"] for plan in gateway.plans for unit in plan["units"]
            for row in unit["table"]["rows"]] == [1, 3]
    assert any(w["type"] == "agentic_plan_input_budget_exceeded"
               for c in chunks for w in c.metadata.get("_warnings", []))
    _assert_coverage(chunks, table)
    _assert_budget(gateway, 20000, len)


def test_budget_counts_system_message_and_schema_not_only_prompt():
    unit = _text("single")
    maximum = len(build_plan_prompt([unit], 10))
    gateway = Gateway()
    chunks = EvidenceUnitAgenticChunker(
        llm_gateway=gateway, max_plan_input_tokens=maximum, plan_token_counter=len,
        enrich_final_chunks=False,
    ).chunk([unit])
    assert not gateway.requests
    assert chunks[0].evidence.items[0].content == unit.content
    assert "max_plan_input_tokens" in chunks[0].metadata["_fallback_reason"]


def test_boundary_request_over_budget_is_kept_without_llm_call():
    gateway = Gateway()

    def counter(text):
        return 100_001 if "window boundary merge planner" in text else len(text)

    chunks = EvidenceUnitAgenticChunker(
        llm_gateway=gateway, window_size=1, max_plan_input_tokens=100_000,
        plan_token_counter=counter, enrich_final_chunks=False,
    ).chunk([_text("a"), _text("b")])
    assert len(gateway.requests) == 2
    assert len(chunks) == 2
    assert any("max_plan_input_tokens" in w.get("reason", "")
               for chunk in chunks for w in chunk.metadata.get("_warnings", []))


def test_unset_budget_keeps_original_windows_and_does_not_count():
    units = [_text(str(index)) for index in range(5)]
    windows = []

    def plan(window, *_):
        windows.append(window)
        return [{"operations": [{"unit_id": u.id, "action": "include"} for u in window]}]

    def unused_counter(_):
        raise AssertionError("unset budget must not change planning")

    options = dict(plan_fn=plan, window_size=2, max_concurrency=1, enrich_final_chunks=False)
    expected = EvidenceUnitAgenticChunker(**options).chunk(units)
    windows.clear()
    actual = EvidenceUnitAgenticChunker(**options, max_plan_input_tokens=None,
                                      plan_token_counter=unused_counter).chunk(units)
    assert [[u.id for u in w] for w in windows] == [["0", "1"], ["2", "3"], ["4"]]
    assert all(u is original for u, original in zip([u for w in windows for u in w], units))
    assert [c.to_dict() for c in actual] == [c.to_dict() for c in expected]


@pytest.mark.parametrize("maximum", [0, -1, True, "100"])
def test_invalid_budget_is_rejected(maximum):
    with pytest.raises(ValueError, match="positive integer"):
        EvidenceUnitAgenticChunker(max_plan_input_tokens=maximum)


@pytest.mark.parametrize("value", [-1, None, True])
def test_invalid_counter_is_rejected(value):
    with pytest.raises(ValueError, match="nonnegative integer"):
        EvidenceUnitAgenticChunker(max_plan_input_tokens=100, plan_token_counter=lambda _: value).chunk([_text("a")])
