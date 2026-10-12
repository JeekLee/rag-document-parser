from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_document_parser import EvidenceUnit, PdfBackend, SourceEvidence
from rag_document_parser.chunk import EvidenceUnitAgenticChunker
from rag_document_parser.chunk.enrichment import _chunk_payload
from rag_document_parser.chunk.materialization import ChunkPlanMaterializer, table_source_text
from rag_document_parser.chunk.planning_prompt import build_plan_prompt
from rag_document_parser.chunk.tokens import count_tokens


@pytest.fixture(scope="module")
def qa_units():
    path = Path(__file__).parent / "fixtures/corpus/pdf/cesarean-copay-qa.pdf"
    return PdfBackend().parse(path.read_bytes(), ".pdf").units


def _captured_qa_plan():
    # Primary operations and context links from the v0.7.7 audit's five-item
    # fallback-replay.json, reproduced without depending on local /tmp files.
    groups = [
        ([f"b{i}" for i in range(1, 9)], None, []),
        (["b9"], [1, 1], [f"b{i}" for i in range(1, 10)]),
        (["b9"], [2, 2], [f"b{i}" for i in range(1, 10)]),
        (["b10", "b11"], [1, 1], [f"b{i}" for i in range(1, 12)]),
        (["b11"], [2, 9], [f"b{i}" for i in range(1, 12)]),
    ]
    return [
        {
            "unit_ids": ids, "context_unit_ids": context, "title": "QA policy",
            "operations": [
                {"unit_id": unit_id, "action": "include_rows", "row_ranges": [rows]}
                if rows and unit_id in {"b9", "b11"}
                else {"unit_id": unit_id, "action": "include"}
                for unit_id in ids
            ],
        }
        for ids, rows, context in groups
    ]


def _coverage(items):
    return [
        (item.source_unit_ids[0], row["index"])
        for item in items if item.format == "structured_table"
        for row in item.content["rows"]
    ]


@pytest.fixture(scope="module")
def criteria_units():
    path = Path(__file__).parent / "fixtures/corpus/pdf/benefit-criteria-2024-278.pdf"
    return PdfBackend().parse(path.read_bytes(), ".pdf").units


def _captured_smoke_plan(name):
    # Primary operations from the actual planning-smoke calls, kept independent
    # of the developer's local capture files and with no model requests.
    if name == "criteria":
        groups = [range(1, 9), range(9, 15)]
        rows = {"b8": [1, 1], "b10": [1, 1], "b14": [1, 4]}
    else:
        groups = [range(1, 10), range(10, 12)]
        rows = {"b9": [1, 2], "b11": [1, 9]}
    return [
        {
            "unit_ids": [f"b{i}" for i in group],
            "context_unit_ids": [],
            "operations": [
                {"unit_id": f"b{i}",
                 "action": "include_rows" if name == "criteria" and f"b{i}" in rows else "include",
                 "row_ranges": [rows[f"b{i}"]] if f"b{i}" in rows else None}
                for i in group
            ],
        }
        for group in groups
    ]


def _assert_complete_coverage(chunks, units):
    items = [item for chunk in chunks for item in chunk.evidence.items]
    assert list(dict.fromkeys(item.source_unit_ids[0] for item in items)) == [unit.id for unit in units]
    assert [item.source_unit_ids[0] for item in items if item.type == "text"] == [
        unit.id for unit in units if unit.type == "text"
    ]
    assert _coverage(items) == [
        (unit.id, row["index"]) for unit in units if unit.type == "table"
        for row in unit.content["rows"]
    ]
    assert all("_fallback_reason" not in chunk.metadata for chunk in chunks)
    assert all(count_tokens(chunk.source.text) <= 2048 for chunk in chunks)


@pytest.mark.parametrize("separate_plan_rows", [False, True])
def test_captured_criteria_smoke_keeps_heading_bands_and_pending_evidence_within_budget(criteria_units, separate_plan_rows):
    before = [unit.to_dict() for unit in criteria_units]
    plan = _captured_smoke_plan("criteria")
    if separate_plan_rows:
        # Also cover a planner attaching a band to an oversized previous chunk
        # and placing its content row in the next chunk.
        plan[-1]["operations"][-1]["row_ranges"] = [[1, 1]]
        plan.extend({"operations": [{"unit_id": "b14", "action": "include_rows", "row_ranges": [[i, i]]}]}
                    for i in range(2, 5))
    chunks = EvidenceUnitAgenticChunker(
        plan_fn=lambda *_: plan, enrich_final_chunks=False,
    ).chunk(criteria_units)
    _assert_complete_coverage(chunks, criteria_units)
    assert [_coverage(chunk.evidence.items) for chunk in chunks
            if "b14" in chunk.metadata["source_unit_ids"]] == [
        [("b14", 1), ("b14", 2)], [("b14", 3), ("b14", 4)],
    ]
    first_comparison = next(chunk for chunk in chunks if ("b14", 1) in _coverage(chunk.evidence.items))
    assert first_comparison.metadata["source_unit_ids"] == ["b12", "b13", "b14"]
    disease_chunk = next(chunk for chunk in chunks if "b10" in chunk.metadata["source_unit_ids"])
    assert "b9" in disease_chunk.metadata["source_unit_ids"]
    nested = criteria_units[9].content.rows[0].cells[2].children[0].content
    codes = [cell["text"].strip() for row in [*nested["header_rows"][1:], *nested["rows"]]
             for cell in row["cells"] if cell["text"].strip()]
    assert len(codes) == len(set(codes)) == 84
    payload = json.dumps(_chunk_payload(disease_chunk)["evidence"], ensure_ascii=False)
    assert all(code in disease_chunk.source.text and code in payload for code in codes)
    assert [unit.to_dict() for unit in criteria_units] == before


def test_captured_qa_smoke_preserves_rowspan_groups(qa_units):
    chunks = EvidenceUnitAgenticChunker(
        plan_fn=lambda *_: _captured_smoke_plan("qa"), enrich_final_chunks=False,
    ).chunk(qa_units)
    _assert_complete_coverage(chunks, qa_units)
    for pair in [[("b9", 1), ("b9", 2)], [("b11", 5), ("b11", 6)]]:
        assert any(set(pair) <= set(_coverage(chunk.evidence.items)) for chunk in chunks)


@pytest.mark.parametrize("fallback", [False, True])
def test_captured_qa_plan_preserves_ordered_coverage_and_coherent_boundaries(qa_units, fallback):
    plan = _captured_qa_plan()
    before = [unit.to_dict() for unit in qa_units]
    materialized = ChunkPlanMaterializer(10).materialize(qa_units, plan)
    assert len(materialized) == 5
    assert all("_fallback_reason" not in chunk.metadata for chunk in materialized)
    assert [chunk.metadata["operations"] for chunk in materialized] == [part["operations"] for part in plan]
    repaired = [warning["unit_id"] for chunk in materialized
                for warning in chunk.metadata.get("_warnings", [])]
    assert repaired == ["b9", "b10", "b11"]

    chunks = EvidenceUnitAgenticChunker(
        plan_fn=lambda *_: None if fallback else plan, enrich_final_chunks=False,
    ).chunk(qa_units)
    items = [item for chunk in chunks for item in chunk.evidence.items]
    ids = list(dict.fromkeys(unit_id for item in items for unit_id in item.source_unit_ids))
    assert ids == [f"b{i}" for i in range(1, 12)]
    assert _coverage(items) == [("b9", i) for i in range(1, 3)] + [("b11", i) for i in range(1, 10)]
    assert [item.source_unit_ids[0] for item in items if item.type == "text"] == [
        unit.id for unit in qa_units if unit.type == "text"
    ]
    assert all(count_tokens(chunk.source.text) <= 2048 for chunk in chunks)
    # Each rowspan-connected answer stays together; independent questions split
    # near the target even when the original planned group was below the max.
    assert any(_coverage(chunk.evidence.items) == [("b9", 1), ("b9", 2)] for chunk in chunks)
    assert any({("b11", 5), ("b11", 6)} <= set(_coverage(chunk.evidence.items)) for chunk in chunks)
    assert max(len([key for key in _coverage(chunk.evidence.items) if key[0] == "b11"])
               for chunk in chunks) < 8
    text_only = [chunk for chunk in chunks if all(item.type == "text" for item in chunk.evidence.items)]
    assert len(text_only) == 1
    assert text_only[0].metadata["source_unit_ids"] == [f"b{i}" for i in range(1, 7)]
    if not fallback:
        assert all("_fallback_reason" not in chunk.metadata for chunk in chunks)
        assert {warning["unit_id"] for chunk in chunks for warning in chunk.metadata.get("_warnings", [])
                if warning["type"] == "agentic_context_reference_dropped"} == {"b9", "b10", "b11"}
    assert [unit.to_dict() for unit in qa_units] == before


def test_planning_outline_exposes_later_rows_and_rowspans(qa_units):
    prompt = build_plan_prompt(qa_units, 10, 700, 1800)
    assert '"target_tokens_per_chunk": 700' in prompt
    assert '"max_tokens_per_chunk": 1800' in prompt
    assert '"index": 9,' in prompt
    assert '"rowspan": 2' in prompt
    last_question = qa_units[-1].content["rows"][-1]["cells"][1]["text"]
    assert last_question in prompt


def _text(unit_id, text, section=()):
    return EvidenceUnit(
        id=unit_id, type="text", format="plain", content=text,
        source=SourceEvidence(kind="text", text=text),
        metadata={"common": {"section_path": list(section)}},
    )


@pytest.mark.parametrize("shape", ["groups", "full_width", "partial", "different_labels", "nested"])
@pytest.mark.parametrize("separate_plan_rows", [False, True])
def test_only_structural_heading_bands_join_following_rows(shape, separate_plan_rows):
    cells = [{"column_id": "c1", "text": "Section A", "colspan": 2},
             {"column_id": "c3", "text": "Section A", "colspan": 2}]
    if shape == "full_width":
        cells = [{"column_id": "c1", "text": "Section A", "colspan": 4}]
    elif shape == "partial":
        cells = cells[:1]
    elif shape == "different_labels":
        cells[1]["text"] = "A data value"
    elif shape == "nested":
        cells[0]["children"] = [{"type": "text", "format": "plain", "content": "Nested evidence"}]
    table = {
        "columns": [{"id": f"c{i}", "text": f"Column {i}"} for i in range(1, 5)],
        "rows": [
            {"index": 1, "cells": cells},
            {"index": 2, "cells": [{"column_id": f"c{i}", "text": "content " * 20} for i in range(1, 5)]},
        ],
    }
    unit = EvidenceUnit(id="table", type="table", format="structured_table", content=table,
                        source=SourceEvidence(kind="table", text=table_source_text(table)))
    ranges = [[1, 1], [2, 2]] if separate_plan_rows else [[1, 2]]
    chunks = EvidenceUnitAgenticChunker(
        plan_fn=lambda *_: [{"operations": [{"unit_id": "table", "action": "include_rows", "row_ranges": [rows]}]}
                           for rows in ranges],
        target_tokens_per_chunk=80, max_tokens_per_chunk=500, enrich_final_chunks=False,
    ).chunk([unit])
    expected = [[("table", 1), ("table", 2)]] if shape in {"groups", "full_width"} else [[("table", 1)], [("table", 2)]]
    assert [_coverage(chunk.evidence.items) for chunk in chunks] == expected


@pytest.mark.parametrize("row_count", [1, 2])
def test_pending_primary_evidence_splits_before_fitting_table_with_title(row_count):
    table = {
        "columns": [{"id": "c1", "text": "Answer"}],
        "rows": [{"index": i, "cells": [{"column_id": "c1", "text": "answer " * 100}]}
                 for i in range(1, row_count + 1)],
    }
    unit = EvidenceUnit(id="table", type="table", format="structured_table", content=table,
                        source=SourceEvidence(kind="table", text=table_source_text(table)))
    units = [_text("previous", "Previous facts. " * 60), _text("title", "Comparison table"), unit]
    chunks = EvidenceUnitAgenticChunker(
        plan_fn=lambda *_: [{"operations": [{"unit_id": u.id, "action": "include"} for u in units]}],
        target_tokens_per_chunk=100, max_tokens_per_chunk=300, enrich_final_chunks=False,
    ).chunk(units)
    assert chunks[0].metadata["source_unit_ids"] == ["previous"]
    assert chunks[1].metadata["source_unit_ids"] == ["title", "table"]
    assert all(count_tokens(chunk.source.text) <= 300 for chunk in chunks)
    assert _coverage([item for chunk in chunks for item in chunk.evidence.items]) == [
        ("table", i) for i in range(1, row_count + 1)
    ]
    assert chunks[0].source.text == units[0].source.text


@pytest.mark.parametrize("texts,sections", [
    (["1. Apples", "2. Cars", "3. Rivers"], [("Apples",), ("Cars",), ("Rivers",)]),
    (["Apples", "Cars", "Rivers"], [(), (), ()]),
    (["1. Topic", "사례 2 unrelated", "[별지 제3호 서식]"], [(), (), ()]),
])
def test_short_fallback_fragments_respect_structural_boundaries(texts, sections):
    units = [_text(f"u{i}", text, section) for i, (text, section) in enumerate(zip(texts, sections))]
    chunks = EvidenceUnitAgenticChunker(enrich_final_chunks=False).chunk(units)
    assert [chunk.metadata["source_unit_ids"] for chunk in chunks] == [[unit.id] for unit in units]


def test_context_warning_survives_moving_all_heading_evidence():
    units = [_text("heading", "1. Topic"), _text("body", "The body answers a question.")]
    plan = [{"operations": [{"unit_id": unit.id, "action": "include"}],
             "context_unit_ids": ["unknown"] if i == 0 else []} for i, unit in enumerate(units)]
    chunks = EvidenceUnitAgenticChunker(plan_fn=lambda *_: plan, enrich_final_chunks=False).chunk(units)
    assert len(chunks) == 1
    assert chunks[0].metadata["source_unit_ids"] == ["heading", "body"]
    assert any(warning["type"] == "agentic_context_reference_dropped"
               for warning in chunks[0].metadata["_warnings"])


def test_target_splitting_preserves_long_nested_source_and_warns_for_atomic_rows():
    nested_marker = "nested evidence at the end"
    long_text = "evidence " * 200 + "last cell sentence"
    table = {
        "columns": [{"id": "c1", "text": "Answer"}],
        "rows": [
            {"index": 1, "cells": [{"column_id": "c1", "text": long_text, "children": [
                {"type": "text", "format": "plain", "content": nested_marker},
            ]}]},
            {"index": 2, "cells": [{"column_id": "c1", "text": "short answer"}]},
        ],
    }
    unit = EvidenceUnit(id="table", type="table", format="structured_table", content=table,
                        source=SourceEvidence(kind="table", text=table_source_text(table)))
    chunks = EvidenceUnitAgenticChunker(
        plan_fn=lambda *_: [{"operations": [{"unit_id": "table", "action": "include"}]}],
        target_tokens_per_chunk=100, max_tokens_per_chunk=200, enrich_final_chunks=False,
    ).chunk([unit])
    assert len(chunks) == 2
    assert long_text in chunks[0].source.text
    assert chunks[0].source.text.count(nested_marker) == 1
    assert nested_marker not in chunks[1].source.text
    assert _coverage([item for chunk in chunks for item in chunk.evidence.items]) == [("table", 1), ("table", 2)]
    assert any(warning["type"] == "agentic_chunk_exceeds_max_tokens"
               for warning in chunks[0].metadata["_warnings"])


def test_table_below_max_splits_at_target_without_breaking_rowspan():
    table = {
        "columns": [{"id": "c1", "text": "Question"}, {"id": "c2", "text": "Answer"}],
        "rows": [
            {"index": index, "cells": [
                *([{"column_id": "c1", "text": "Shared question", "rowspan": 2}] if index == 1 else []),
                {"column_id": "c2", "text": f"Answer {index}: " + "details " * 30},
            ]}
            for index in range(1, 5)
        ],
    }
    source = table_source_text(table)
    target, maximum = count_tokens(source) // 2, count_tokens(source) * 2
    unit = EvidenceUnit(id="table", type="table", format="structured_table", content=table,
                        source=SourceEvidence(kind="table", text=source))
    assert target < count_tokens(source) < maximum
    chunks = EvidenceUnitAgenticChunker(
        plan_fn=lambda *_: [{"operations": [{"unit_id": "table", "action": "include"}]}],
        target_tokens_per_chunk=target, max_tokens_per_chunk=maximum, enrich_final_chunks=False,
    ).chunk([unit])
    assert len(chunks) > 1
    assert _coverage(chunks[0].evidence.items) == [("table", 1), ("table", 2)]
    assert _coverage([item for chunk in chunks for item in chunk.evidence.items]) == [
        ("table", index) for index in range(1, 5)
    ]
    assert all(count_tokens(chunk.source.text) <= maximum for chunk in chunks)


@pytest.mark.parametrize("primary_ids", [["u2", "u1"], ["unknown"], ["u1", "u1"]])
def test_invalid_primary_plan_falls_back_to_complete_ordered_evidence(primary_ids):
    units = [_text("u1", "First evidence"), _text("u2", "Second evidence")]
    plan = [{"operations": [{"unit_id": unit_id, "action": "include"} for unit_id in primary_ids],
             "context_unit_ids": ["new-context"]}]
    chunks = EvidenceUnitAgenticChunker(plan_fn=lambda *_: plan, enrich_final_chunks=False).chunk(units)
    assert [chunk.source.text for chunk in chunks] == [unit.source.text for unit in units]
    assert all(chunk.metadata["_fallback_reason"] for chunk in chunks)
    assert all(chunk.metadata["_rejected_plan"] == plan for chunk in chunks)
