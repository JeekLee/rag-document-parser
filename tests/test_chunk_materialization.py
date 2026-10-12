from __future__ import annotations

import pytest

from rag_document_parser import EvidenceUnit, SourceEvidence
from rag_document_parser.chunk.materialization import ChunkPlanMaterializer


def _text_unit(unit_id: str, text: str) -> EvidenceUnit:
    return EvidenceUnit(
        id=unit_id,
        type="text",
        format="plain",
        source=SourceEvidence(kind="text", text=text),
        content=text,
    )


def test_chunk_plan_materializer_owns_validation_and_omission_repair():
    first = _text_unit("u1", "first")
    second = _text_unit("u2", "second")
    materializer = ChunkPlanMaterializer(max_units_per_chunk=1)

    with pytest.raises(ValueError, match="duplicate unit id"):
        materializer.materialize([first, first], [])

    chunks = materializer.materialize(
        [first, second],
        [{"operations": [{"unit_id": "u1", "action": "include"}]}],
    )

    assert [chunk.metadata["source_unit_ids"] for chunk in chunks] == [["u1"], ["u2"]]
    assert chunks[1].metadata["_fallback_reason"] == "chunk plan omitted units: u2"


@pytest.mark.parametrize("fields", [{}, {"summary": "Custom summary"},
                                    {"keywords": ["Custom"]}, {"questions": []}])
def test_only_plans_without_legacy_text_fields_need_enrichment(fields):
    unit = _text_unit("u1", "Source evidence")
    chunk = ChunkPlanMaterializer(10).materialize(
        [unit], [{"operations": [{"unit_id": "u1", "action": "include"}], **fields}],
    )[0]
    assert chunk.metadata.get("_needs_enrichment", False) is (not fields)
    if "summary" in fields:
        assert chunk.summary == fields["summary"]
    if "keywords" in fields:
        assert chunk.keywords == fields["keywords"]


def _table_unit(indexes=(1, 2, 3)):
    return EvidenceUnit(
        id="table", type="table", format="structured_table",
        source=SourceEvidence(kind="table", text="table"),
        content={
            "columns": [{"id": "c1", "text": "Value"}],
            "rows": [{"index": index, "cells": [{"column_id": "c1", "text": str(index)}]}
                     for index in indexes],
        },
    )


def _rows(*ranges):
    return {"unit_id": "table", "action": "include_rows", "row_ranges": list(ranges)}


def test_context_recovery_preserves_primary_operations_and_explains_each_error():
    units = [_text_unit(f"u{i}", str(i)) for i in range(1, 4)]
    plan = [
        {"operations": [{"unit_id": "u1", "action": "include"}]},
        {"operations": [{"unit_id": "u2", "action": "include"}],
         "context_unit_ids": ["u1", "u1", "u2", "u3", "unknown", 42]},
        {"operations": [{"unit_id": "u3", "action": "include"}]},
    ]
    chunks = ChunkPlanMaterializer(10).materialize(units, plan)
    assert [chunk.metadata["operations"] for chunk in chunks] == [part["operations"] for part in plan]
    assert chunks[1].metadata["context_unit_ids"] == ["u1"]
    warnings = chunks[1].metadata["_warnings"]
    assert [warning["unit_id"] for warning in warnings] == ["u2", "u3", "unknown", 42]
    assert [warning["reason"].split(":")[0].split(";")[0] for warning in warnings] == [
        "self reference", "new context unit", "unknown context unit id", "unknown context unit id",
    ]
    assert all("_fallback_reason" not in chunk.metadata for chunk in chunks)


@pytest.mark.parametrize("operations,match", [
    ([{"unit_id": "missing", "action": "include"}], "unknown unit id"),
    ([{"unit_id": "u2", "action": "include"}, {"unit_id": "u1", "action": "include"}], "source order"),
    ([{"unit_id": "u1", "action": "include"}] * 2, "duplicate unit id"),
])
def test_context_recovery_does_not_hide_primary_integrity_errors(operations, match):
    with pytest.raises(ValueError, match=match):
        ChunkPlanMaterializer(10).materialize(
            [_text_unit("u1", "one"), _text_unit("u2", "two")],
            [{"operations": operations, "context_unit_ids": ["unknown"]}],
        )


@pytest.mark.parametrize("ranges,match", [
    (([3, 3], [1, 1]), "ordered"),
    (([1, 3], [3, 3]), "overlap"),
    (([2, 3],), "actual table rows"),
])
def test_selected_row_boundaries_must_be_real_ordered_and_disjoint(ranges, match):
    with pytest.raises(ValueError, match=match):
        ChunkPlanMaterializer(10).materialize([_table_unit((1, 3))], [{"operations": [_rows(*ranges)]}])


def test_reversed_row_chunks_are_rejected_and_disjoint_omissions_repaired_in_order():
    materializer = ChunkPlanMaterializer(10)
    unit = _table_unit()
    with pytest.raises(ValueError, match="source order"):
        materializer.materialize([unit], [
            {"operations": [_rows([3, 3])]}, {"operations": [_rows([1, 2])]},
        ])
    chunks = materializer.materialize([unit], [{"operations": [_rows([2, 2])], "title": "kept"}])
    assert [row["index"] for chunk in chunks for item in chunk.evidence.items
            for row in item.content["rows"]] == [1, 2, 3]
    assert chunks[1].metadata["title"] == "kept"
    assert "_fallback_reason" not in chunks[1].metadata
