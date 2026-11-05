"""Issue #80: preserve v0.7.9 output without quadratic table traversal."""
from __future__ import annotations

import hashlib
import json

import pytest

from rag_document_parser import EvidenceUnit, EvidenceUnitAgenticChunker, SourceEvidence
from rag_document_parser.chunk import tokens
from rag_document_parser.chunk.agentic import _improve_chunk_boundaries
from rag_document_parser.chunk.materialization import ChunkPlanMaterializer, TableRowIndex
from rag_document_parser.models import TableRow


def _unit(size=18):
    rows = []
    for index in range(1, size + 1):
        # Leave a gap in source indexes and cross some chunk boundaries with
        # rowspans. Nested content and >120-character values must survive.
        source_index = index if index < 10 else index + 1
        cells = [{"column_id": "c2", "text": f"detail {index}: " + "원문 보존. " * 24}]
        if index % 3 == 1:
            cells.insert(0, {"column_id": "c1", "text": f"record {index}", "rowspan": 3})
        if index % 4 == 0:
            cells[-1]["children"] = [{
                "type": "table", "format": "structured_table",
                "content": {"caption": "nested context", "columns": [{"id": "c1", "text": "Code"}],
                            "rows": [{"index": 1, "cells": [{"column_id": "c1", "text": f"Z{index}.9"}]}]},
            }]
        rows.append({"index": source_index, "cells": cells, "source_row": source_index + 5})
    return EvidenceUnit(
        id="table", type="table", format="structured_table",
        content={
            "caption": "Full table", "columns": [{"id": "c1", "text": "Record"}, {"id": "c2", "text": "Details"}],
            "header_rows": [{"index": 1, "cells": [{"column_id": "c1", "text": "Full header", "colspan": 2}]}],
            "rows": rows, "compact": {"preview": "not a substitute for complete cell text"},
        },
        source=SourceEvidence(kind="table", text="Table source"),
    )


def _fallback_plan(*_):
    raise RuntimeError("fixture plan failure")


def _chunks(maximum):
    prefix = EvidenceUnit(id="title", type="text", format="plain", content="Table title",
                          source=SourceEvidence(kind="text", text="Table title"))
    return EvidenceUnitAgenticChunker(
        plan_fn=_fallback_plan, target_tokens_per_chunk=maximum // 2,
        max_tokens_per_chunk=maximum, enrich_final_chunks=False,
    ).chunk([prefix, _unit()])


@pytest.mark.parametrize(("maximum", "expected"), [
    (200, "4b75dbaf5ba15deeb7c76eecbe0f4c31787d74a97537979322252058f8bebd19"),
    (800, "ed4c87acccdc5f77dc03ccff3f557dc633c87c33291c074c303dcce9579d64b0"),
    (2048, "bf5316b63c0549568a40b8016143ed980bd35562be16dedab7d6728c8cf4cfc8"),
])
def test_row_splitting_matches_pre_optimization_output(monkeypatch, maximum, expected):
    # Captured from 0d034a9 before the optimization. Fix the tokenizer so the
    # golden includes stable boundaries, text, evidence, provenance and warnings.
    monkeypatch.setattr(tokens, "_token_encoder", lambda: None)
    output = [chunk.to_dict() for chunk in _chunks(maximum)]
    serialized = json.dumps(output, ensure_ascii=False, sort_keys=True).encode()
    assert hashlib.sha256(serialized).hexdigest() == expected


def test_table_split_work_grows_with_rows_not_table_area(monkeypatch):
    monkeypatch.setattr(tokens, "_token_encoder", lambda: None)
    original_to_dict = TableRow.to_dict
    work = {"row_visits": 0, "serialized_rows": 0}

    class CountedRows(list):
        def __iter__(self):
            for row in super().__iter__():
                work["row_visits"] += 1
                yield row

    def counted_to_dict(row):
        work["serialized_rows"] += 1
        return original_to_dict(row)

    monkeypatch.setattr(TableRow, "to_dict", counted_to_dict)

    def measure(size):
        unit = _unit(size)
        # Some backends retain full-table previews; copying the envelope must
        # not recursively serialize those unselected rows either.
        unit.content.compact = {"cached_rows": unit.content.rows}
        unit.content = unit.content.model_copy(update={"rows": CountedRows(unit.content.rows)})
        fallback = ChunkPlanMaterializer(10).fallback([unit], "fixture plan failure")
        before = unit.to_dict()
        work.update(row_visits=0, serialized_rows=0)
        chunks = _improve_chunk_boundaries(fallback, [unit], 10, 400, 800)
        measured = dict(work)
        assert unit.to_dict() == before
        assert [row.index for c in chunks for item in c.evidence.items for row in item.content.rows] == [
            row.index for row in unit.content.rows
        ]
        return measured

    small, large = measure(60), measure(120)
    for operation in work:
        assert small[operation] > 0
        assert large[operation] <= 2.5 * small[operation]


def test_reused_row_index_handles_sparse_disjoint_ranges_and_clips_spans():
    unit = _unit()
    index = TableRowIndex(unit.content)
    before = unit.to_dict()
    subset = index.subset([(2, 3), (5, 6), (9, 11)])
    assert [row["index"] for row in subset["rows"]] == [2, 3, 5, 6, 9, 11]
    for row, origin, span in zip(subset["rows"][::2], [1, 4, 7], [2, 2, 1]):
        cell = row["cells"][0]
        assert cell["text"] == f"record {origin}"
        assert cell["rowspan"] == span
        assert cell["metadata"]["rowspan_context"]["source_row_index"] == origin
    other = index.subset([(1, 2)])
    assert other["rows"][0]["cells"][0]["rowspan"] == 2
    assert unit.to_dict() == before
