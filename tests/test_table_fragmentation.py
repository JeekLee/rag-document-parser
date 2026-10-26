from copy import deepcopy
from pathlib import Path

import pytest

from rag_document_parser import EvidenceUnit, EvidenceUnitAgenticChunker, Hwp5Backend, RagChunk, SourceEvidence
from rag_document_parser.chunk.enrichment import _chunk_payload
from rag_document_parser.chunk.evidence_projection import table_source_text
from rag_document_parser.chunk.tokens import count_tokens
from rag_document_parser.models import StructuredTableContent


def _chunk(table, maximum=400):
    unit = EvidenceUnit(id="t", type="table", format="structured_table", content=table,
                        source=SourceEvidence(kind="table", text=table_source_text(table)))
    before = unit.to_dict()
    chunks = EvidenceUnitAgenticChunker(
        plan_fn=lambda *_: [{"operations": [{"unit_id": "t", "action": "include"}]}],
        target_tokens_per_chunk=maximum // 2, max_tokens_per_chunk=maximum,
        enrich_final_chunks=False,
    ).chunk([unit])
    assert unit.to_dict() == before
    for chunk in chunks:
        assert RagChunk.model_validate(chunk.to_dict()).to_dict() == chunk.to_dict()
    return chunks


def _assert_cell_coverage(chunks, original):
    """Reassemble primary text and children; repeated keys/spans are context."""
    collected = {}
    order = []
    for chunk in chunks:
        for item in chunk.evidence.items:
            if item.type != "table":
                continue
            uid = item.source_unit_ids[0]
            assert item.content["columns"] == original[uid]["columns"]
            assert item.content.get("header_rows", []) == original[uid].get("header_rows", [])
            for row in item.content["rows"]:
                for cell in row["cells"]:
                    if cell.get("metadata", {}).get("rowspan_context"):
                        continue
                    marker = cell.get("metadata", {}).get("cell_fragment")
                    if marker and marker["role"] != "content":
                        continue
                    key = (uid, row["index"], cell["column_id"])
                    if not order or order[-1] != key:
                        order.append(key)
                    text, children = collected.setdefault(key, ["", []])
                    if marker:
                        assert marker["text_range"] == [len(text), len(text) + len(cell["text"])]
                        assert marker["children_range"] == [len(children), len(children) + len(cell["children"])]
                    collected[key] = [text + cell["text"], children + list(cell["children"])]
    expected = {
        (uid, row["index"], cell["column_id"]): [cell["text"], list(cell.get("children", []))]
        for uid, table in original.items() for row in table["rows"] for cell in row["cells"]
    }
    assert collected == expected
    # Context keys may be emitted before long cells in the same partial row;
    # primary table rows themselves must remain in source order.
    row_order = list(dict.fromkeys((uid, row) for uid, row, _ in order))
    assert row_order == list(dict.fromkeys((uid, row) for uid, row, _ in expected))


@pytest.mark.parametrize("second_long_cell", [False, True])
def test_row_fragments_preserve_keys_offsets_children_and_partial_row_contract(second_long_cell):
    table = {
        "caption": "Policy comparison", "columns": [
            {"id": "c1", "text": "ID"}, {"id": "c2", "text": "Before"}, {"id": "c3", "text": "After"},
        ],
        "header_rows": [{"index": 1, "cells": [{"column_id": "c1", "text": "Policies", "colspan": 3}]}],
        "rows": [{"index": 7, "cells": [
            {"column_id": "c1", "text": "Policy-A"},
            {"column_id": "c2", "text": ("옛 기준입니다.  \n다음 문단입니다.\n\n" * 75) + "끝.  ",
             "children": [{"type": "text", "format": "plain", "content": "child follows before text"}]},
            {"column_id": "c3", "text": "New condition. " * 200 if second_long_cell else "Unchanged"},
        ]}],
        "compact": {"cached_text": "must not leak the complete table into fragments"},
    }
    chunks = _chunk(table)
    original = StructuredTableContent.model_validate(table)
    _assert_cell_coverage(chunks, {"t": original})
    assert len(chunks) > 1
    for index, chunk in enumerate(chunks, start=1):
        item = chunk.evidence.items[0]
        assert count_tokens(chunk.source.text) <= 400
        assert item.content.get("compact") is None
        assert item.metadata["table_fragment"] == {"row_index": 7, "part": index, "count": len(chunks)}
        assert chunk.metadata["operations"][0]["table_fragment"] == item.metadata["table_fragment"]
        assert _chunk_payload(chunk)["evidence"][0]["table"]["table_fragment"] == item.metadata["table_fragment"]
        key = item.content["rows"][0]["cells"][0]
        assert key["text"] == "Policy-A"
        assert key["metadata"]["cell_fragment"]["role"] == ("content" if index == 1 else "context")
        if second_long_cell:
            # Before and After are independent continuations, not invented aligned subrows.
            active = [c for c in item.content["rows"][0]["cells"][1:]
                      if c["metadata"]["cell_fragment"]["role"] == "content"]
            assert len(active) == 1
            assert "omitted in this row fragment" in chunk.source.text


@pytest.mark.parametrize("text", ["가" * 2000, "word.\n\n" * 500, "🙂é\u0301 " * 800])
def test_splitting_is_lossless_with_or_without_natural_boundaries(text):
    table = {"columns": [{"id": "c1", "text": "Value"}],
             "rows": [{"index": 1, "cells": [{"column_id": "c1", "text": text}]}]}
    chunks = _chunk(table, maximum=300)
    cells = [c.evidence.items[0].content["rows"][0]["cells"][0] for c in chunks]
    assert len(chunks) > 1
    assert "".join(c["text"] for c in cells) == text
    assert all(count_tokens(c.source.text) <= 300 for c in chunks)
    if text.startswith("word."):
        assert all(c["text"].endswith("\n") for c in cells[:-1])


@pytest.mark.parametrize("oversized", ["header", "child"])
def test_irreducible_content_is_preserved_and_warned(oversized):
    table = {"columns": [{"id": "c1", "text": "Value"}],
             "rows": [{"index": 1, "cells": [{"column_id": "c1", "text": "본문 " * 300}]}]}
    if oversized == "header":
        table["header_rows"] = [{"index": 1, "cells": [{"column_id": "c1", "text": "긴 헤더 " * 300}]}]
    else:
        table["rows"][0]["cells"][0]["children"] = [
            {"type": "text", "format": "plain", "content": "atomic child " * 500},
        ]
    chunks = _chunk(table, maximum=200)
    assert len(chunks) == 1
    assert chunks[0].evidence.items[0].content == StructuredTableContent.model_validate(table)
    assert "table_fragment" not in chunks[0].evidence.items[0].metadata
    assert any(w["type"] == "agentic_chunk_exceeds_max_tokens" for w in chunks[0].metadata["_warnings"])


@pytest.mark.parametrize("pack_size", [1, 10])
def test_original_issue75_document_fits_budget_and_preserves_every_cell(pack_size):
    pytest.importorskip("olefile")
    path = Path(__file__).parent / "fixtures/corpus/hwp/hira-spine-1a-5.hwp"
    doc = Hwp5Backend().parse(path.read_bytes(), ".hwp")
    before = deepcopy([u.to_dict() for u in doc.units])
    chunks = EvidenceUnitAgenticChunker(
        plan_fn=lambda units, *_: [
            {"operations": [{"unit_id": u.id, "action": "include"} for u in units[start:start + pack_size]]}
            for start in range(0, len(units), pack_size)
        ], enrich_final_chunks=False,
    ).chunk(doc.units)
    assert all(count_tokens(c.source.text) <= 2048 for c in chunks)
    assert not any(c.metadata.get("_fallback_reason") for c in chunks)
    assert not any(w["type"] == "agentic_chunk_exceeds_max_tokens"
                   for c in chunks for w in c.metadata.get("_warnings", []))
    assert [item.content for c in chunks for item in c.evidence.items if item.type == "text"] == [
        u.content for u in doc.units if u.type == "text"
    ]
    _assert_cell_coverage(chunks, {u.id: u.content for u in doc.units if u.type == "table"})
    assert [u.to_dict() for u in doc.units] == before
