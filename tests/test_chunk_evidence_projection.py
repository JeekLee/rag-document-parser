from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_document_parser import EvidenceUnit, HwpxBackend, PdfBackend, SourceEvidence
from rag_document_parser.chunk.enrichment import (
    _batch_enrichment_prompt,
    _chunk_payload,
    _enrichment_prompt,
    _table_payload,
)
from rag_document_parser.chunk.materialization import (
    ChunkPlanMaterializer,
    table_source_text,
    table_subset,
)


def _table(*texts: str) -> dict:
    return {
        "columns": [{"id": "c1", "text": "Value"}],
        "rows": [
            {"index": index, "cells": [{"column_id": "c1", "text": text}]}
            for index, text in enumerate(texts, 1)
        ],
    }


def _unit(table: dict) -> EvidenceUnit:
    return EvidenceUnit(
        id="b10",
        type="table",
        format="structured_table",
        source=SourceEvidence(kind="table", text="old projection without children"),
        content=table,
        metadata={"pdf": {"page": 3}},
    )


def _materialize(unit: EvidenceUnit, action: str):
    materializer = ChunkPlanMaterializer(max_units_per_chunk=1)
    if action == "fallback":
        return materializer.fallback([unit], "test")[0]
    operation = {"unit_id": unit.id, "action": action}
    if action == "include_rows":
        operation["row_ranges"] = [[1, 1]]
    chunks = materializer.materialize([unit], [{"unit_ids": [unit.id], "operations": [operation]}])
    return next(chunk for chunk in chunks if chunk.metadata["operations"] == [operation])


@pytest.mark.parametrize("typed", [False, True])
def test_recursive_projection_keeps_children_headers_and_provenance(typed):
    inner = _table("deep body")
    inner["caption"] = "deep caption"
    inner["header_rows"] = [
        {"index": 1, "cells": [{"column_id": "c1", "text": "deep header", "colspan": 2}]}
    ]
    middle = _table("")
    middle["rows"][0]["cells"][0]["children"] = [
        {"type": "table", "format": "structured_table", "content": inner},
        {"kind": "text", "format": "plain", "content": "child text"},
        {
            "type": "image", "format": "asset_ref",
            "content": {"asset_id": "image-1", "caption": "image caption"},
            "metadata": {"page": 3, "bbox": [1, 2, 3, 4]},
        },
        {
            "type": "diagram", "format": "structured_diagram",
            "content": {
                "caption": "diagram caption", "asset_id": "diagram-1",
                "nodes": [
                    {"id": "n1", "shape_type": "rect", "text": "start node"},
                    {"id": "n2", "shape_type": "rect", "text": "end node"},
                ],
                "edges": [{"from": "n1", "to": "n2", "label": "edge label"}],
            },
        },
        {"type": "asset", "format": "asset_ref", "content": {"asset_id": "asset-1"}},
    ]
    table = _table("parent text", "unselected row")
    table["rows"][0]["source_row"] = 17
    table["rows"][0]["cells"][0].update({
        "colspan": 2,
        "metadata": {"origin": "cell-A17"},
        "children": [{"type": "table", "format": "structured_table", "content": middle}],
    })
    table["rows"][1]["cells"][0]["children"] = [
        {"type": "text", "format": "plain", "content": "unselected child"}
    ]
    # A projection must not serialize an old full-table cache into a row subset.
    table["compact"] = {"preview": "unselected row and unselected child"}
    if typed:
        table = _unit(table).content
    before = json.dumps(table.to_dict() if typed else table)
    subset = table_subset(table, [(1, 1)])
    source = table_source_text(subset)
    payload = _table_payload(subset, {"row_ranges": [[1, 1]]})
    serialized = json.dumps(payload, ensure_ascii=False)

    for text in (
        "parent text", "deep body", "deep caption", "deep header", "child text",
        "image-1", "image caption", "diagram-1", "diagram caption", "start node",
        "end node", "edge label", "asset-1",
    ):
        assert source.count(text) == 1
        assert serialized.count(text) == 1
    assert "unselected" not in source
    assert "unselected" not in serialized
    assert [row["index"] for row in payload["rows"]] == [1]
    assert payload["row_ranges"] == [[1, 1]]
    assert payload["rows"][0]["source_row"] == 17
    cell = payload["rows"][0]["cells"][0]
    assert cell["column_id"] == "c1"
    assert cell["colspan"] == 2
    assert cell["metadata"] == {"origin": "cell-A17"}
    children = cell["children"][0]["content"]["rows"][0]["cells"][0]["children"]
    assert children[0]["content"]["header_rows"][0]["cells"][0]["colspan"] == 2
    assert children[2]["metadata"] == {"page": 3, "bbox": [1, 2, 3, 4]}
    assert children[3]["content"]["edges"][0]["from"] == "n1"
    assert json.dumps(table.to_dict() if typed else table) == before


@pytest.mark.parametrize("include_origin", [False, True])
def test_nested_rowspan_context_is_projected_once(include_origin):
    table = _table("", "selected second row", "selected third row")
    table["columns"].append({"id": "c2", "text": "Details"})
    table["rows"][0]["cells"][0].update({
        "rowspan": 3,
        "children": [{"type": "table", "format": "structured_table", "content": _table("spanning child")}],
    })
    table["rows"][0]["cells"].append({"column_id": "c2", "text": "origin detail"})
    for row in table["rows"][1:]:
        row["cells"][0]["column_id"] = "c2"
    start = 1 if include_origin else 2
    subset = table_subset(_unit(table).content, [(start, 3)])
    payload = _table_payload(subset, {"row_ranges": [[start, 3]]})
    source = table_source_text(subset)

    assert source.count("spanning child") == 1
    assert json.dumps(payload).count("spanning child") == 1
    assert ("origin detail" in source) is include_origin
    assert ("origin detail" in json.dumps(payload)) is include_origin
    assert [row["index"] for row in payload["rows"]] == list(range(start, 4))
    carried = payload["rows"][0]["cells"][0]
    assert carried["rowspan"] == 4 - start
    if not include_origin:
        assert carried["metadata"]["rowspan_context"] == {"source_row_index": 1, "source_rowspan": 3}


@pytest.mark.parametrize("typed", [False, True])
def test_row_subset_clips_selected_origin_rowspans_to_fragment(typed):
    # Issue #78 (HIRA t82): an origin near the fragment end kept its full rowspan.
    table = _table("cancer", "", "", "", "next")
    table["columns"].append({"id": "c2", "text": "Regimen"})
    table["rows"][0]["cells"][0]["rowspan"] = 4
    for row in table["rows"][1:4]:
        row["cells"] = []
    for row in table["rows"]:
        row["cells"].append({"column_id": "c2", "text": f"regimen {row['index']}"})
    content = _unit(table).content if typed else table

    subset = table_subset(content, [(1, 2), (4, 5)])
    payload = _table_payload(subset, {"row_ranges": [[1, 2], [4, 5]]})

    assert [row["index"] for row in payload["rows"]] == [1, 2, 4, 5]
    origin, carried = payload["rows"][0]["cells"][0], payload["rows"][2]["cells"][0]
    assert (origin["text"], origin["rowspan"]) == ("cancer", 2)
    assert (carried["text"], carried["rowspan"]) == ("cancer", 1)
    assert carried["metadata"]["rowspan_context"] == {"source_row_index": 1, "source_rowspan": 4}
    assert table["rows"][0]["cells"][0]["rowspan"] == 4
    for position, row in enumerate(payload["rows"]):
        for cell in row["cells"]:
            assert position + cell.get("rowspan", 1) <= len(payload["rows"])


@pytest.mark.parametrize("action", ["include", "include_rows", "fallback"])
def test_full_and_subset_projection_keeps_large_nested_evidence_in_prompts(action):
    nested = _table(*(f"nested value {index:03d}" for index in range(85)))
    table = _table("prefix " * 2000 + "end of long cell")
    table["rows"][0]["cells"][0]["children"] = [
        {"type": "table", "format": "structured_table", "content": nested}
    ]
    unit = _unit(table)
    chunk = _materialize(unit, action)
    payload = _chunk_payload(chunk)

    assert "nested value 084" in chunk.source.text
    # The bounded source preview is not a substitute for complete structured evidence.
    assert "nested value 084" not in payload["source_text"]
    cell = payload["evidence"][0]["table"]["rows"][0]["cells"][0]
    assert cell["text"].endswith("end of long cell")
    assert len(cell["children"][0]["content"]["rows"]) == 85
    for prompt in (_enrichment_prompt(chunk), _batch_enrichment_prompt([chunk])):
        assert "end of long cell" in prompt
        for index in range(85):
            assert prompt.count(f"nested value {index:03d}") == 1
    assert chunk.evidence.items[0].source_unit_ids == ["b10"]
    assert chunk.evidence.items[0].metadata["pdf"] == {"page": 3}


@pytest.fixture(scope="module", params=["pdf", "hwpx"])
def benefit_unit(request):
    suffix = request.param
    path = Path(__file__).parent / "fixtures" / "corpus" / suffix / f"benefit-criteria-2024-278.{suffix}"
    backend = PdfBackend() if suffix == "pdf" else HwpxBackend()
    parsed = backend.parse(path.read_bytes(), f".{suffix}")
    return next(unit for unit in parsed.units if unit.id == "b10")


@pytest.mark.parametrize("action", ["include", "include_rows"])
def test_benefit_criteria_preserves_all_84_nested_codes(benefit_unit, action):
    nested = benefit_unit.content.rows[0].cells[2].children[0].content
    # The extractor stores the first five codes in its second header row.
    codes = [
        cell["text"].strip()
        for row in [*nested["header_rows"][1:], *nested["rows"]]
        for cell in row["cells"]
        if cell["text"].strip()
    ]
    assert len(codes) == len(set(codes)) == 84
    assert {"A04.7", "I60~I63", "Z54.0"} <= set(codes)
    before = benefit_unit.to_dict()
    chunk = _materialize(benefit_unit, action)
    payload = _chunk_payload(chunk)
    evidence = json.dumps(payload["evidence"], ensure_ascii=False)
    prompt = _enrichment_prompt(chunk)

    for code in codes:
        assert code in chunk.source.text
        assert code in evidence
        assert code in prompt
    table = payload["evidence"][0]["table"]
    assert [row["index"] for row in table["rows"]] == [1]
    assert table["row_ranges"] == ([[1, 1]] if action == "include_rows" else [])
    child_table = table["rows"][0]["cells"][2]["children"][0]["content"]
    assert len(child_table["header_rows"]) == 2
    assert [row["index"] for row in child_table["rows"]] == list(range(1, 17))
    assert chunk.evidence.items[0].content.rows[0].cells[2].children[0].content == nested
    assert benefit_unit.to_dict() == before
