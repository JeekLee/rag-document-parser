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
