from __future__ import annotations


def test_evidence_unit_carries_direct_format_and_content():
    from rag_document_parser import EvidenceUnit, SourceEvidence

    unit = EvidenceUnit(
        id="b1",
        type="text",
        format="plain",
        source=SourceEvidence(kind="text", text="source text"),
        content="display text",
        metadata={"common": {"chunk_kind": "text"}},
    )

    assert unit.to_dict() == {
        "id": "b1",
        "type": "text",
        "format": "plain",
        "source": {"kind": "text", "text": "source text"},
        "content": "display text",
        "metadata": {"common": {"chunk_kind": "text"}},
    }


def test_chunk_evidence_is_composite_items_only():
    from rag_document_parser import Evidence, EvidenceItem

    evidence = Evidence(
        items=[
            EvidenceItem(
                type="text",
                format="plain",
                content="display text",
                source_unit_ids=["b1"],
                metadata={"page": 1},
            )
        ]
    )

    assert evidence.to_dict() == {
        "items": [
            {
                "type": "text",
                "format": "plain",
                "content": "display text",
                "source_unit_ids": ["b1"],
                "metadata": {"page": 1},
            }
        ]
    }
