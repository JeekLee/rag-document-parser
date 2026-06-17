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


def test_rag_chunk_serializes_composite_evidence_and_enrichment_fields():
    from rag_document_parser import Evidence, EvidenceItem, RagChunk, SourceEvidence

    chunk = RagChunk(
        id="chunk-1",
        type="mixed",
        source=SourceEvidence(kind="mixed", text="source text"),
        evidence=Evidence(
            items=[
                EvidenceItem(
                    type="text",
                    format="plain",
                    content="display text",
                    source_unit_ids=["b1"],
                    metadata={},
                )
            ]
        ),
        summary="summary",
        keywords=["keyword"],
        questions=["question?"],
        metadata={"source_unit_ids": ["b1"]},
    )

    assert chunk.to_dict() == {
        "id": "chunk-1",
        "type": "mixed",
        "source": {"kind": "mixed", "text": "source text"},
        "evidence": {
            "items": [
                {
                    "type": "text",
                    "format": "plain",
                    "content": "display text",
                    "source_unit_ids": ["b1"],
                    "metadata": {},
                }
            ]
        },
        "summary": "summary",
        "keywords": ["keyword"],
        "questions": ["question?"],
        "metadata": {"source_unit_ids": ["b1"]},
    }


def test_evidence_item_omits_format_when_not_set():
    from rag_document_parser import EvidenceItem

    assert EvidenceItem(type="text", content="plain").to_dict() == {
        "type": "text",
        "content": "plain",
        "source_unit_ids": [],
        "metadata": {},
    }
