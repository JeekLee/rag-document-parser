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
        source=SourceEvidence(kind="chunk", text="source text"),
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
        "source": {"kind": "chunk", "text": "source text"},
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


def test_structured_evidence_content_is_modeled_not_raw_dict():
    from rag_document_parser import EvidenceUnit, SourceEvidence
    from rag_document_parser.models import StructuredTableContent

    unit = EvidenceUnit(
        id="tbl1",
        type="table",
        format="structured_table",
        source=SourceEvidence(kind="table", text="row 1: A"),
        content={
            "caption": None,
            "columns": [{"id": "c1", "text": "구분"}],
            "rows": [
                {
                    "index": 1,
                    "cells": [
                        {
                            "column_id": "c1",
                            "text": "A",
                            "rowspan": 1,
                            "colspan": 1,
                            "children": [],
                        }
                    ],
                }
            ],
        },
    )

    assert isinstance(unit.content, StructuredTableContent)
    assert not isinstance(unit.content, dict)
    assert unit.content.columns[0].text == "구분"
    assert unit.content["columns"][0]["text"] == "구분"
    assert unit.to_dict()["content"]["rows"][0]["cells"][0]["text"] == "A"


def test_rag_chunk_shape_is_introspectable_from_pydantic_schema():
    from rag_document_parser import RagChunk

    schema = RagChunk.model_json_schema()

    assert schema["properties"]["id"]["type"] == "string"
    assert "Evidence" in str(schema["properties"]["evidence"])
    assert schema["properties"]["summary"]["type"] == "string"
    assert schema["properties"]["keywords"]["items"]["type"] == "string"
    assert schema["properties"]["questions"]["items"]["type"] == "string"


def test_parsed_document_is_a_canonical_model():
    from rag_document_parser.models import ParsedDocument

    parsed = ParsedDocument(units=[], assets=[], quality_warnings=[])

    assert parsed.to_dict() == {
        "units": [],
        "assets": [],
        "quality_warnings": [],
    }
