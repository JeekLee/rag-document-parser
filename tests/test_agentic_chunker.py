from __future__ import annotations


def _text_unit(id: str, text: str):
    from rag_document_parser import EvidenceUnit, SourceEvidence

    return EvidenceUnit(
        id=id,
        type="text",
        format="plain",
        source=SourceEvidence(kind="text", text=text),
        content=text,
        metadata={"common": {"chunk_kind": "text", "section_path": [], "display_format": "plain"}},
    )


def _table_unit(id: str):
    from rag_document_parser import EvidenceUnit, SourceEvidence

    table = {
        "caption": None,
        "columns": [
            {"id": "c1", "text": "항목"},
            {"id": "c2", "text": "내용"},
        ],
        "rows": [
            {
                "index": 1,
                "cells": [
                    {"column_id": "c1", "text": "A", "rowspan": 1, "colspan": 1, "children": []},
                    {"column_id": "c2", "text": "Alpha", "rowspan": 1, "colspan": 1, "children": []},
                ],
            },
            {
                "index": 2,
                "cells": [
                    {"column_id": "c1", "text": "B", "rowspan": 1, "colspan": 1, "children": []},
                    {"column_id": "c2", "text": "Beta", "rowspan": 1, "colspan": 1, "children": []},
                ],
            },
        ],
    }
    return EvidenceUnit(
        id=id,
        type="table",
        format="structured_table",
        source=SourceEvidence(
            kind="table",
            text="table: 2 columns\nrow 1: 항목=A; 내용=Alpha\nrow 2: 항목=B; 내용=Beta",
        ),
        content=table,
        metadata={
            "common": {"chunk_kind": "table", "section_path": [], "display_format": "structured_table"},
            "table": {"table_id": "t1", "headers": ["항목", "내용"], "row_count": 2},
        },
    )


def test_agentic_chunker_materializes_cross_kind_chunk_from_plan():
    from rag_document_parser.chunk import EvidenceUnitAgenticChunker

    units = [_text_unit("b1", "기준 설명"), _table_unit("b2")]

    def plan_fn(window, cfg, max_units):
        return [
            {
                "unit_ids": ["b1", "b2"],
                "operations": [
                    {"unit_id": "b1", "action": "include"},
                    {"unit_id": "b2", "action": "include"},
                ],
                "context_unit_ids": [],
                "title": "기준 설명과 표",
                "summary": "기준 설명과 표를 함께 제공한다.",
                "keywords": ["기준", "표"],
                "questions": ["기준 설명과 표에는 무엇이 있나요?"],
            }
        ]

    chunks = EvidenceUnitAgenticChunker(llm=None, plan_fn=plan_fn).chunk(units)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.type == "mixed"
    assert chunk.summary == "기준 설명과 표를 함께 제공한다."
    assert chunk.keywords == ["기준", "표"]
    assert chunk.questions == ["기준 설명과 표에는 무엇이 있나요?"]
    assert chunk.metadata["source_unit_ids"] == ["b1", "b2"]
    assert chunk.metadata["context_unit_ids"] == []
    assert [item.type for item in chunk.evidence.items] == ["text", "table"]
    assert chunk.evidence.items[0].content == "기준 설명"
    assert chunk.evidence.items[1].content["rows"][1]["index"] == 2


def test_agentic_chunker_materializes_table_row_subset():
    from rag_document_parser.chunk import EvidenceUnitAgenticChunker

    def plan_fn(window, cfg, max_units):
        return [
            {
                "unit_ids": ["b2"],
                "operations": [
                    {"unit_id": "b2", "action": "include_rows", "row_ranges": [[2, 2]]}
                ],
                "title": "B 항목",
                "summary": "B 항목만 제공한다.",
                "keywords": ["B", "Beta"],
                "questions": ["B 항목의 내용은 무엇인가요?"],
            }
        ]

    chunks = EvidenceUnitAgenticChunker(llm=None, plan_fn=plan_fn).chunk([_table_unit("b2")])

    table_item = chunks[0].evidence.items[0]
    assert table_item.type == "table"
    assert table_item.format == "structured_table"
    assert [row["index"] for row in table_item.content["rows"]] == [2]
    assert "row 2" in chunks[0].source.text
    assert "row 1" not in chunks[0].source.text
