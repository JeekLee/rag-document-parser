from __future__ import annotations

import hashlib
import json


def test_parse_text_document_returns_rag_ready_chunks():
    from rag_document_parser import RagDocumentParser

    raw = (
        "# 요양급여 기준\n\n"
        "코로나19 대면투약관리료는 다음 기준에 따라 산정한다.\n\n"
        "| 대상 | 청구방법 |\n"
        "| --- | --- |\n"
        "| 약국 | 대면투약관리료 코드로 청구 |\n"
    ).encode()

    result = RagDocumentParser().parse(
        raw,
        suffix=".md",
        source_id="notice-1/attachment-1",
        source_name="기준.md",
    )

    assert result.source.sha256 == hashlib.sha256(raw).hexdigest()
    assert result.source.suffix == ".md"
    assert not hasattr(result, "preview_markdown")
    assert [chunk.type for chunk in result.chunks] == ["text", "table"]

    text_chunk, table_chunk = result.chunks
    assert text_chunk.source.kind == "text"
    assert text_chunk.source.text == "코로나19 대면투약관리료는 다음 기준에 따라 산정한다."
    assert text_chunk.source.section_path == ["요양급여 기준"]
    assert text_chunk.embedding_text == (
        "section: 요양급여 기준\n"
        "코로나19 대면투약관리료는 다음 기준에 따라 산정한다."
    )
    assert not hasattr(text_chunk, "source_pointer")
    assert text_chunk.evidence.kind == "text"
    assert text_chunk.evidence.format == "plain"
    assert text_chunk.evidence.content == text_chunk.source.text

    assert table_chunk.source.kind == "table"
    assert table_chunk.source.section_path == ["요양급여 기준"]
    assert table_chunk.source.headers == ["대상", "청구방법"]
    assert table_chunk.source.rows == [
        {"index": 1, "cells": {"대상": "약국", "청구방법": "대면투약관리료 코드로 청구"}}
    ]
    assert table_chunk.source.text == "대상=약국; 청구방법=대면투약관리료 코드로 청구"
    assert table_chunk.embedding_text == (
        "section: 요양급여 기준\n"
        "table:\n"
        "columns: 대상 | 청구방법\n"
        "row 1: 대상=약국; 청구방법=대면투약관리료 코드로 청구"
    )
    assert table_chunk.evidence.kind == "table"
    assert table_chunk.evidence.format == "markdown_table"
    assert table_chunk.evidence.content == (
        "| 대상 | 청구방법 |\n"
        "| --- | --- |\n"
        "| 약국 | 대면투약관리료 코드로 청구 |"
    )
    assert table_chunk.metadata["common"] == {
        "chunk_kind": "table",
        "section_path": ["요양급여 기준"],
        "display_format": "markdown_table",
    }
    assert table_chunk.metadata["table"] == {
        "table_id": "t1",
        "headers": ["대상", "청구방법"],
        "row_count": 1,
    }


def test_parse_result_to_dict_is_json_serializable():
    from rag_document_parser import RagDocumentParser

    raw = b"plain paragraph"
    payload = RagDocumentParser().parse(raw, suffix=".txt").to_dict()

    assert payload["source"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert "preview_markdown" not in payload
    assert "source_pointer" not in payload["chunks"][0]
    assert payload["chunks"][0]["source"] == {
        "kind": "text",
        "text": "plain paragraph",
        "section_path": [],
    }
    assert payload["chunks"][0]["embedding_text"] == "plain paragraph"
    assert payload["chunks"][0]["evidence"] == {
        "kind": "text",
        "format": "plain",
        "content": "plain paragraph",
    }
    assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload


def test_source_does_not_require_position_offsets():
    from rag_document_parser import RagDocumentParser

    raw = "# H\n\n한글".encode()
    chunk = RagDocumentParser().parse(raw, suffix=".md").chunks[0]

    assert chunk.source.text == "한글"
    assert chunk.source.section_path == ["H"]
    assert not hasattr(chunk, "source_pointer")
