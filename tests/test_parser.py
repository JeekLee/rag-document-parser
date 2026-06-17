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
    assert result.preview_markdown.startswith("# 요양급여 기준")
    assert [chunk.type for chunk in result.chunks] == ["text", "table"]

    text_chunk, table_chunk = result.chunks
    assert text_chunk.llm_text == (
        "section: 요양급여 기준\n"
        "코로나19 대면투약관리료는 다음 기준에 따라 산정한다."
    )
    assert text_chunk.source.section_path == ["요양급여 기준"]
    assert text_chunk.display.format == "markdown"

    assert table_chunk.llm_text == (
        "section: 요양급여 기준\n"
        "table:\n"
        "columns: 대상 | 청구방법\n"
        "row 1: 대상=약국; 청구방법=대면투약관리료 코드로 청구"
    )
    assert table_chunk.display.format == "markdown"
    assert table_chunk.source.table_id == "t1"


def test_parse_result_to_dict_is_json_serializable():
    from rag_document_parser import RagDocumentParser

    raw = b"plain paragraph"
    payload = RagDocumentParser().parse(raw, suffix=".txt").to_dict()

    assert payload["source"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert payload["chunks"][0]["display"]["content"] == "plain paragraph"
    assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload
