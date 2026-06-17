from __future__ import annotations

import hashlib
import json


def test_parse_text_document_returns_llm_enriched_chunks(monkeypatch):
    from rag_document_parser import LlmConfig, RagDocumentParser

    responses = iter(
        [
            {
                "summary": "코로나19 대면투약관리료 산정 기준을 설명합니다.",
                "keywords": ["코로나19", "대면투약관리료", "산정 기준"],
                "questions": ["코로나19 대면투약관리료는 어떤 기준에 따라 산정하나요?"],
            },
            {
                "summary": "약국의 대면투약관리료 청구방법을 설명하는 표입니다.",
                "keywords": ["약국", "대면투약관리료", "청구방법"],
                "questions": ["약국은 대면투약관리료를 어떻게 청구하나요?"],
            },
        ]
    )
    prompts: list[str] = []

    def fake_chat_json(prompt, cfg):
        prompts.append(prompt)
        return next(responses)

    monkeypatch.setattr("rag_document_parser.parser._chat_json", fake_chat_json)

    raw = (
        "# 요양급여 기준\n\n"
        "코로나19 대면투약관리료는 다음 기준에 따라 산정한다.\n\n"
        "| 대상 | 청구방법 |\n"
        "| --- | --- |\n"
        "| 약국 | 대면투약관리료 코드로 청구 |\n"
    ).encode()

    result = RagDocumentParser(
        llm=LlmConfig(url="http://llm.test/v1", api_key="test", model="test-model")
    ).parse(
        raw,
        suffix=".md",
        source_id="notice-1/attachment-1",
        source_name="기준.md",
    )

    assert result.source.sha256 == hashlib.sha256(raw).hexdigest()
    assert result.source.suffix == ".md"
    assert not hasattr(result, "preview_markdown")
    assert [chunk.type for chunk in result.chunks] == ["text", "table"]
    assert len(prompts) == 2
    assert "embedding_text" not in result.chunks[0].to_dict()

    text_chunk, table_chunk = result.chunks
    assert text_chunk.source.kind == "text"
    assert text_chunk.source.text == "코로나19 대면투약관리료는 다음 기준에 따라 산정한다."
    assert text_chunk.source.section_path == ["요양급여 기준"]
    assert text_chunk.summary == "코로나19 대면투약관리료 산정 기준을 설명합니다."
    assert text_chunk.keywords == ["코로나19", "대면투약관리료", "산정 기준"]
    assert text_chunk.questions == ["코로나19 대면투약관리료는 어떤 기준에 따라 산정하나요?"]
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
    assert table_chunk.summary == "약국의 대면투약관리료 청구방법을 설명하는 표입니다."
    assert table_chunk.keywords == ["약국", "대면투약관리료", "청구방법"]
    assert table_chunk.questions == ["약국은 대면투약관리료를 어떻게 청구하나요?"]
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


def test_markdown_backend_returns_evidence_units():
    from rag_document_parser import EvidenceUnit, MarkdownBackend

    raw = (
        "# Section\n\n"
        "Plain paragraph.\n\n"
        "| A | B |\n"
        "| --- | --- |\n"
        "| one | two |\n"
    ).encode()

    parsed = MarkdownBackend().parse(raw, ".md")

    assert not hasattr(parsed, "chunks")
    assert [unit.id for unit in parsed.units] == ["b1", "b2"]
    assert all(isinstance(unit, EvidenceUnit) for unit in parsed.units)

    text_unit, table_unit = parsed.units
    assert text_unit.type == "text"
    assert text_unit.source.text == "Plain paragraph."
    assert text_unit.evidence.content == "Plain paragraph."
    assert text_unit.metadata["common"] == {
        "chunk_kind": "text",
        "section_path": ["Section"],
        "display_format": "plain",
    }

    assert table_unit.type == "table"
    assert table_unit.source.headers == ["A", "B"]
    assert table_unit.source.rows == [
        {"index": 1, "cells": {"A": "one", "B": "two"}}
    ]
    assert table_unit.evidence.format == "markdown_table"


def test_parse_result_to_dict_is_json_serializable(monkeypatch):
    from rag_document_parser import LlmConfig, RagDocumentParser

    monkeypatch.setattr(
        "rag_document_parser.parser._chat_json",
        lambda prompt, cfg: {
            "summary": "Plain paragraph summary.",
            "keywords": ["plain", "paragraph"],
            "questions": ["What does the paragraph say?"],
        },
    )

    raw = b"plain paragraph"
    payload = RagDocumentParser(
        llm=LlmConfig(url="http://llm.test/v1", api_key="test", model="test-model")
    ).parse(raw, suffix=".txt").to_dict()

    assert payload["source"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert "preview_markdown" not in payload
    assert "source_pointer" not in payload["chunks"][0]
    assert "embedding_text" not in payload["chunks"][0]
    assert payload["chunks"][0]["source"] == {
        "kind": "text",
        "text": "plain paragraph",
        "section_path": [],
    }
    assert payload["chunks"][0]["summary"] == "Plain paragraph summary."
    assert payload["chunks"][0]["keywords"] == ["plain", "paragraph"]
    assert payload["chunks"][0]["questions"] == ["What does the paragraph say?"]
    assert payload["chunks"][0]["evidence"] == {
        "kind": "text",
        "format": "plain",
        "content": "plain paragraph",
    }
    assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload


def test_source_does_not_require_position_offsets(monkeypatch):
    from rag_document_parser import LlmConfig, RagDocumentParser

    monkeypatch.setattr(
        "rag_document_parser.parser._chat_json",
        lambda prompt, cfg: {
            "summary": "한글 요약",
            "keywords": ["한글"],
            "questions": ["무슨 내용인가요?"],
        },
    )

    raw = "# H\n\n한글".encode()
    chunk = RagDocumentParser(
        llm=LlmConfig(url="http://llm.test/v1", api_key="test", model="test-model")
    ).parse(raw, suffix=".md").chunks[0]

    assert chunk.source.text == "한글"
    assert chunk.source.section_path == ["H"]
    assert not hasattr(chunk, "source_pointer")


def test_parser_requires_llm_config():
    from rag_document_parser import RagDocumentParser

    try:
        RagDocumentParser(llm=None)
    except ValueError as exc:
        assert "llm is required" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_parse_fails_when_llm_enrichment_is_invalid(monkeypatch):
    from rag_document_parser import LlmConfig, RagDocumentParser

    monkeypatch.setattr("rag_document_parser.parser._chat_json", lambda prompt, cfg: None)

    parser = RagDocumentParser(
        llm=LlmConfig(url="http://llm.test/v1", api_key="test", model="test-model")
    )
    try:
        parser.parse(b"plain paragraph", suffix=".txt")
    except ValueError as exc:
        assert "LLM enrichment failed" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_custom_backend_can_be_registered_for_suffix(monkeypatch):
    from rag_document_parser import (
        Evidence,
        EvidenceUnit,
        LlmConfig,
        RagDocumentParser,
        SourceEvidence,
    )
    from rag_document_parser.backends import ParsedDocument

    monkeypatch.setattr(
        "rag_document_parser.parser._chat_json",
        lambda prompt, cfg: {
            "summary": "Custom backend summary.",
            "keywords": ["custom"],
            "questions": ["What did the custom backend parse?"],
        },
    )

    class CustomBackend:
        calls: list[tuple[bytes, str]]

        def __init__(self) -> None:
            self.calls = []

        def parse(self, data: bytes, suffix: str) -> ParsedDocument:
            self.calls.append((data, suffix))
            return ParsedDocument(
                units=[
                    EvidenceUnit(
                        id="c1",
                        type="text",
                        source=SourceEvidence(kind="text", text="custom text"),
                        evidence=Evidence(kind="text", format="plain", content="custom text"),
                    )
                ],
                quality_warnings=[
                    {
                        "type": "custom_warning",
                        "severity": "low",
                        "message": "backend warning",
                    }
                ],
            )

    backend = CustomBackend()
    result = RagDocumentParser(
        llm=LlmConfig(url="http://llm.test/v1", api_key="test", model="test-model"),
        backends={".custom": backend},
    ).parse(b"custom bytes", suffix=".CUSTOM")

    assert backend.calls == [(b"custom bytes", ".custom")]
    assert result.source.suffix == ".custom"
    assert result.chunks[0].source.text == "custom text"
    assert result.chunks[0].summary == "Custom backend summary."
    assert result.quality_warnings == [
        {
            "type": "custom_warning",
            "severity": "low",
            "message": "backend warning",
        }
    ]


def test_unsupported_suffix_fails_before_llm_call(monkeypatch):
    from rag_document_parser import LlmConfig, RagDocumentParser

    def fail_chat_json(prompt, cfg):
        raise AssertionError("LLM should not be called for unsupported formats")

    monkeypatch.setattr("rag_document_parser.parser._chat_json", fail_chat_json)

    parser = RagDocumentParser(
        llm=LlmConfig(url="http://llm.test/v1", api_key="test", model="test-model")
    )
    try:
        parser.parse(b"not supported", suffix=".docx")
    except ValueError as exc:
        assert "Unsupported format" in str(exc)
        assert ".docx" in str(exc)
    else:
        raise AssertionError("expected ValueError")
