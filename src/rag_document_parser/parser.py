from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .llm import LlmConfig, chat_json as _chat_json
from .models import Evidence, ParseResult, RagChunk, SourceEvidence, SourceInfo


@dataclass
class RagDocumentParser:
    llm: LlmConfig | None = None

    def __post_init__(self) -> None:
        if self.llm is None:
            raise ValueError("llm is required")

    def parse(
        self,
        source: bytes | str,
        *,
        suffix: str,
        source_id: str | None = None,
        source_name: str | None = None,
        source_url: str | None = None,
    ) -> ParseResult:
        data = source.encode() if isinstance(source, str) else bytes(source)
        normalized_suffix = suffix.lower()
        markdown = data.decode("utf-8", errors="replace")
        sha256 = hashlib.sha256(data).hexdigest()
        source_info = SourceInfo(
            sha256=sha256,
            suffix=normalized_suffix,
            bytes=len(data),
            id=source_id,
            name=source_name,
            url=source_url,
        )
        return ParseResult(
            source=source_info,
            chunks=_enrich_chunks(_chunks_from_markdown(markdown), self.llm),
        )


def _chunks_from_markdown(markdown: str) -> list[RagChunk]:
    chunks: list[RagChunk] = []
    section_path: list[str] = []
    paragraph_lines: list[str] = []
    table_lines: list[str] = []
    block_index = 1
    table_index = 1

    def flush_paragraph() -> None:
        nonlocal block_index
        meaningful = [line for line in paragraph_lines if line.strip()]
        text = " ".join(line.strip() for line in meaningful).strip()
        paragraph_lines.clear()
        if not text:
            return
        chunk_id = f"b{block_index}"
        block_index += 1
        chunks.append(
            RagChunk(
                id=chunk_id,
                type="text",
                source=SourceEvidence(
                    kind="text",
                    text=text,
                    section_path=list(section_path),
                ),
                evidence=Evidence(kind="text", format="plain", content=text),
                summary="",
                metadata={
                    "common": {
                        "chunk_kind": "text",
                        "section_path": list(section_path),
                        "display_format": "plain",
                    }
                },
            )
        )

    def flush_table() -> None:
        nonlocal block_index, table_index
        lines = [line for line in table_lines if line.strip()]
        table_lines.clear()
        if not lines:
            return
        table_id = f"t{table_index}"
        block_id = f"b{block_index}"
        block_index += 1
        table_index += 1
        headers, rows = _table_parts(lines)
        table_source_text = _table_source_text(headers, rows)
        chunks.append(
            RagChunk(
                id=block_id,
                type="table",
                source=SourceEvidence(
                    kind="table",
                    text=table_source_text,
                    section_path=list(section_path),
                    headers=headers,
                    rows=_source_rows(headers, rows),
                ),
                evidence=Evidence(
                    kind="table",
                    format="markdown_table",
                    content="\n".join(lines),
                ),
                summary="",
                metadata={
                    "common": {
                        "chunk_kind": "table",
                        "section_path": list(section_path),
                        "display_format": "markdown_table",
                    },
                    "table": {
                        "table_id": table_id,
                        "headers": headers,
                        "row_count": len(rows),
                    },
                },
            )
        )

    for line in markdown.splitlines():
        line = line.rstrip()
        if line.lstrip().startswith("#"):
            flush_paragraph()
            flush_table()
            heading = line.lstrip("#").strip()
            if heading:
                section_path = [heading]
            continue
        if line.lstrip().startswith("|"):
            flush_paragraph()
            table_lines.append(line)
            continue
        if table_lines:
            flush_table()
        if line.strip():
            paragraph_lines.append(line)
        else:
            flush_paragraph()

    flush_paragraph()
    flush_table()
    return chunks


def _enrich_chunks(chunks: list[RagChunk], llm: LlmConfig | None) -> list[RagChunk]:
    if llm is None:
        raise ValueError("llm is required")

    enriched: list[RagChunk] = []
    for chunk in chunks:
        enrichment = _chunk_enrichment(chunk, llm)
        enriched.append(
            RagChunk(
                id=chunk.id,
                type=chunk.type,
                source=chunk.source,
                evidence=chunk.evidence,
                summary=enrichment["summary"],
                keywords=enrichment["keywords"],
                questions=enrichment["questions"],
                metadata=chunk.metadata,
            )
        )
    return enriched


def _chunk_enrichment(chunk: RagChunk, llm: LlmConfig) -> dict[str, Any]:
    prompt = _enrichment_prompt(chunk)
    payload = _chat_json(prompt, llm)
    enrichment = _normalize_enrichment(payload)
    if enrichment is None:
        raise ValueError(f"LLM enrichment failed for chunk {chunk.id}")
    return enrichment


def _enrichment_prompt(chunk: RagChunk) -> str:
    payload = {
        "id": chunk.id,
        "type": chunk.type,
        "source": chunk.source.to_dict(),
        "evidence": chunk.evidence.to_dict(),
        "metadata": chunk.metadata,
    }
    return (
        "아래 문서 청크를 RAG 검색과 근거 제시에 사용할 수 있도록 보강해줘.\n"
        "반드시 다음 JSON object만 반환해: "
        '{"summary": string, "keywords": string[], "questions": string[]}.\n'
        "summary는 청크 내용을 짧게 요약하고, keywords는 검색에 유용한 핵심어를 뽑고, "
        "questions는 이 청크만으로 답변 가능한 질문을 작성해.\n\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )


def _normalize_enrichment(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    summary = payload.get("summary")
    keywords = payload.get("keywords")
    questions = payload.get("questions")
    if not isinstance(summary, str) or not summary.strip():
        return None
    if not _is_string_list(keywords) or not _is_string_list(questions):
        return None
    return {
        "summary": summary.strip(),
        "keywords": [item.strip() for item in keywords if item.strip()],
        "questions": [item.strip() for item in questions if item.strip()],
    }


def _is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _table_parts(lines: list[str]) -> tuple[list[str], list[list[str]]]:
    rows = [_split_table_row(line) for line in lines]
    if len(rows) < 3:
        return [], []
    return rows[0], rows[2:]


def _source_rows(headers: list[str], rows: list[list[str]]) -> list[dict[str, object]]:
    source_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        source_rows.append(
            {
                "index": index,
                "cells": {
                    header: value
                    for header, value in zip(headers, row, strict=False)
                },
            }
        )
    return source_rows


def _table_source_text(headers: list[str], rows: list[list[str]]) -> str:
    parts: list[str] = []
    for row in rows:
        cells = [
            f"{header}={value}"
            for header, value in zip(headers, row, strict=False)
            if value
        ]
        if cells:
            parts.append("; ".join(cells))
    return "\n".join(parts)


def _split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]
