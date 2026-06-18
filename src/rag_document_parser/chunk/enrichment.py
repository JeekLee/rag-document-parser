from __future__ import annotations

import json
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

from ..models import RagChunk
from .llm import LlmConfig, chat_json

ChunkEnrichmentFn = Callable[[RagChunk, LlmConfig | None], Any]
ChatJsonFn = Callable[[str, LlmConfig], Any]


class Enricher(Protocol):
    def enrich(self, chunks: list[RagChunk]) -> list[RagChunk]:
        """Add summary, keyword, and question metadata to chunks."""
        ...


_CHUNK_ENRICHMENT_PROMPT = """\
당신은 최종 RagChunk enrichment 생성기입니다.
아래 RagChunk는 EvidenceUnit chunking, boundary 보정, table row split 후 확정된 최종 검색 단위입니다.

규칙:
- 원문과 evidence에 있는 정보만 사용합니다.
- summary는 이 chunk가 답할 수 있는 내용을 1-2문장으로 요약합니다.
- keywords는 검색에 유용한 핵심 명사/표현 3-8개를 작성합니다.
- questions는 이 chunk 하나로 답할 수 있는 실제 검색 질문 2-5개를 작성합니다.
- Q&A 표라면 질의 컬럼의 실제 질문을 우선 활용합니다.
- "무엇을 알 수 있나요?" 같은 범용 질문은 피합니다.
- JSON object만 출력합니다.

RagChunk:
{chunk}

JSON object only:
{{
  "summary": "요약",
  "keywords": ["키워드"],
  "questions": ["질문"]
}}
"""


@dataclass(frozen=True)
class _ChunkEnrichment:
    summary: str
    keywords: list[str]
    questions: list[str]


class RagChunkEnricher:
    def __init__(
        self,
        *,
        llm: LlmConfig | None = None,
        enrich_fn: ChunkEnrichmentFn | None = None,
        chat_fn: ChatJsonFn = chat_json,
        max_concurrency: int = 4,
    ) -> None:
        self._llm = llm
        self._enrich_fn = enrich_fn
        self._chat_fn = chat_fn
        self._concurrency = max(1, max_concurrency)

    def enrich(self, chunks: list[RagChunk]) -> list[RagChunk]:
        if not chunks:
            return []
        if self._llm is None and self._enrich_fn is None:
            return [self._heuristic_enrich_if_needed(chunk) for chunk in chunks]

        with ThreadPoolExecutor(max_workers=self._concurrency) as executor:
            return list(executor.map(self._enrich_chunk, chunks))

    def _heuristic_enrich_if_needed(self, chunk: RagChunk) -> RagChunk:
        if not _needs_enrichment(chunk):
            return chunk
        enrichment = _heuristic_enrichment(chunk)
        return _replace_enrichment(chunk, enrichment, method="heuristic")

    def _enrich_chunk(self, chunk: RagChunk) -> RagChunk:
        try:
            raw = self._call_enricher(chunk)
            enrichment = _parse_enrichment(raw, fallback=_heuristic_enrichment(chunk))
        except Exception as exc:
            if _needs_enrichment(chunk):
                return _replace_enrichment(
                    chunk,
                    _heuristic_enrichment(chunk),
                    method="heuristic",
                    warning=_enrichment_warning(exc),
                )
            return _preserve_enrichment(chunk, warning=_enrichment_warning(exc))
        method = "custom" if self._enrich_fn is not None else "llm"
        return _replace_enrichment(chunk, enrichment, method=method)

    def _call_enricher(self, chunk: RagChunk) -> Any:
        if self._enrich_fn is not None:
            return self._enrich_fn(chunk, self._llm)
        if self._llm is None:
            return _heuristic_enrichment(chunk)
        return self._chat_fn(_enrichment_prompt(chunk), self._llm)


def _enrichment_prompt(chunk: RagChunk) -> str:
    return _CHUNK_ENRICHMENT_PROMPT.format(
        chunk=json.dumps(_chunk_payload(chunk), ensure_ascii=False, indent=2)
    )


def _chunk_payload(chunk: RagChunk) -> dict[str, Any]:
    metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
    return {
        "id": chunk.id,
        "source_unit_ids": _strings(metadata.get("source_unit_ids")),
        "context_unit_ids": _strings(metadata.get("context_unit_ids")),
        "title": metadata.get("title") if isinstance(metadata.get("title"), str) else "",
        "source_text": _truncate(chunk.source.text, 12000),
        "evidence": _evidence_payload(chunk),
    }


def _evidence_payload(chunk: RagChunk) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for item in chunk.evidence.items:
        record: dict[str, Any] = {
            "type": item.type,
            "format": item.format,
            "source_unit_ids": list(item.source_unit_ids),
        }
        if item.type == "table" and isinstance(item.content, dict):
            record["table"] = _table_payload(item.content, item.metadata)
        elif isinstance(item.content, str):
            record["text"] = _truncate(item.content, 2000)
        else:
            record["content_preview"] = _truncate(_json_preview(item.content), 2000)
        payload.append(record)
    return payload


def _table_payload(table: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    rows = table.get("rows", [])
    columns = table.get("columns", [])
    return {
        "caption": table.get("caption") if isinstance(table.get("caption"), str) else "",
        "columns": [
            str(column.get("text", "")).strip()
            for column in columns
            if isinstance(column, dict) and str(column.get("text", "")).strip()
        ][:80],
        "row_ranges": metadata.get("row_ranges") if isinstance(metadata.get("row_ranges"), list) else [],
        "rows": [
            _table_row_payload(row, columns if isinstance(columns, list) else [])
            for row in rows
            if isinstance(row, dict)
        ][:60],
    }


def _table_row_payload(row: dict[str, Any], columns: list[Any]) -> dict[str, Any]:
    cells: list[dict[str, str]] = []
    for cell in row.get("cells", []) if isinstance(row.get("cells"), list) else []:
        if not isinstance(cell, dict):
            continue
        text = _normalize_space(str(cell.get("text", "")))
        if not text:
            continue
        cells.append(
            {
                "column": _column_label(columns, cell.get("column_id")),
                "text": _truncate(text, 500),
            }
        )
    return {"index": row.get("index"), "cells": cells}


def _parse_enrichment(raw: Any, *, fallback: _ChunkEnrichment) -> _ChunkEnrichment:
    if not isinstance(raw, dict):
        raise ValueError("chunk enrichment result must be an object")

    summary = raw.get("summary") if isinstance(raw.get("summary"), str) else ""
    keywords = _strings(raw.get("keywords"))
    questions = _strings(raw.get("questions"))
    return _ChunkEnrichment(
        summary=_normalize_space(summary) or fallback.summary,
        keywords=_unique([_normalize_space(keyword) for keyword in keywords])[:8] or fallback.keywords,
        questions=_unique([_clean_question(question) for question in questions])[:5] or fallback.questions,
    )


def _replace_enrichment(
    chunk: RagChunk,
    enrichment: _ChunkEnrichment,
    *,
    method: str,
    warning: dict[str, Any] | None = None,
) -> RagChunk:
    metadata = _enriched_metadata(chunk, method=method, warning=warning)
    return RagChunk(
        id=chunk.id,
        source=chunk.source,
        evidence=chunk.evidence,
        summary=enrichment.summary,
        keywords=list(enrichment.keywords),
        questions=list(enrichment.questions),
        metadata=metadata,
    )


def _preserve_enrichment(
    chunk: RagChunk,
    *,
    warning: dict[str, Any],
) -> RagChunk:
    metadata = _enriched_metadata(chunk, method="preserved", warning=warning)
    return RagChunk(
        id=chunk.id,
        source=chunk.source,
        evidence=chunk.evidence,
        summary=chunk.summary,
        keywords=list(chunk.keywords),
        questions=list(chunk.questions),
        metadata=metadata,
    )


def _enriched_metadata(
    chunk: RagChunk,
    *,
    method: str,
    warning: dict[str, Any] | None,
) -> dict[str, Any]:
    metadata = dict(chunk.metadata)
    metadata.pop("_needs_enrichment", None)
    metadata["_enrichment"] = {
        "stage": "post_chunking",
        "method": method,
    }
    if warning is not None:
        warnings = _dicts(metadata.get("_warnings"))
        warnings.append(warning)
        metadata["_warnings"] = warnings
    return metadata


def _enrichment_warning(exc: Exception) -> dict[str, Any]:
    reason = str(exc) or exc.__class__.__name__
    return {"type": "rag_chunk_enrichment_failed", "reason": reason}


def _needs_enrichment(chunk: RagChunk) -> bool:
    metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
    if metadata.get("_needs_enrichment") is True:
        return True
    return not chunk.summary or not chunk.keywords or not chunk.questions


def _heuristic_enrichment(chunk: RagChunk) -> _ChunkEnrichment:
    qa_questions = _qa_table_questions(chunk)
    context = _chunk_context(chunk)
    keywords = _fallback_keywords(chunk.source.text)
    if qa_questions:
        return _ChunkEnrichment(
            summary=_qa_summary(context, qa_questions, chunk),
            keywords=keywords,
            questions=qa_questions,
        )
    if _has_table_evidence(chunk):
        return _ChunkEnrichment(
            summary=_table_summary(context, chunk),
            keywords=keywords,
            questions=_table_questions(context, chunk),
        )
    return _ChunkEnrichment(
        summary=_text_summary(chunk.source.text),
        keywords=keywords,
        questions=_text_questions(chunk),
    )


def _qa_table_questions(chunk: RagChunk) -> list[str]:
    questions: list[str] = []
    for item in chunk.evidence.items:
        if item.type != "table" or not isinstance(item.content, dict):
            continue
        columns = item.content.get("columns", [])
        if not isinstance(columns, list):
            columns = []
        for row in item.content.get("rows", []) if isinstance(item.content.get("rows"), list) else []:
            if not isinstance(row, dict):
                continue
            for cell in row.get("cells", []) if isinstance(row.get("cells"), list) else []:
                if not isinstance(cell, dict):
                    continue
                label = _normalize_space(_column_label(columns, cell.get("column_id")))
                if not _looks_like_question_column(label):
                    continue
                text = _normalize_space(str(cell.get("text", "")))
                if len(text) < 4:
                    continue
                questions.append(_clean_question(text))
    return _unique(questions)[:5]


def _looks_like_question_column(label: str) -> bool:
    lowered = label.lower()
    return (
        "질의" in label
        or "질문" in label
        or lowered in {"q", "qa", "q&a", "question"}
        or lowered.startswith("q ")
    )


def _qa_summary(context: str, questions: list[str], chunk: RagChunk) -> str:
    subject = context or _metadata_title(chunk) or "Q&A"
    row_label = _chunk_row_label(chunk)
    prefix = f"{subject}의 {row_label} Q&A" if row_label else f"{subject} Q&A"
    preview = " / ".join(question.rstrip("?") for question in questions[:2])
    return _truncate(f"{prefix}를 다룹니다: {preview}", 300)


def _table_summary(context: str, chunk: RagChunk) -> str:
    subject = context or _metadata_title(chunk) or "표"
    row_label = _chunk_row_label(chunk)
    preview = _cell_preview(chunk)
    if row_label and preview:
        return _truncate(f"{subject}의 {row_label} 항목을 다룹니다: {preview}", 300)
    if row_label:
        return _truncate(f"{subject}의 {row_label} 항목을 다룹니다.", 300)
    if preview:
        return _truncate(f"{subject} 표 내용을 다룹니다: {preview}", 300)
    return _text_summary(chunk.source.text)


def _table_questions(context: str, chunk: RagChunk) -> list[str]:
    subject = context or _metadata_title(chunk) or "이 표"
    row_label = _chunk_row_label(chunk)
    if row_label:
        return [
            f"{subject}의 {row_label} 항목은 무엇인가요?",
            f"{subject}에서 해당 행의 기준이나 답변은 어떻게 설명되나요?",
        ]
    return [
        f"{subject} 표에는 어떤 항목이 포함되어 있나요?",
        f"{subject} 표의 주요 기준이나 값은 무엇인가요?",
    ]


def _text_questions(chunk: RagChunk) -> list[str]:
    title = _metadata_title(chunk)
    topic = title or _first_topic(chunk.source.text) or "이 내용"
    return [
        f"{topic}의 핵심 내용은 무엇인가요?",
        f"{topic}에서 확인해야 할 기준이나 조건은 무엇인가요?",
    ]


def _chunk_context(chunk: RagChunk) -> str:
    for line in chunk.source.text.splitlines():
        stripped = line.strip()
        if stripped.startswith("context:"):
            return _truncate(_normalize_space(stripped.removeprefix("context:")), 220)

    title = _metadata_title(chunk)
    if title:
        return title

    for item in chunk.evidence.items:
        if item.type != "table" or not isinstance(item.content, dict):
            continue
        caption = item.content.get("caption")
        if isinstance(caption, str) and caption.strip():
            return _truncate(_normalize_space(caption), 220)
    return ""


def _metadata_title(chunk: RagChunk) -> str:
    metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
    title = metadata.get("title")
    return _normalize_space(title) if isinstance(title, str) else ""


def _chunk_row_label(chunk: RagChunk) -> str:
    ranges: list[list[int]] = []
    for item in chunk.evidence.items:
        if not isinstance(item.metadata, dict):
            continue
        row_ranges = item.metadata.get("row_ranges")
        if isinstance(row_ranges, list):
            ranges.extend(
                row_range
                for row_range in row_ranges
                if (
                    isinstance(row_range, list)
                    and len(row_range) == 2
                    and type(row_range[0]) is int
                    and type(row_range[1]) is int
                )
            )
    if not ranges:
        return ""
    return ", ".join(
        f"{start}행" if start == end else f"{start}-{end}행"
        for start, end in ranges
    )


def _cell_preview(chunk: RagChunk) -> str:
    values: list[str] = []
    for item in chunk.evidence.items:
        if item.type != "table" or not isinstance(item.content, dict):
            continue
        rows = item.content.get("rows", [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            cells = row.get("cells", [])
            if not isinstance(cells, list):
                continue
            for cell in cells:
                if not isinstance(cell, dict):
                    continue
                text = _normalize_space(str(cell.get("text", "")))
                if len(text) >= 2:
                    values.append(_truncate(text, 80))
                if len(values) >= 4:
                    return " / ".join(_unique(values))
    return " / ".join(_unique(values))


def _has_table_evidence(chunk: RagChunk) -> bool:
    return any(item.type == "table" for item in chunk.evidence.items)


def _text_summary(text: str) -> str:
    normalized = _normalize_space(text)
    if not normalized:
        return ""
    sentence = re.split(r"(?<=[.!?。！？])\s+", normalized)[0]
    return _truncate(sentence, 300)


def _first_topic(text: str) -> str:
    for line in text.splitlines():
        normalized = _normalize_space(line)
        if normalized:
            return _truncate(normalized, 80)
    return ""


def _fallback_keywords(text: str) -> list[str]:
    ignored = {
        "table",
        "rows",
        "columns",
        "context",
        "row",
        "col",
        "chunk",
    }
    result: list[str] = []
    for token in re.findall(r"[0-9A-Za-z가-힣]{2,}", text):
        normalized = token.strip()
        if normalized.lower() in ignored or normalized in result:
            continue
        result.append(normalized)
        if len(result) >= 8:
            return result
    return result


def _column_label(columns: list[Any], column_id: Any) -> str:
    for column in columns:
        if (
            isinstance(column, dict)
            and column.get("id") == column_id
            and isinstance(column.get("text"), str)
            and column["text"].strip()
        ):
            return column["text"].strip()
    return str(column_id or "")


def _clean_question(question: str) -> str:
    normalized = _normalize_space(question)
    if not normalized:
        return ""
    if normalized.endswith(("?", "？")):
        return normalized
    return normalized.rstrip(".。") + "?"


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


def _dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _json_preview(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)
