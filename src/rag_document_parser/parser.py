from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .backends import DocumentBackend, default_backends
from .llm import LlmConfig, chat_json as _chat_json
from .models import ParseResult, RagChunk, SourceInfo


@dataclass
class RagDocumentParser:
    llm: LlmConfig | None = None
    backends: dict[str, DocumentBackend] | None = None

    def __post_init__(self) -> None:
        if self.llm is None:
            raise ValueError("llm is required")
        backends = default_backends()
        if self.backends:
            backends.update(
                {
                    _normalize_suffix(suffix): backend
                    for suffix, backend in self.backends.items()
                }
            )
        self._backends = backends

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
        normalized_suffix = _normalize_suffix(suffix)
        backend = self._backend_for(normalized_suffix)
        parsed = backend.parse(data, normalized_suffix)
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
            chunks=_enrich_chunks(parsed.chunks, self.llm),
            quality_warnings=list(parsed.quality_warnings),
        )

    def _backend_for(self, suffix: str) -> DocumentBackend:
        try:
            return self._backends[suffix]
        except KeyError as exc:
            supported = ", ".join(sorted(self._backends))
            raise ValueError(
                f"Unsupported format: {suffix!r} (supported: {supported})"
            ) from exc


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


def _normalize_suffix(suffix: str) -> str:
    normalized = suffix.lower()
    if normalized.startswith("."):
        return normalized
    return f".{normalized}"
