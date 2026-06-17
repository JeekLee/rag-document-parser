from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .backends import DocumentBackend, default_backends
from .llm import LlmConfig, chat_json as _chat_json
from .models import (
    DocumentAsset,
    Evidence,
    EvidenceUnit,
    ParseResult,
    PendingAsset,
    RagChunk,
    SourceInfo,
)
from .storage import S3Config, put_object as _put_object


@dataclass
class RagDocumentParser:
    llm: LlmConfig | None = None
    object_storage: S3Config | None = None
    backends: dict[str, DocumentBackend] | None = None

    def __post_init__(self) -> None:
        if self.llm is None:
            raise ValueError("llm is required")
        if self.object_storage is None:
            raise ValueError("object_storage is required")
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
        assets = _upload_assets(parsed.assets, self.object_storage, sha256)
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
            chunks=_enrich_chunks(_chunks_from_units(parsed.units, assets), self.llm),
            assets=assets,
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


def _upload_assets(
    assets: list[PendingAsset],
    object_storage: S3Config,
    document_sha256: str,
) -> list[DocumentAsset]:
    uploaded: list[DocumentAsset] = []
    for asset in assets:
        ext = asset.ext.lstrip(".")
        key = f"{document_sha256}/assets/{asset.id}.{ext}"
        uri = _put_object(object_storage, key, asset.data, asset.mime)
        uploaded.append(
            DocumentAsset(
                id=asset.id,
                kind=asset.kind,
                uri=uri,
                mime=asset.mime,
                ext=ext,
                sha256=hashlib.sha256(asset.data).hexdigest(),
                bytes=len(asset.data),
                metadata=dict(asset.metadata),
            )
        )
    return uploaded


def _chunks_from_units(
    units: list[EvidenceUnit],
    assets: list[DocumentAsset],
) -> list[RagChunk]:
    assets_by_id = {asset.id: asset for asset in assets}
    chunks: list[RagChunk] = []
    for unit in units:
        chunks.append(
            RagChunk(
                id=unit.id,
                type=unit.type,
                source=unit.source,
                evidence=_resolve_asset_evidence(unit.evidence, assets_by_id),
                summary="",
                metadata=dict(unit.metadata),
            )
        )
    return chunks


def _resolve_asset_evidence(
    evidence: Evidence,
    assets_by_id: dict[str, DocumentAsset],
) -> Evidence:
    if evidence.format != "asset_ref":
        return evidence
    if not isinstance(evidence.content, dict):
        raise ValueError("asset_ref evidence content must be an object")
    asset_id = evidence.content.get("asset_id")
    if not isinstance(asset_id, str):
        raise ValueError("asset_ref evidence requires asset_id")
    try:
        asset = assets_by_id[asset_id]
    except KeyError as exc:
        raise ValueError(f"asset_ref evidence points to unknown asset: {asset_id}") from exc
    return Evidence(
        kind=evidence.kind,
        format=evidence.format,
        content={
            **evidence.content,
            "uri": asset.uri,
            "mime": asset.mime,
            "ext": asset.ext,
            "sha256": asset.sha256,
            "bytes": asset.bytes,
        },
    )


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
