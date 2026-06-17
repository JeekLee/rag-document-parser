from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Evidence:
    kind: str
    format: str
    content: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "format": self.format,
            "content": self.content,
        }


@dataclass(frozen=True)
class SourceInfo:
    sha256: str
    suffix: str
    bytes: int
    id: str | None = None
    name: str | None = None
    url: str | None = None

    def to_dict(self) -> dict[str, str | int | None]:
        return {
            "sha256": self.sha256,
            "suffix": self.suffix,
            "bytes": self.bytes,
            "id": self.id,
            "name": self.name,
            "url": self.url,
        }


@dataclass(frozen=True)
class PendingAsset:
    id: str
    kind: str
    data: bytes
    mime: str
    ext: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentAsset:
    id: str
    kind: str
    uri: str
    mime: str
    ext: str
    sha256: str
    bytes: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "uri": self.uri,
            "mime": self.mime,
            "ext": self.ext,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SourceEvidence:
    kind: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "text": self.text,
        }


@dataclass(frozen=True)
class EvidenceUnit:
    id: str
    type: str
    source: SourceEvidence
    evidence: Evidence
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "source": self.source.to_dict(),
            "evidence": self.evidence.to_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RagChunk:
    id: str
    type: str
    source: SourceEvidence
    evidence: Evidence
    summary: str
    keywords: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "source": self.source.to_dict(),
            "evidence": self.evidence.to_dict(),
            "summary": self.summary,
            "keywords": list(self.keywords),
            "questions": list(self.questions),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ParseResult:
    source: SourceInfo
    chunks: list[RagChunk]
    assets: list[DocumentAsset] = field(default_factory=list)
    quality_warnings: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.to_dict(),
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "assets": [asset.to_dict() for asset in self.assets],
            "quality_warnings": list(self.quality_warnings),
        }
