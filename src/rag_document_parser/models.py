from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvidenceItem:
    type: str
    content: Any
    format: str | None = None
    source_unit_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": self.type,
            "content": self.content,
            "source_unit_ids": list(self.source_unit_ids),
            "metadata": dict(self.metadata),
        }
        if self.format is not None:
            payload["format"] = self.format
        return payload


@dataclass(frozen=True)
class Evidence:
    items: list[EvidenceItem] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"items": [item.to_dict() for item in self.items]}


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
    format: str
    source: SourceEvidence
    content: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "format": self.format,
            "source": self.source.to_dict(),
            "content": self.content,
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
    units: list[EvidenceUnit]
    assets: list[DocumentAsset] = field(default_factory=list)
    quality_warnings: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.to_dict(),
            "units": [unit.to_dict() for unit in self.units],
            "assets": [asset.to_dict() for asset in self.assets],
            "quality_warnings": list(self.quality_warnings),
        }
