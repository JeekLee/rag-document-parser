from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Evidence:
    format: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {
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
class SourcePointer:
    sha256: str
    char_start: int | None = None
    char_end: int | None = None
    byte_start: int | None = None
    byte_end: int | None = None
    page: int | None = None
    bbox: list[float] | None = None
    section_path: list[str] = field(default_factory=list)
    block_id: str | None = None
    table_id: str | None = None
    row_range: tuple[int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "byte_start": self.byte_start,
            "byte_end": self.byte_end,
            "page": self.page,
            "bbox": list(self.bbox) if self.bbox else None,
            "section_path": list(self.section_path),
            "block_id": self.block_id,
            "table_id": self.table_id,
            "row_range": list(self.row_range) if self.row_range else None,
        }


@dataclass(frozen=True)
class RagChunk:
    id: str
    type: str
    source: str
    embedding_text: str
    evidence: Evidence
    source_pointer: SourcePointer
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "source": self.source,
            "embedding_text": self.embedding_text,
            "evidence": self.evidence.to_dict(),
            "source_pointer": self.source_pointer.to_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ParseResult:
    source: SourceInfo
    chunks: list[RagChunk]
    quality_warnings: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.to_dict(),
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "quality_warnings": list(self.quality_warnings),
        }
