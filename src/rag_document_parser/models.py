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
    page: int | None = None
    section_path: list[str] = field(default_factory=list)
    block_id: str | None = None
    table_id: str | None = None
    row_range: tuple[int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "page": self.page,
            "section_path": list(self.section_path),
            "block_id": self.block_id,
            "table_id": self.table_id,
            "row_range": list(self.row_range) if self.row_range else None,
        }


@dataclass(frozen=True)
class RagChunk:
    id: str
    type: str
    llm_text: str
    display: Evidence
    source: SourcePointer
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "llm_text": self.llm_text,
            "display": self.display.to_dict(),
            "source": self.source.to_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ParseResult:
    source: SourceInfo
    preview_markdown: str
    chunks: list[RagChunk]
    quality_warnings: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.to_dict(),
            "preview_markdown": self.preview_markdown,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "quality_warnings": list(self.quality_warnings),
        }
