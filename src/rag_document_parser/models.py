from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Evidence:
    kind: str
    format: str
    content: str

    def to_dict(self) -> dict[str, str]:
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
class SourceEvidence:
    kind: str
    text: str
    section_path: list[str] = field(default_factory=list)
    headers: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    caption: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": self.kind,
            "text": self.text,
            "section_path": list(self.section_path),
        }
        if self.headers:
            payload["headers"] = list(self.headers)
        if self.rows:
            payload["rows"] = [
                {
                    "index": row["index"],
                    "cells": dict(row["cells"]),
                }
                for row in self.rows
            ]
        if self.caption is not None:
            payload["caption"] = self.caption
        return payload


@dataclass(frozen=True)
class RagChunk:
    id: str
    type: str
    source: SourceEvidence
    embedding_text: str
    evidence: Evidence
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "source": self.source.to_dict(),
            "embedding_text": self.embedding_text,
            "evidence": self.evidence.to_dict(),
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
