from __future__ import annotations

from typing import Protocol

from ..models import RagChunk


class Enricher(Protocol):
    def enrich(self, chunks: list[RagChunk]) -> list[RagChunk]:
        """Add summary, keyword, and question metadata to chunks."""
        ...
