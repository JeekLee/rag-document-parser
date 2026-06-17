from __future__ import annotations

from typing import Protocol

from ..models import EvidenceUnit, RagChunk


class Chunker(Protocol):
    def chunk(self, units: list[EvidenceUnit]) -> list[RagChunk]:
        """Convert extracted evidence units into retrieval chunks."""
        ...
