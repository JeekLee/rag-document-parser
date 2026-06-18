from __future__ import annotations

from typing import Protocol

from ..models import ParsedDocument


class DocumentBackend(Protocol):
    def parse(self, data: bytes, suffix: str) -> ParsedDocument:
        """Parse raw document bytes into source-preserving evidence units."""
        ...
