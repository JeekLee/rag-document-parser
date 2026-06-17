from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..models import EvidenceUnit, PendingAsset


@dataclass(frozen=True)
class ParsedDocument:
    units: list[EvidenceUnit]
    assets: list[PendingAsset] = field(default_factory=list)
    quality_warnings: list[dict[str, Any]] = field(default_factory=list)


class DocumentBackend(Protocol):
    def parse(self, data: bytes, suffix: str) -> ParsedDocument:
        """Parse raw document bytes into source-preserving evidence units."""
        ...
