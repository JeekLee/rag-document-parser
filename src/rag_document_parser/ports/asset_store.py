from __future__ import annotations

from typing import Protocol


class AssetStore(Protocol):
    """Application port for durable document-asset storage."""

    def put(self, key: str, data: bytes, content_type: str) -> str:
        """Store an asset and return its canonical URI."""
        ...
