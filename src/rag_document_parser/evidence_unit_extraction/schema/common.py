from __future__ import annotations

from collections.abc import Iterable
from typing import TypedDict


class CommonMetadata(TypedDict):
    chunk_kind: str
    section_path: list[str]
    display_format: str


class CommonMetadataPayload(TypedDict):
    common: CommonMetadata


def common_metadata(
    chunk_kind: str,
    display_format: str,
    *,
    section_path: Iterable[str] | None = None,
) -> CommonMetadataPayload:
    return {
        "common": {
            "chunk_kind": chunk_kind,
            "section_path": list(section_path or []),
            "display_format": display_format,
        }
    }
