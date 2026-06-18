from __future__ import annotations

from collections.abc import Iterable

from ...models import CommonMetadata, CommonMetadataPayload


def common_metadata(
    chunk_kind: str,
    display_format: str,
    *,
    section_path: Iterable[str] | None = None,
) -> CommonMetadataPayload:
    return CommonMetadataPayload(
        common=CommonMetadata(
            chunk_kind=chunk_kind,
            section_path=list(section_path or []),
            display_format=display_format,
        )
    )
