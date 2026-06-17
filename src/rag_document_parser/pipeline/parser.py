from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..extract.assets import resolve_units, upload_assets
from ..extract.backend import DocumentBackend
from ..extract.registry import default_backends
from ..input import normalize_source, normalize_suffix
from ..models import ParseResult, SourceInfo
from ..storage import S3Config


@dataclass
class RagDocumentParser:
    object_storage: S3Config | None = None
    backends: dict[str, DocumentBackend] | None = None

    def __post_init__(self) -> None:
        if self.object_storage is None:
            raise ValueError("object_storage is required")
        backends = default_backends()
        if self.backends:
            backends.update(
                {
                    normalize_suffix(suffix): backend
                    for suffix, backend in self.backends.items()
                }
            )
        self._backends = backends

    def parse(
        self,
        source: bytes | str,
        *,
        suffix: str,
        source_id: str | None = None,
        source_name: str | None = None,
        source_url: str | None = None,
    ) -> ParseResult:
        data = normalize_source(source)
        normalized_suffix = normalize_suffix(suffix)
        backend = self._backend_for(normalized_suffix)
        parsed = backend.parse(data, normalized_suffix)
        sha256 = hashlib.sha256(data).hexdigest()
        assets = upload_assets(parsed.assets, self.object_storage, sha256)
        source_info = SourceInfo(
            sha256=sha256,
            suffix=normalized_suffix,
            bytes=len(data),
            id=source_id,
            name=source_name,
            url=source_url,
        )
        return ParseResult(
            source=source_info,
            units=resolve_units(parsed.units, assets),
            assets=assets,
            quality_warnings=list(parsed.quality_warnings),
        )

    def _backend_for(self, suffix: str) -> DocumentBackend:
        try:
            return self._backends[suffix]
        except KeyError as exc:
            supported = ", ".join(sorted(self._backends))
            raise ValueError(
                f"Unsupported format: {suffix!r} (supported: {supported})"
            ) from exc
