from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from .backends import DocumentBackend, default_backends
from .models import (
    DocumentAsset,
    Evidence,
    EvidenceUnit,
    ParseResult,
    PendingAsset,
    SourceInfo,
)
from .storage import S3Config, put_object as _put_object


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
                    _normalize_suffix(suffix): backend
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
        data = source.encode() if isinstance(source, str) else bytes(source)
        normalized_suffix = _normalize_suffix(suffix)
        backend = self._backend_for(normalized_suffix)
        parsed = backend.parse(data, normalized_suffix)
        sha256 = hashlib.sha256(data).hexdigest()
        assets = _upload_assets(parsed.assets, self.object_storage, sha256)
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
            units=_resolve_units(parsed.units, assets),
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


def _upload_assets(
    assets: list[PendingAsset],
    object_storage: S3Config,
    document_sha256: str,
) -> list[DocumentAsset]:
    uploaded: list[DocumentAsset] = []
    for asset in assets:
        ext = asset.ext.lstrip(".")
        key = f"{document_sha256}/assets/{asset.id}.{ext}"
        uri = _put_object(object_storage, key, asset.data, asset.mime)
        uploaded.append(
            DocumentAsset(
                id=asset.id,
                kind=asset.kind,
                uri=uri,
                mime=asset.mime,
                ext=ext,
                sha256=hashlib.sha256(asset.data).hexdigest(),
                bytes=len(asset.data),
                metadata=dict(asset.metadata),
            )
        )
    return uploaded


def _resolve_units(
    units: list[EvidenceUnit],
    assets: list[DocumentAsset],
) -> list[EvidenceUnit]:
    assets_by_id = {asset.id: asset for asset in assets}
    resolved: list[EvidenceUnit] = []
    for unit in units:
        resolved.append(
            EvidenceUnit(
                id=unit.id,
                type=unit.type,
                source=unit.source,
                evidence=_resolve_asset_evidence(unit.evidence, assets_by_id),
                metadata=dict(unit.metadata),
            )
        )
    return resolved


def _resolve_asset_evidence(
    evidence: Evidence,
    assets_by_id: dict[str, DocumentAsset],
) -> Evidence:
    if evidence.format != "asset_ref":
        return Evidence(
            kind=evidence.kind,
            format=evidence.format,
            content=_resolve_asset_refs_in_value(evidence.content, assets_by_id),
        )
    if not isinstance(evidence.content, dict):
        raise ValueError("asset_ref evidence content must be an object")
    asset_id = evidence.content.get("asset_id")
    if not isinstance(asset_id, str):
        raise ValueError("asset_ref evidence requires asset_id")
    try:
        asset = assets_by_id[asset_id]
    except KeyError as exc:
        raise ValueError(f"asset_ref evidence points to unknown asset: {asset_id}") from exc
    return Evidence(
        kind=evidence.kind,
        format=evidence.format,
        content={
            **evidence.content,
            "uri": asset.uri,
            "mime": asset.mime,
            "ext": asset.ext,
            "sha256": asset.sha256,
            "bytes": asset.bytes,
        },
    )


def _resolve_asset_refs_in_value(
    value: Any,
    assets_by_id: dict[str, DocumentAsset],
) -> Any:
    if isinstance(value, list):
        return [_resolve_asset_refs_in_value(item, assets_by_id) for item in value]
    if not isinstance(value, dict):
        return value
    nested = _nested_evidence(value)
    if nested is not None:
        return _resolve_asset_evidence(nested, assets_by_id).to_dict()
    return {
        key: _resolve_asset_refs_in_value(nested_value, assets_by_id)
        for key, nested_value in value.items()
    }


def _nested_evidence(value: dict[str, Any]) -> Evidence | None:
    kind = value.get("kind")
    fmt = value.get("format")
    if not isinstance(kind, str) or not isinstance(fmt, str):
        return None
    if "content" not in value:
        return None
    return Evidence(kind=kind, format=fmt, content=value["content"])


def _normalize_suffix(suffix: str) -> str:
    normalized = suffix.lower()
    if normalized.startswith("."):
        return normalized
    return f".{normalized}"
