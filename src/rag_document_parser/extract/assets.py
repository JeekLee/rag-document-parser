from __future__ import annotations

import hashlib
from typing import Any

from ..models import DocumentAsset, Evidence, EvidenceUnit, PendingAsset
from ..storage import S3Config, put_object as _put_object


def upload_assets(
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


def resolve_units(
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
                evidence=resolve_asset_evidence(unit.evidence, assets_by_id),
                metadata=dict(unit.metadata),
            )
        )
    return resolved


def resolve_asset_evidence(
    evidence: Evidence,
    assets_by_id: dict[str, DocumentAsset],
) -> Evidence:
    if evidence.format != "asset_ref":
        return Evidence(
            kind=evidence.kind,
            format=evidence.format,
            content=resolve_asset_refs_in_value(evidence.content, assets_by_id),
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


def resolve_asset_refs_in_value(
    value: Any,
    assets_by_id: dict[str, DocumentAsset],
) -> Any:
    if isinstance(value, list):
        return [resolve_asset_refs_in_value(item, assets_by_id) for item in value]
    if not isinstance(value, dict):
        return value
    nested = nested_evidence(value)
    if nested is not None:
        return resolve_asset_evidence(nested, assets_by_id).to_dict()
    return {
        key: resolve_asset_refs_in_value(nested_value, assets_by_id)
        for key, nested_value in value.items()
    }


def nested_evidence(value: dict[str, Any]) -> Evidence | None:
    kind = value.get("kind")
    fmt = value.get("format")
    if not isinstance(kind, str) or not isinstance(fmt, str):
        return None
    if "content" not in value:
        return None
    return Evidence(kind=kind, format=fmt, content=value["content"])
