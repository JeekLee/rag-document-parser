from __future__ import annotations

from ...models import AssetRefContent


def asset_ref_content(asset_id: str, *, caption: str | None = None) -> AssetRefContent:
    return AssetRefContent(asset_id=asset_id, caption=caption)
