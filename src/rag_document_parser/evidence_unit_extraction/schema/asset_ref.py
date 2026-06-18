from __future__ import annotations

from typing import TypedDict


class AssetRefContent(TypedDict):
    asset_id: str
    caption: str | None


def asset_ref_content(asset_id: str, *, caption: str | None = None) -> AssetRefContent:
    return {"asset_id": asset_id, "caption": caption}
