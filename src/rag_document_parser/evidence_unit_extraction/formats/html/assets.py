from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass

from bs4.element import Tag

from ....models import PendingAsset
from .state import HtmlParseState
from .text import HtmlTextExtractor
from .text import tag_name as _tag_name

_DATA_URI_RE = re.compile(r"^data:([^;,]+);base64,(.*)$", re.IGNORECASE | re.DOTALL)
_SUPPORTED_IMAGE_MIME = {
    "image/gif": "gif",
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
}
_TEXT_EXTRACTOR = HtmlTextExtractor()
_text_with_links = _TEXT_EXTRACTOR.text
_HtmlParseState = HtmlParseState


@dataclass(frozen=True)
class HtmlImageExtractor:
    """Extracts embedded HTML image assets and their source descriptions."""

    def reference(
        self,
        image: Tag,
        state: HtmlParseState,
    ) -> tuple[str, str | None] | None:
        return _image_asset_ref(image, state)

    def figure_caption(self, figure: Tag) -> str | None:
        return _figure_caption(figure)

    def caption_from_context(self, image: Tag) -> str | None:
        return _image_caption_from_context(image)

    def source_text(
        self,
        section_path: list[str],
        asset_id: str,
        caption: str | None,
        alt: str | None,
    ) -> str:
        return _image_source_text(section_path, asset_id, caption, alt)


def _figure_caption(figure: Tag) -> str | None:
    for child in figure.children:
        if isinstance(child, Tag) and _tag_name(child) == "figcaption":
            text = _text_with_links(child)
            return text or None
    return None


def _image_caption_from_context(image: Tag) -> str | None:
    figure = image.find_parent("figure")
    if not isinstance(figure, Tag):
        return None
    return _figure_caption(figure)


def _image_asset_ref(
    image: Tag,
    state: _HtmlParseState,
) -> tuple[str, str | None] | None:
    src = str(image.get("src") or "").strip()
    alt = str(image.get("alt") or "").strip() or None
    if not src:
        state.quality_warnings.append(
            {
                "type": "html_image_external_reference",
                "severity": "medium",
                "src": src,
                "message": "HTML image source did not contain embedded bytes.",
            }
        )
        return None
    match = _DATA_URI_RE.match(src)
    if match is None:
        state.quality_warnings.append(
            {
                "type": "html_image_external_reference",
                "severity": "medium",
                "src": src,
                "message": "HTML image references external bytes and was not fetched.",
            }
        )
        return None
    mime = match.group(1).lower()
    ext = _SUPPORTED_IMAGE_MIME.get(mime)
    if ext is None:
        state.quality_warnings.append(
            {
                "type": "html_image_mime_unsupported",
                "severity": "medium",
                "mime": mime,
                "message": f"HTML embedded image MIME type is unsupported: {mime}",
            }
        )
        return None
    try:
        data = base64.b64decode(match.group(2).strip(), validate=True)
    except (binascii.Error, ValueError):
        state.quality_warnings.append(
            {
                "type": "html_image_data_uri_invalid",
                "severity": "medium",
                "mime": mime,
                "message": "HTML embedded image data URI could not be decoded.",
            }
        )
        return None

    asset_id = state.next_image_id()
    state.assets.append(
        PendingAsset(
            id=asset_id,
            kind="image",
            data=data,
            mime=mime,
            ext=ext,
            metadata={"source": "html_data_uri"},
        )
    )
    return asset_id, alt


def _image_source_text(
    section_path: list[str],
    asset_id: str,
    caption: str | None,
    alt: str | None,
) -> str:
    lines: list[str] = []
    if section_path:
        lines.append(f"section: {' > '.join(section_path)}")
    lines.append(f"image: {asset_id}")
    if caption:
        lines.append(f"caption: {caption}")
    if alt:
        lines.append(f"alt: {alt}")
    return "\n".join(lines)
