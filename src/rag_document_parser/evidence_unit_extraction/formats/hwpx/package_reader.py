from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from typing import Any
from xml.etree import ElementTree as ET

from ....models import PendingAsset
from .xml_utils import local_name as _local_name

_OPF = "http://www.idpf.org/2007/opf/"


@dataclass(frozen=True)
class HwpxPackageReader:
    """Reads HWPX package sections, BinData mappings, and image assets."""

    def section_names(self, archive: zipfile.ZipFile) -> list[str]:
        return _section_names(archive)

    def load_bin_data_map(self, archive: zipfile.ZipFile) -> dict[str, str]:
        return _load_bin_data_map(archive)

    def extract_image(
        self,
        picture: ET.Element,
        archive: zipfile.ZipFile,
        bin_data_map: dict[str, str],
        index: int,
        warnings: list[dict[str, Any]],
    ) -> tuple[str, PendingAsset] | None:
        return _extract_image(picture, archive, bin_data_map, index, warnings)


def _section_names(z: zipfile.ZipFile) -> list[str]:
    names = [name for name in z.namelist() if re.match(r"Contents/section\d+\.xml$", name)]
    return sorted(
        names,
        key=lambda name: int(re.search(r"\d+", name.rsplit("/", 1)[-1]).group()),
    )


def _load_bin_data_map(z: zipfile.ZipFile) -> dict[str, str]:
    if "Contents/content.hpf" not in z.namelist():
        return {}
    root = ET.fromstring(z.read("Contents/content.hpf"))
    result: dict[str, str] = {}
    for item in root.iter(f"{{{_OPF}}}item"):
        item_id = item.get("id", "")
        href = item.get("href", "")
        if item_id and href.startswith("BinData/"):
            result[item_id] = href
    return result


def _extract_image(
    picture: ET.Element,
    z: zipfile.ZipFile,
    bin_data_map: dict[str, str],
    index: int,
    warnings: list[dict[str, Any]],
) -> tuple[str, PendingAsset] | None:
    ref = _image_ref(picture)
    if ref is None:
        _warn_unresolved_image(warnings, "")
        return None
    href = bin_data_map.get(ref, "")
    if not href or href not in z.namelist():
        _warn_unresolved_image(warnings, ref)
        return None
    data = z.read(href)
    mime = _detect_mime(data)
    ext = _mime_to_ext(mime)
    asset_id = f"img-{index:04d}"
    return (
        asset_id,
        PendingAsset(
            id=asset_id,
            kind="image",
            data=data,
            mime=mime,
            ext=ext,
            metadata={"source_path": href},
        ),
    )


def _warn_unresolved_image(
    warnings: list[dict[str, Any]],
    ref: str,
) -> None:
    warnings.append(
        {
            "type": "hwpx_image_reference_unresolved",
            "severity": "medium",
            "ref": ref,
            "message": f"HWPX image reference could not be resolved: {ref}",
        }
    )


def _image_ref(picture: ET.Element) -> str | None:
    for element in picture.iter():
        if _local_name(element.tag) not in {"img", "image"}:
            continue
        ref = element.get("binaryItemIDRef")
        if ref:
            return ref
    return None


def _detect_mime(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] in (b"GIF8", b"GIF9"):
        return "image/gif"
    if data[:2] == b"BM":
        return "image/bmp"
    return "application/octet-stream"


def _mime_to_ext(mime: str) -> str:
    return {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/gif": "gif",
        "image/bmp": "bmp",
    }.get(mime, "bin")
