from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

from ....models import PendingAsset
from .geometry import _bbox_area, _bbox_in_cell
from .models import PdfImage as _PdfImage
from .models import Segment as _Segment

_MIN_IMAGE_AREA_PT2 = 2500


@dataclass(frozen=True)
class PdfEmbeddedImageExtractor:
    """Extracts embedded PDF images and assigns them to page or table cells."""

    def reader(self, data: bytes) -> object:
        return _pdf_reader(data)

    def extract_page(
        self,
        data: bytes,
        page_idx: int,
        page: object,
        start_idx: int = 1,
        reader: object | None = None,
    ) -> list[tuple[float, _PdfImage]]:
        return _extract_page_images(data, page_idx, page, start_idx, reader)

    def split_segments_and_cell_children(
        self,
        items: list[tuple[float, object]],
        tables: list[object],
        assets: list[PendingAsset],
        page_idx: int,
        warnings: list[dict[str, Any]],
    ) -> tuple[
        list[_Segment],
        dict[int, dict[tuple[int, int], list[dict[str, object]]]],
    ]:
        return _image_segments_and_cell_children(
            items,
            tables,
            assets,
            page_idx,
            warnings,
        )

    def merge_cell_children(
        self,
        target: dict[int, dict[tuple[int, int], list[dict[str, object]]]],
        source: dict[int, dict[tuple[int, int], list[dict[str, object]]]],
    ) -> None:
        _merge_cell_children(target, source)

    def append_asset(
        self,
        assets: list[PendingAsset],
        item: _PdfImage,
        page_idx: int,
    ) -> str:
        return _append_pdf_image_asset(assets, item, page_idx)


def _pdf_reader(data: bytes) -> object:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "PDF image extraction requires pypdf. Install the PDF extraction "
            "dependencies before parsing PDF images."
        ) from exc
    return PdfReader(io.BytesIO(data))


def _append_pdf_image_asset(
    assets: list[PendingAsset],
    item: object,
    page_idx: int,
    *,
    metadata: dict[str, Any] | None = None,
) -> str:
    asset_id = f"img-{len(assets) + 1:04d}"
    asset_metadata = dict(getattr(item, "metadata", {}) or {})
    if metadata:
        asset_metadata.update(metadata)
    asset_metadata.setdefault("page", page_idx + 1)
    assets.append(
        PendingAsset(
            id=asset_id,
            kind="image",
            data=bytes(getattr(item, "data")),
            mime=str(getattr(item, "mime")),
            ext=str(getattr(item, "ext")),
            metadata=asset_metadata,
        )
    )
    return asset_id


def _image_segments_and_cell_children(
    img_items: list[tuple[float, object]],
    tables: list[object],
    assets: list[PendingAsset],
    page_idx: int,
    warnings: list[dict[str, Any]],
) -> tuple[list[_Segment], dict[int, dict[tuple[int, int], list[dict[str, object]]]]]:
    image_segments: list[_Segment] = []
    cell_children: dict[int, dict[tuple[int, int], list[dict[str, object]]]] = {}
    for top, item in sorted(img_items, key=lambda candidate: candidate[0]):
        metadata = dict(getattr(item, "metadata", {}) or {})
        bbox = _metadata_bbox(metadata)
        cell_ref = _containing_table_cell(tables, bbox) if bbox is not None else None
        if cell_ref is not None:
            table_idx, row_idx, col_idx = cell_ref
            asset_id = _append_pdf_image_asset(
                assets,
                item,
                page_idx,
                metadata={
                    "source": "pdf_table_cell_image",
                    "bbox": bbox,
                    "confidence": "medium",
                },
            )
            child = {
                "type": "image",
                "format": "asset_ref",
                "content": {"asset_id": asset_id, "caption": None},
                "metadata": {
                    "source": "pdf_table_cell_image",
                    "bbox": bbox,
                    "confidence": "medium",
                },
            }
            cell_children.setdefault(table_idx, {}).setdefault((row_idx, col_idx), []).append(
                child
            )
            warnings.append(
                {
                    "type": "pdf_table_cell_image_inferred",
                    "severity": "low",
                    "page": page_idx + 1,
                    "message": (
                        "PDF image was assigned to a table cell by bounding-box "
                        "containment."
                    ),
                }
            )
            continue

        asset_id = _append_pdf_image_asset(assets, item, page_idx)
        image_segments.append(
            _Segment(
                top=float(top),
                bottom=float(top) + 1.0,
                kind="image",
                payload={"asset_id": asset_id, "caption": None},
                page=page_idx + 1,
                metadata={"confidence": "medium"},
            )
        )
    return image_segments, cell_children


def _merge_cell_children(
    target: dict[int, dict[tuple[int, int], list[dict[str, object]]]],
    source: dict[int, dict[tuple[int, int], list[dict[str, object]]]],
) -> None:
    for table_idx, cell_map in source.items():
        target_cell_map = target.setdefault(table_idx, {})
        for cell_ref, children in cell_map.items():
            target_cell_map.setdefault(cell_ref, []).extend(children)


def _metadata_bbox(metadata: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = metadata.get("bbox")
    if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
        return None
    try:
        return tuple(float(value) for value in bbox)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def _containing_table_cell(
    tables: list[object],
    bbox: tuple[float, float, float, float],
) -> tuple[int, int, int] | None:
    best: tuple[int, int, int, float] | None = None
    for table_idx, table in enumerate(tables):
        for row_idx, row in enumerate(getattr(table, "rows", [])):
            for col_idx, cell_bbox in enumerate(getattr(row, "cells", []) or []):
                if cell_bbox is None:
                    continue
                if not _bbox_in_cell(bbox, cell_bbox, tol=3.0):
                    continue
                area = _bbox_area(cell_bbox)
                if best is None or area < best[3]:
                    best = (table_idx, row_idx, col_idx, area)
    if best is None:
        return None
    return best[:3]


def _extract_page_images(
    data: bytes,
    page_idx: int,
    page: object,
    start_idx: int = 1,
    reader: object | None = None,
) -> list[tuple[float, _PdfImage]]:
    if reader is None:
        reader = _pdf_reader(data)
    if page_idx >= len(reader.pages):
        return []

    pypdf_by_name: dict[str, object] = {}
    for image_file in reader.pages[page_idx].images:
        if image_file.image is None:
            continue
        raw_name = image_file.name.lstrip("/")
        stem = raw_name.rsplit(".", 1)[0] if "." in raw_name else raw_name
        pypdf_by_name[raw_name] = image_file.image
        pypdf_by_name[stem] = image_file.image

    page_width = float(page.width)
    page_height = float(page.height)
    results: list[tuple[float, _PdfImage]] = []
    image_index = start_idx
    for image_info in getattr(page, "images", []):
        image_width = float(image_info.get("x1", 0.0) - image_info.get("x0", 0.0))
        image_height = float(image_info.get("y1", 0.0) - image_info.get("y0", 0.0))
        if (
            page_width
            and page_height
            and image_width / page_width > 0.8
            and image_height / page_height > 0.8
        ):
            continue
        if image_width * image_height < _MIN_IMAGE_AREA_PT2:
            continue

        name = str(image_info.get("name", "")).lstrip("/")
        image = pypdf_by_name.get(name)
        if image is None:
            continue
        bbox = _image_info_bbox(image_info)
        item = _pil_to_pdf_image(
            image,
            metadata={
                "source": "embedded",
                "name": name,
                "index_hint": image_index,
                "bbox": bbox,
            },
        )
        results.append((float(image_info.get("top", 0.0)), item))
        image_index += 1
    return sorted(results, key=lambda item: item[0])


def _image_info_bbox(image_info: dict[str, object]) -> tuple[float, float, float, float]:
    x0 = float(image_info.get("x0", 0.0))
    x1 = float(image_info.get("x1", x0))
    top = float(image_info.get("top", image_info.get("y0", 0.0)))
    bottom = float(image_info.get("bottom", image_info.get("y1", top)))
    return (min(x0, x1), min(top, bottom), max(x0, x1), max(top, bottom))


def _pil_to_pdf_image(pil_image: object, metadata: dict[str, Any]) -> _PdfImage:
    image_format = (getattr(pil_image, "format", None) or "PNG").upper()
    if image_format == "JPEG":
        ext = "jpg"
        mime = "image/jpeg"
        save_format = "JPEG"
        mode = "RGB"
    else:
        ext = "png"
        mime = "image/png"
        save_format = "PNG"
        mode = "RGBA"

    if getattr(pil_image, "mode", None) != mode:
        pil_image = pil_image.convert(mode)

    buffer = io.BytesIO()
    pil_image.save(buffer, format=save_format)
    return _PdfImage(
        data=buffer.getvalue(),
        mime=mime,
        ext=ext,
        metadata=metadata,
    )
