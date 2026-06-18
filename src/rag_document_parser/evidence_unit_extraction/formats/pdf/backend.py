from __future__ import annotations

import io
import base64
import json
import re
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib import request

from ....llm import LlmConfig, chat_completions_url
from ....models import EvidenceUnit, PendingAsset, SourceEvidence
from ...backend import ParsedDocument
from ...schema import (
    structured_diagram as _structured_diagram_content,
    structured_table as _structured_table_content,
    table_column,
    table_row,
)
from ...table_source import (
    build_column_source_labels as _build_column_source_labels,
    common_semantic_header_prefix as _common_semantic_header_prefix,
    is_semantic_column_label as _is_semantic_column_label,
    semantic_column_group_label as _semantic_column_group_label,
)


_PAGE_NUM_RE = re.compile(r"(?m)^\s*(?:-\s*)?\d+\s*(?:-\s*)?$")
_CJK = re.compile(r"[가-힣一-鿿㐀-䶿]")
_MIN_IMAGE_AREA_PT2 = 2500
_DEFAULT_RENDER_SCALE = 2.0
_SCANNED_OCR_RENDER_SCALE = 3.0
_PdfDiagramShape = tuple[float, float, float, float, str]


@dataclass(frozen=True)
class _PdfImage:
    data: bytes
    mime: str
    ext: str
    is_diagram: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _Segment:
    top: float
    bottom: float
    kind: str
    payload: Any
    page: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _NestedResolution:
    suppressed: set[int]
    children: dict[int, dict[tuple[int, int], list[int]]]


@dataclass(frozen=True)
class _TableCellSpans:
    spans: dict[tuple[int, int], tuple[int, int]]
    covered: set[tuple[int, int]]


class _OcrResults(dict[int, str]):
    def __init__(self) -> None:
        super().__init__()
        self.failed_pages: list[dict[str, object]] = []


PdfOcrConfig = LlmConfig


@dataclass
class PdfBackend:
    supported_suffixes = (".pdf",)
    max_ocr_workers: int = 4
    ocr_fn: Callable[[bytes, int], str] | None = None
    ocr_llm: PdfOcrConfig | None = None

    def parse(self, data: bytes, suffix: str) -> ParsedDocument:
        try:
            import pdfplumber
        except ImportError as exc:
            raise NotImplementedError(
                "PDF extraction requires pdfplumber. Install the PDF extraction "
                "dependencies before parsing .pdf files."
            ) from exc

        assets: list[PendingAsset] = []
        warnings: list[dict[str, Any]] = []

        with pdfplumber.open(io.BytesIO(data)) as pdf:
            page_segments: list[list[_Segment]] = [[] for _ in pdf.pages]
            scanned: list[tuple[int, bytes]] = []
            pdf_reader: object | None = None

            for page_idx, page in enumerate(pdf.pages):
                ocr_fallback_reason = _ocr_fallback_reason(
                    page,
                    allow_degraded_native=(
                        self.ocr_fn is not None or self.ocr_llm is not None
                    ),
                )
                if ocr_fallback_reason is not None:
                    png = _render_scanned_page_for_ocr(
                        data,
                        page_idx,
                        page,
                        warnings,
                    ) if self.ocr_fn is not None or self.ocr_llm is not None else b""
                    scanned.append((page_idx, png))
                    if ocr_fallback_reason == "degraded_native_text":
                        warnings.append(
                            {
                                "type": "pdf_native_text_ocr_fallback",
                                "severity": "low",
                                "page": page_idx + 1,
                                "message": (
                                    "Native PDF text looked degraded; OCR fallback "
                                    "was used."
                                ),
                            }
                        )
                    continue

                tables = _find_tables(page, warnings, page_idx)
                img_items: list[tuple[float, object]] = []
                if getattr(page, "images", None):
                    try:
                        if pdf_reader is None:
                            pdf_reader = _pdf_reader(data)
                        img_items = _extract_page_images(
                            data,
                            page_idx,
                            page,
                            start_idx=len(assets) + 1,
                            reader=pdf_reader,
                        )
                    except ImportError as exc:
                        warnings.append(
                            {
                                "type": "pdf_images_skipped_missing_dependency",
                                "severity": "medium",
                                "page": page_idx + 1,
                                "message": str(exc),
                            }
                        )
                    except Exception as exc:
                        warnings.append(
                            {
                                "type": "pdf_images_skipped",
                                "severity": "medium",
                                "page": page_idx + 1,
                                "message": str(exc),
                            }
                        )

                table_bboxes = [table.bbox for table in tables]
                diagram_segments = _diagram_segments(
                    data,
                    page,
                    page_idx,
                    table_bboxes,
                    assets,
                    warnings,
                )
                image_segments, cell_image_children = _image_segments_and_cell_children(
                    img_items,
                    tables,
                    assets,
                    page_idx,
                    warnings,
                )
                _merge_cell_children(
                    cell_image_children,
                    _table_cell_diagram_children(
                        data,
                        page,
                        page_idx,
                        tables,
                        _resolve_nested_tables(tables),
                        assets,
                        warnings,
                    ),
                )

                page_segments[page_idx].extend(
                    _page_segments_ordered(
                        page,
                        page_idx + 1,
                        tables,
                        image_segments,
                        diagram_segments,
                        cell_image_children,
                    )
                )

            ocr_by_page = _ocr_pages(
                scanned,
                data,
                self.max_ocr_workers,
                self.ocr_fn,
                self.ocr_llm,
            )
            for page_idx, text in ocr_by_page.items():
                cleaned = _clean_text(text)
                if cleaned:
                    segments, ocr_parse_warnings = _ocr_text_segments(
                        cleaned,
                        page_idx + 1,
                    )
                    page_segments[page_idx].extend(segments)
                    warnings.extend(ocr_parse_warnings)
            warnings.extend(_ocr_warnings(ocr_by_page.failed_pages))

        return ParsedDocument(
            units=_segments_to_units(
                _expand_text_segments(
                    _merge_continuation_tables(page_segments)
                )
            ),
            assets=assets,
            quality_warnings=warnings,
        )


def _pdf_reader(data: bytes) -> object:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "PDF image extraction requires pypdf. Install the PDF extraction "
            "dependencies before parsing PDF images."
        ) from exc
    return PdfReader(io.BytesIO(data))


def _render_scanned_page_for_ocr(
    data: bytes,
    page_idx: int,
    page: object,
    warnings: list[dict[str, Any]],
) -> bytes:
    try:
        return _render_page_to_png(
            data,
            page_idx,
            (0.0, 0.0, float(page.width), float(page.height)),
            scale=_SCANNED_OCR_RENDER_SCALE,
        )
    except ImportError:
        return b""
    except Exception as exc:
        warnings.append(
            {
                "type": "pdf_scanned_render_failed",
                "severity": "medium",
                "page": page_idx + 1,
                "message": str(exc),
            }
        )
        return b""


def _find_tables(page: object, warnings: list[dict[str, Any]], page_idx: int) -> list[object]:
    try:
        return list(page.find_tables())
    except Exception as exc:
        warnings.append(
            {
                "type": "pdf_tables_skipped",
                "severity": "medium",
                "page": page_idx + 1,
                "message": str(exc),
            }
        )
        return []


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


def _table_cell_diagram_children(
    data: bytes,
    page: object,
    page_idx: int,
    tables: list[object],
    nested: _NestedResolution,
    assets: list[PendingAsset],
    warnings: list[dict[str, Any]],
) -> dict[int, dict[tuple[int, int], list[dict[str, object]]]]:
    cell_children: dict[int, dict[tuple[int, int], list[dict[str, object]]]] = {}
    page_shapes = _pdf_page_diagram_shapes(page)
    for table_idx, table in enumerate(tables):
        nested_children = nested.children.get(table_idx, {})
        for row_idx, row in enumerate(getattr(table, "rows", [])):
            for col_idx, cell_bbox in enumerate(getattr(row, "cells", []) or []):
                if cell_bbox is None:
                    continue
                nested_table_bboxes = [
                    tables[child_idx].bbox
                    for child_idx in nested_children.get((row_idx, col_idx), [])
                ]
                for _top, bbox in _detect_diagram_bboxes(
                    page,
                    nested_table_bboxes,
                    container_bbox=cell_bbox,
                    page_shapes=page_shapes,
                ):
                    child = _table_cell_diagram_child(
                        data,
                        page,
                        page_idx,
                        bbox,
                        assets,
                        warnings,
                    )
                    if child is not None:
                        cell_children.setdefault(table_idx, {}).setdefault(
                            (row_idx, col_idx),
                            [],
                        ).append(child)
    return cell_children


def _table_cell_diagram_child(
    data: bytes,
    page: object,
    page_idx: int,
    bbox: tuple[float, float, float, float],
    assets: list[PendingAsset],
    warnings: list[dict[str, Any]],
) -> dict[str, object] | None:
    structured = _structured_diagram_from_pdf(page, bbox)
    source_text = _diagram_source_text(structured)
    if source_text and _should_skip_pdf_diagram(structured):
        return None

    asset_id: str | None = None
    try:
        asset_id = _append_pdf_image_asset(
            assets,
            _PdfImage(
                data=_render_page_to_png(data, page_idx, bbox),
                mime="image/png",
                ext="png",
                is_diagram=True,
                metadata={
                    "source": "table_cell_diagram_fallback",
                    "bbox": bbox,
                    "is_diagram": True,
                },
            ),
            page_idx,
        )
    except ImportError as exc:
        warnings.append(
            {
                "type": "pdf_table_cell_diagram_render_missing_dependency",
                "severity": "medium",
                "page": page_idx + 1,
                "message": str(exc),
            }
        )
    except Exception as exc:
        warnings.append(
            {
                "type": "pdf_table_cell_diagram_render_failed",
                "severity": "medium",
                "page": page_idx + 1,
                "message": str(exc),
            }
        )

    if source_text:
        structured["confidence"] = "medium"
        if asset_id is not None:
            structured["asset_id"] = asset_id
        warnings.append(
            {
                "type": "pdf_table_cell_diagram_inferred",
                "severity": "low",
                "page": page_idx + 1,
                "message": (
                    "PDF vector diagram was assigned to a table cell by "
                    "bounding-box containment."
                ),
            }
        )
        return {
            "type": "diagram",
            "format": "structured_diagram",
            "content": structured,
            "metadata": {
                "source": "pdf_table_cell_diagram",
                "bbox": bbox,
                "confidence": "medium",
            },
        }

    if asset_id is None:
        warnings.append(
            {
                "type": "pdf_table_cell_diagram_structuring_failed",
                "severity": "medium",
                "page": page_idx + 1,
                "message": (
                    "PDF table cell diagram structure could not be inferred "
                    "and no fallback image was available."
                ),
            }
        )
        return None

    warnings.append(
        {
            "type": "pdf_table_cell_diagram_structuring_failed",
            "severity": "medium",
            "page": page_idx + 1,
            "message": (
                "PDF table cell diagram was preserved as a fallback image "
                "because vector structure could not be inferred."
            ),
        }
    )
    return {
        "type": "diagram",
        "format": "structured_diagram",
        "content": {
            "caption": None,
            "nodes": [],
            "edges": [],
            "connectors": [],
            "mermaid": None,
            "asset_id": asset_id,
            "confidence": "low",
        },
        "metadata": {
            "source": "pdf_table_cell_diagram",
            "bbox": bbox,
            "confidence": "low",
        },
    }


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


def _diagram_segments(
    data: bytes,
    page: object,
    page_idx: int,
    table_bboxes: list[tuple[float, float, float, float]],
    assets: list[PendingAsset],
    warnings: list[dict[str, Any]],
) -> list[_Segment]:
    segments: list[_Segment] = []
    for top, bbox in _detect_diagram_bboxes(page, table_bboxes):
        structured = _structured_diagram_from_pdf(page, bbox)
        source_text = _diagram_source_text(structured)
        if source_text and _should_skip_pdf_diagram(structured):
            continue
        asset_id: str | None = None
        try:
            asset_id = _append_pdf_image_asset(
                assets,
                _PdfImage(
                    data=_render_page_to_png(data, page_idx, bbox),
                    mime="image/png",
                    ext="png",
                    is_diagram=True,
                    metadata={
                        "source": "diagram_fallback",
                        "bbox": bbox,
                        "is_diagram": True,
                    },
                ),
                page_idx,
            )
        except ImportError as exc:
            warnings.append(
                {
                    "type": "pdf_diagram_render_missing_dependency",
                    "severity": "medium",
                    "page": page_idx + 1,
                    "message": str(exc),
                }
            )
        except Exception as exc:
            warnings.append(
                {
                    "type": "pdf_diagram_render_failed",
                    "severity": "medium",
                    "page": page_idx + 1,
                    "message": str(exc),
                }
            )

        if source_text:
            structured["confidence"] = "medium"
            if asset_id is not None:
                structured["asset_id"] = asset_id
            segments.append(
                _Segment(
                    top=top,
                    bottom=bbox[3],
                    kind="diagram",
                    payload=structured,
                    page=page_idx + 1,
                    metadata={"confidence": "medium"},
                )
            )
            warnings.append(
                {
                    "type": "pdf_diagram_inferred",
                    "severity": "low",
                    "page": page_idx + 1,
                    "message": (
                        "PDF vector diagram structure was inferred from shapes "
                        "and text bounding boxes."
                    ),
                }
            )
            continue

        if asset_id is None:
            warnings.append(
                {
                    "type": "pdf_diagram_structuring_failed",
                    "severity": "medium",
                    "page": page_idx + 1,
                    "message": (
                        "PDF diagram structure could not be inferred and no "
                        "fallback image was available."
                    ),
                }
            )
            continue

        fallback = _structured_diagram_content(
            nodes=[],
            extra={"asset_id": asset_id, "confidence": "low"},
        )
        segments.append(
            _Segment(
                top=top,
                bottom=bbox[3],
                kind="diagram",
                payload=fallback,
                page=page_idx + 1,
                metadata={"confidence": "low"},
            )
        )
        warnings.append(
            {
                "type": "pdf_diagram_structuring_failed",
                "severity": "medium",
                "page": page_idx + 1,
                "message": (
                    "PDF diagram was preserved as a fallback image because "
                    "vector structure could not be inferred."
                ),
            }
        )
    return segments


def _page_segments_ordered(
    page: object,
    page_number: int,
    tables: list[object],
    image_segments: list[_Segment],
    diagram_segments: list[_Segment],
    cell_image_children: dict[int, dict[tuple[int, int], list[dict[str, object]]]],
) -> list[_Segment]:
    segments: list[_Segment] = []
    nested = _resolve_nested_tables(tables)
    for table_idx, table in enumerate(tables):
        if table_idx in nested.suppressed:
            continue
        structured = _structured_table_from_pdf_table(
            page,
            tables,
            table_idx,
            nested,
            seen=set(),
            cell_image_children=cell_image_children,
        )
        if structured is None:
            continue
        if _is_empty_single_cell_table_artifact(structured):
            continue
        text_box = _title_table_text(structured)
        if text_box is None:
            text_box = _single_line_header_only_table_text(table, structured)
        if text_box is not None:
            segments.append(
                _Segment(
                    top=float(table.bbox[1]),
                    bottom=float(table.bbox[3]),
                    kind="text",
                    payload=text_box,
                    page=page_number,
                )
            )
            continue
        if not structured["columns"] and not structured["rows"]:
            continue
        segments.append(
            _Segment(
                top=float(table.bbox[1]),
                bottom=float(table.bbox[3]),
                kind="table",
                payload=structured,
                page=page_number,
            )
        )

    segments.extend(image_segments)
    segments.extend(diagram_segments)
    obstacle_bands = sorted(
        [(segment.top, segment.bottom) for segment in segments],
        key=lambda band: band[0],
    )
    prev_bottom = 0.0
    page_height = float(getattr(page, "height", 0.0))
    page_width = float(getattr(page, "width", 0.0))
    for top, bottom in obstacle_bands:
        if top > prev_bottom + 2:
            text = _crop_text(page, 0.0, prev_bottom, page_width, top)
            if text:
                segments.append(
                    _Segment(
                        top=prev_bottom,
                        bottom=top,
                        kind="text",
                        payload=text,
                        page=page_number,
                    )
                )
        prev_bottom = max(prev_bottom, bottom)
    if page_height and prev_bottom < page_height - 2:
        text = _crop_text(page, 0.0, prev_bottom, page_width, page_height)
        if text:
            segments.append(
                _Segment(
                    top=prev_bottom,
                    bottom=page_height,
                    kind="text",
                    payload=text,
                    page=page_number,
                )
            )

    return sorted(segments, key=lambda segment: segment.top)


def _segments_to_units(page_segments: list[list[_Segment]]) -> list[EvidenceUnit]:
    units: list[EvidenceUnit] = []
    block_index = 1
    table_index = 1

    for segments in page_segments:
        for segment in sorted(segments, key=lambda item: item.top):
            if segment.kind == "text":
                text = str(segment.payload).strip()
                if not text or _is_pdf_artifact_text(text):
                    continue
                pdf_metadata = {"page": segment.page}
                pdf_metadata.update(segment.metadata)
                units.append(
                    EvidenceUnit(
                        id=f"b{block_index}",
                        type="text",
                        format="plain",
                        source=SourceEvidence(kind="text", text=text),
                        content=text,
                        metadata={
                            "common": {
                                "chunk_kind": "text",
                                "section_path": [],
                                "display_format": "plain",
                            },
                            "pdf": pdf_metadata,
                        },
                    )
                )
                block_index += 1
                continue
            if segment.kind == "table":
                table = segment.payload
                table_id = f"t{table_index}"
                headers = [str(column["text"]) for column in table["columns"]]
                pdf_metadata = {
                    "page": segment.page,
                    "confidence": segment.metadata.get("confidence", "high"),
                }
                if segment.metadata.get("ocr"):
                    pdf_metadata["ocr"] = True
                units.append(
                    EvidenceUnit(
                        id=f"b{block_index}",
                        type="table",
                        format="structured_table",
                        source=SourceEvidence(
                            kind="table",
                            text=_table_source_text(table),
                        ),
                        content=table,
                        metadata={
                            "common": {
                                "chunk_kind": "table",
                                "section_path": [],
                                "display_format": "structured_table",
                            },
                            "table": {
                                "table_id": table_id,
                                "headers": headers,
                                "row_count": len(table["rows"]),
                            },
                            "pdf": pdf_metadata,
                        },
                    )
                )
                block_index += 1
                table_index += 1
                continue
            if segment.kind == "image":
                asset_id = str(segment.payload["asset_id"])
                units.append(
                    EvidenceUnit(
                        id=f"b{block_index}",
                        type="image",
                        format="asset_ref",
                        source=SourceEvidence(
                            kind="image",
                            text=f"image: {asset_id}",
                        ),
                        content=dict(segment.payload),
                        metadata={
                            "common": {
                                "chunk_kind": "image",
                                "section_path": [],
                                "display_format": "image",
                            },
                            "asset": {"asset_id": asset_id},
                            "pdf": {
                                "page": segment.page,
                                "confidence": segment.metadata.get(
                                    "confidence",
                                    "medium",
                                ),
                            },
                        },
                    )
                )
                block_index += 1
                continue
            if segment.kind == "diagram":
                diagram = segment.payload
                source_text = _diagram_source_text(diagram)
                if not source_text:
                    asset_id = str(diagram.get("asset_id", "")).strip()
                    source_text = f"diagram image: {asset_id}" if asset_id else ""
                if not source_text:
                    continue
                confidence = str(
                    segment.metadata.get(
                        "confidence",
                        diagram.get("confidence", "low"),
                    )
                )
                diagram_metadata: dict[str, object] = {
                    "node_count": len(diagram.get("nodes", [])),
                    "edge_count": len(diagram.get("edges", [])),
                    "confidence": confidence,
                }
                if diagram.get("asset_id"):
                    diagram_metadata["fallback_asset_id"] = diagram["asset_id"]
                units.append(
                    EvidenceUnit(
                        id=f"b{block_index}",
                        type="diagram",
                        format="structured_diagram",
                        source=SourceEvidence(kind="diagram", text=source_text),
                        content=diagram,
                        metadata={
                            "common": {
                                "chunk_kind": "diagram",
                                "section_path": [],
                                "display_format": "structured_diagram",
                            },
                            "diagram": diagram_metadata,
                            "pdf": {
                                "page": segment.page,
                                "confidence": confidence,
                            },
                        },
                    )
                )
                block_index += 1

    return units


def _merge_continuation_tables(page_segments: list[list[_Segment]]) -> list[list[_Segment]]:
    merged_pages: list[list[_Segment]] = [[] for _ in page_segments]
    previous_table: _Segment | None = None

    for page_idx, segments in enumerate(page_segments):
        page_has_content_before_table = False
        for segment in sorted(segments, key=lambda item: item.top):
            if segment.kind != "table":
                merged_pages[page_idx].append(segment)
                is_artifact_text = (
                    segment.kind == "text"
                    and _is_pdf_artifact_text(str(segment.payload).strip())
                )
                if segment.kind in {"text", "image", "diagram"} and not is_artifact_text:
                    page_has_content_before_table = True
                    previous_table = None
                continue

            if (
                previous_table is not None
                and not page_has_content_before_table
                and _is_table_continuation(previous_table.payload, segment.payload)
            ):
                _append_table_rows(previous_table.payload, segment.payload)
                continue

            merged_pages[page_idx].append(segment)
            previous_table = segment

    return merged_pages


def _expand_text_segments(
    page_segments: list[list[_Segment]],
) -> list[list[_Segment]]:
    expanded_pages: list[list[_Segment]] = []
    for segments in page_segments:
        expanded: list[_Segment] = []
        for segment in segments:
            if segment.kind != "text":
                expanded.append(segment)
                continue
            parts = _revision_history_parts(str(segment.payload))
            if parts is None:
                text_parts = _structured_text_parts(str(segment.payload))
                if text_parts is None:
                    expanded.append(segment)
                    continue
                parts = [("text", part) for part in text_parts]
            for part_index, (kind, payload) in enumerate(parts):
                expanded.append(
                    _Segment(
                        top=segment.top + (part_index * 0.001),
                        bottom=segment.bottom,
                        kind=kind,
                        payload=payload,
                        page=segment.page,
                    )
                )
                continue
        expanded_pages.append(
            _drop_duplicate_short_title_segments(
                sorted(expanded, key=lambda item: item.top)
            )
        )
    return expanded_pages


def _structured_text_parts(text: str) -> list[str] | None:
    return (
        _scanned_official_letter_text_parts(text)
        or _scanned_official_letter_continuation_parts(text)
        or _official_notice_text_parts(text)
        or _related_basis_text_parts(text)
        or _sectioned_text_parts(text)
        or _short_heading_text_parts(text)
    )


def _drop_duplicate_short_title_segments(segments: list[_Segment]) -> list[_Segment]:
    result: list[_Segment] = []
    index = 0
    while index < len(segments):
        segment = segments[index]
        next_segment = segments[index + 1] if index + 1 < len(segments) else None
        if _is_duplicate_short_title_segment(segment, next_segment):
            index += 1
            continue
        result.append(segment)
        index += 1
    return result


def _is_duplicate_short_title_segment(
    segment: _Segment,
    next_segment: _Segment | None,
) -> bool:
    if segment.kind != "text" or next_segment is None or next_segment.kind != "text":
        return False
    if segment.page != next_segment.page:
        return False
    title = str(segment.payload).strip()
    next_text = str(next_segment.payload).strip()
    if (
        0 < len(title) <= 30
        and "\n" not in title
        and title.endswith("대상")
        and title in next_text
        and bool(re.match(r"^[가-힣]\.", next_text))
    ):
        return True
    if not _is_short_pdf_text_box_segment(segment, title):
        return False
    normalized_title = _normalize_duplicate_text(title)
    normalized_next = _normalize_duplicate_text(next_text)
    return (
        len(normalized_title) >= 6
        and normalized_title in normalized_next
    )


def _is_short_pdf_text_box_segment(segment: _Segment, text: str) -> bool:
    return (
        0 < len(text) <= 160
        and "\n" not in text
        and segment.bottom - segment.top <= 24
    )


def _normalize_duplicate_text(text: str) -> str:
    return re.sub(r"\s+", "", text.strip())


def _official_notice_text_parts(text: str) -> list[str] | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or not lines[0].startswith("보건복지부 고시 "):
        return None

    parts = [lines[0]]
    paragraph: list[str] = []
    for line in lines[1:]:
        if _is_standalone_notice_line(line):
            if paragraph:
                parts.append(_join_pdf_text_lines(paragraph))
                paragraph = []
            parts.append(line)
            continue
        paragraph.append(line)
        if line.endswith("다."):
            parts.append(_join_pdf_text_lines(paragraph))
            paragraph = []
    if paragraph:
        parts.append(_join_pdf_text_lines(paragraph))
    return parts if len(parts) > 1 else None


def _scanned_official_letter_text_parts(text: str) -> list[str] | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not _looks_like_scanned_official_letter(lines):
        return None

    parts: list[str] = []
    paragraph: list[str] = []
    for line in lines:
        if _is_official_letter_boundary_line(line):
            if paragraph:
                parts.append(_join_pdf_text_lines(paragraph))
                paragraph = []
            if _is_numbered_paragraph_line(line):
                paragraph = [line]
            else:
                parts.append(line)
            continue
        if paragraph:
            paragraph.append(line)
            continue
        parts.append(line)

    if paragraph:
        parts.append(_join_pdf_text_lines(paragraph))
    return parts if len(parts) > 1 else None


def _looks_like_scanned_official_letter(lines: list[str]) -> bool:
    return (
        any(line.startswith("수신자") for line in lines)
        and any(line.startswith("제목") for line in lines)
        and any(_is_numbered_paragraph_line(line) for line in lines)
    )


def _is_official_letter_boundary_line(line: str) -> bool:
    return (
        _is_numbered_paragraph_line(line)
        or line.startswith("붙임")
        or line.startswith('"긴급지원')
        or line == "보건복지부"
        or line.startswith("수신자")
        or line == "(경유)"
        or line.startswith("제목")
    )


def _is_numbered_paragraph_line(line: str) -> bool:
    return bool(re.match(r"^\d+\.\s+", line))


def _scanned_official_letter_continuation_parts(text: str) -> list[str] | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1 and lines[0] == "보건복지부" and lines[1].startswith("수신자"):
        return _split_official_letter_continuation_lines(lines)
    if len(lines) != 1:
        return None
    line = lines[0]
    if not (line.startswith("보건복지부 수신자 ") and " 시행 " in line and " 전화 " in line):
        return None
    parts = _split_official_letter_continuation_line(line)
    return parts if len(parts) > 1 else None


def _split_official_letter_continuation_lines(lines: list[str]) -> list[str]:
    parts: list[str] = []
    for line in lines:
        split = _split_official_letter_continuation_line(line)
        parts.extend(split if len(split) > 1 else [line])
    return parts


def _split_official_letter_continuation_line(line: str) -> list[str]:
    markers = (
        " 수신자 ",
        " 주무관 ",
        " 주주관 ",
        " 주관 ",
        " 시행 ",
        " 접수 ",
        " 우 ",
        " 전화 ",
    )
    split_points = sorted(
        index
        for marker in markers
        for index in [line.find(marker)]
        if index > 0
    )
    parts: list[str] = []
    start = 0
    for index in split_points:
        part = line[start:index].strip()
        if part:
            parts.append(part)
        start = index + 1
    tail = line[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _related_basis_text_parts(text: str) -> list[str] | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 3 or lines[1] != "1. 관련 근거":
        return None
    if not all(line.startswith("○") for line in lines[2:]):
        return None
    return lines


def _sectioned_text_parts(text: str) -> list[str] | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2 or not any(_is_section_heading_line(line) for line in lines):
        return None

    parts: list[str] = []
    paragraph: list[str] = []
    for line in lines:
        if _is_section_heading_line(line):
            if paragraph:
                parts.append(_join_section_text_lines(paragraph))
                paragraph = []
            parts.append(line)
            continue
        paragraph.append(line)
    if paragraph:
        parts.append(_join_section_text_lines(paragraph))
    return parts if len(parts) > 1 else None


def _is_section_heading_line(line: str) -> bool:
    stripped = line.strip()
    return (
        stripped.startswith("□")
        or stripped == "일반사항"
        or stripped == "⋮"
        or stripped.startswith("개정 ")
        or (stripped.startswith("「") and stripped.endswith("Q&A"))
        or (stripped.endswith("Q&A") and len(stripped) <= 60)
        or (stripped.startswith("<") and stripped.endswith(">"))
    )


def _is_standalone_notice_line(line: str) -> bool:
    return (
        bool(re.match(r"^\d{4}년\s+\d{1,2}월\s+\d{1,2}일$", line))
        or line == "보건복지부 장관"
        or line.endswith("일부개정")
    )


def _short_heading_text_parts(text: str) -> list[str] | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2 or len(lines) > 3:
        return None
    if all(len(line) <= 40 and not line.endswith(("다.", ".")) for line in lines):
        return lines
    return None


def _join_pdf_text_lines(lines: list[str]) -> str:
    text = " ".join(line.strip() for line in lines if line.strip())
    text = text.replace("세부 사항", "세부사항")
    return text.strip()


def _join_section_text_lines(lines: list[str]) -> str:
    groups: list[list[str]] = []
    current: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if current and _is_section_text_boundary_line(line):
            groups.append(current)
            current = [line]
            continue
        current.append(line)
    if current:
        groups.append(current)
    return "\n".join(_join_pdf_text_lines(group) for group in groups).strip()


def _is_section_text_boundary_line(line: str) -> bool:
    stripped = line.strip()
    return bool(
        stripped.startswith(("■", "□", "○", "〇", "-", "ㆍ", "․", "*", "※"))
        or re.match(r"^\d+\.\s+", stripped)
        or re.match(r"^\(?\d+\)\s*", stripped)
        or re.match(r"^[가-힣]\.\s+", stripped)
        or re.match(r"^제\d+조(?:의\d+)?\(", stripped)
    )


def _revision_history_parts(text: str) -> list[tuple[str, object]] | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    try:
        marker_index = lines.index("관련 근거")
    except ValueError:
        return None

    rows: list[dict[str, object]] = []
    after_index = marker_index + 1
    for line in lines[marker_index + 1:]:
        row = _revision_history_row(line)
        if row is None:
            break
        row["index"] = len(rows) + 1
        rows.append(row)
        after_index += 1
    if len(rows) < 2:
        return None

    parts: list[tuple[str, object]] = [
        ("text", line)
        for line in lines[:marker_index]
        if line
    ]
    parts.append(
        (
            "table",
            {
                "caption": None,
                "columns": [
                    {"id": "c1", "text": "개정일"},
                    {"id": "c2", "text": "고시"},
                    {"id": "c3", "text": "시행일"},
                    {"id": "c4", "text": "관련 근거"},
                ],
                "header_rows": [
                    {
                        "index": 1,
                        "cells": [
                            _simple_cell("c1", "개정일"),
                            _simple_cell("c2", "고시"),
                            _simple_cell("c3", "시행일"),
                            _simple_cell("c4", "관련 근거"),
                        ],
                    }
                ],
                "rows": rows,
            },
        )
    )
    parts.extend(("text", part) for part in _trailing_revision_texts(lines[after_index:]))
    return parts


_REVISION_LINE_RE = re.compile(
    r"^(개정\s+[’']?\d{2,4}\.\d{1,2}\.\d{1,2}\.)"
    r"(?:\s+(고시\s+제\d{4}-\d+호))?"
    r"(?:\s+\(([^)]*시행)\))?"
    r"\s*(.*)$"
)


def _revision_history_row(line: str) -> dict[str, object] | None:
    match = _REVISION_LINE_RE.match(line)
    if match is None:
        return None
    values = [(part or "").strip() for part in match.groups()]
    return {
        "index": 0,
        "cells": [
            _simple_cell("c1", values[0]),
            _simple_cell("c2", values[1]),
            _simple_cell("c3", values[2]),
            _simple_cell("c4", values[3]),
        ],
    }


def _simple_cell(
    column_id: str,
    text: str,
    *,
    rowspan: int = 1,
    colspan: int = 1,
) -> dict[str, object]:
    return {
        "column_id": column_id,
        "text": text,
        "rowspan": rowspan,
        "colspan": colspan,
        "children": [],
    }


def _trailing_revision_texts(lines: list[str]) -> list[str]:
    result: list[str] = []
    paragraph: list[str] = []
    for line in lines:
        if line.startswith("*"):
            if paragraph:
                result.append(_join_lines("\n".join(paragraph)))
                paragraph = []
            result.append(line)
            continue
        if line.startswith("※") and paragraph:
            result.append(_join_lines("\n".join(paragraph)))
            paragraph = [line]
            continue
        paragraph.append(line)
    if paragraph:
        result.append(_join_lines("\n".join(paragraph)))
    return result


def _is_table_continuation(
    previous: dict[str, object],
    current: dict[str, object],
) -> bool:
    previous_signature = _table_column_signature(previous)
    return bool(previous_signature) and previous_signature == _table_column_signature(current)


def _table_column_signature(table: dict[str, object]) -> tuple[str, ...]:
    return tuple(
        _normalize_header_text(str(column.get("text", "")))
        for column in table.get("columns", [])
    )


def _append_table_rows(
    target: dict[str, object],
    continuation: dict[str, object],
) -> None:
    target_rows = target.get("rows")
    continuation_rows = continuation.get("rows")
    if not isinstance(target_rows, list) or not isinstance(continuation_rows, list):
        return
    for row in continuation_rows:
        if not isinstance(row, Mapping):
            continue
        copied = dict(row)
        copied["index"] = len(target_rows) + 1
        target_rows.append(copied)
    _merge_wrapped_table_rows(target)
    _promote_code_table_leaf_headers(target)
    _promote_ultrasound_code_matrix(target)
    _expand_parallel_code_action_rows(target)


def _structured_table_from_pdf_table(
    page: object,
    tables: list[object],
    table_idx: int,
    nested: _NestedResolution,
    seen: set[int],
    *,
    cell_image_children: dict[int, dict[tuple[int, int], list[dict[str, object]]]] | None = None,
) -> dict[str, object] | None:
    if table_idx in seen:
        return None
    table = tables[table_idx]
    raw_rows = _table_rows(table)
    if not raw_rows:
        return None
    raw_rows = _trim_empty_trailing_columns(raw_rows)
    column_count = max((len(row) for row in raw_rows), default=0)
    if column_count == 0:
        return _structured_table_content(columns=[], rows=[])
    normalized_rows = [_pad_row(row, column_count) for row in raw_rows]
    header_depth = _header_depth(normalized_rows)

    columns = _columns_from_header_rows(normalized_rows[:header_depth], column_count)
    child_map = nested.children.get(table_idx, {})
    image_child_map = (cell_image_children or {}).get(table_idx, {})
    cell_spans = _table_cell_spans(
        table,
        row_count=len(normalized_rows),
        column_count=column_count,
    )
    header_rows: list[dict[str, object]] = []
    for header_index, raw_row in enumerate(normalized_rows[:header_depth]):
        header_cells = _row_evidence_cells(
            page,
            tables,
            table_idx,
            row_index=header_index,
            raw_row=raw_row,
            columns=columns,
            child_map=child_map,
            cell_spans=cell_spans,
            nested=nested,
            seen=seen | {table_idx},
            cell_image_children=cell_image_children or {},
            image_child_map=image_child_map,
        )
        if header_cells:
            header_rows.append({"index": header_index + 1, "cells": header_cells})

    rows: list[dict[str, object]] = []
    for raw_index, raw_row in enumerate(normalized_rows[header_depth:], start=header_depth):
        cells = _row_evidence_cells(
            page,
            tables,
            table_idx,
            row_index=raw_index,
            raw_row=raw_row,
            columns=columns,
            child_map=child_map,
            cell_spans=cell_spans,
            nested=nested,
            seen=seen | {table_idx},
            cell_image_children=cell_image_children or {},
            image_child_map=image_child_map,
        )
        if not any(str(cell["text"]).strip() or cell["children"] for cell in cells):
            continue
        rows.append({"index": len(rows) + 1, "cells": cells})

    result = _structured_table_content(
        columns=columns,
        rows=rows,
        header_rows=header_rows if header_rows else None,
    )
    _repair_table_of_contents(result, page)
    _merge_blank_header_rowspans(result)
    _merge_wrapped_table_rows(result)
    _promote_code_table_leaf_headers(result)
    _promote_ultrasound_code_matrix(result)
    _expand_parallel_code_action_rows(result)
    return result


def _merge_blank_header_rowspans(table: dict[str, object]) -> None:
    header_rows = table.get("header_rows")
    if not isinstance(header_rows, list) or len(header_rows) < 2:
        return

    previous_cells_by_column: dict[str, dict[str, object]] = {}
    for header_row in header_rows:
        cells = header_row.get("cells") if isinstance(header_row, Mapping) else None
        if not isinstance(cells, list):
            previous_cells_by_column = {}
            continue

        next_cells: list[dict[str, object]] = []
        current_cells_by_column: dict[str, dict[str, object]] = {}
        for cell in cells:
            if not isinstance(cell, Mapping):
                continue
            column_id = str(cell.get("column_id", ""))
            previous = previous_cells_by_column.get(column_id)
            if _is_blank_header_slot(cell) and _can_absorb_blank_header_slot(previous):
                previous["rowspan"] = int(previous.get("rowspan", 1) or 1) + 1
                continue
            next_cells.append(cell)
            if column_id:
                current_cells_by_column[column_id] = cell

        header_row["cells"] = next_cells
        previous_cells_by_column = current_cells_by_column


def _is_blank_header_slot(cell: dict[str, object]) -> bool:
    return (
        not str(cell.get("text", "")).strip()
        and int(cell.get("rowspan", 1) or 1) == 1
        and int(cell.get("colspan", 1) or 1) == 1
    )


def _can_absorb_blank_header_slot(cell: dict[str, object] | None) -> bool:
    return (
        isinstance(cell, Mapping)
        and bool(str(cell.get("text", "")).strip())
        and int(cell.get("rowspan", 1) or 1) == 1
        and int(cell.get("colspan", 1) or 1) == 1
    )


def _merge_wrapped_table_rows(table: dict[str, object]) -> None:
    if _is_table_of_contents(table):
        return
    rows = table.get("rows")
    if not isinstance(rows, list) or len(rows) < 2:
        return

    merged: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if merged and _is_wrapped_table_row(row):
            _append_wrapped_row_cells(merged[-1], row)
            continue
        row["index"] = len(merged) + 1
        merged.append(row)

    table["rows"] = merged


def _is_table_of_contents(table: dict[str, object]) -> bool:
    columns = table.get("columns")
    if not isinstance(columns, list):
        return False
    labels = [
        _normalize_header_text(str(column.get("text", "")))
        for column in columns
        if isinstance(column, Mapping)
    ]
    return "제목" in labels and "페이지" in labels


def _repair_table_of_contents(table: dict[str, object], page: object) -> None:
    if not _is_table_of_contents(table):
        return
    rows = table.get("rows")
    if not isinstance(rows, list) or not rows:
        return

    entries = _table_of_contents_entries(page)
    if len(entries) < len(rows):
        return

    entry_index = 0
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        cells = row.get("cells")
        if not isinstance(cells, list) or len(cells) < 3:
            continue
        title = str(cells[1].get("text", "")) if isinstance(cells[1], Mapping) else ""
        entry = _matching_table_of_contents_entry(title, entries, entry_index)
        if entry is None:
            continue
        entry_index = entries.index(entry, entry_index) + 1
        number, _title, page_number = entry
        if isinstance(cells[0], Mapping) and not str(cells[0].get("text", "")).strip():
            cells[0]["text"] = number
        if isinstance(cells[-1], Mapping) and not str(cells[-1].get("text", "")).strip():
            cells[-1]["text"] = page_number


def _table_of_contents_entries(page: object) -> list[tuple[str, str, str]]:
    try:
        text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
    except TypeError:
        text = page.extract_text() or ""
    except Exception:
        return []

    entries: list[tuple[str, str, str]] = []
    for line in text.splitlines():
        match = re.match(r"^\s*(\d{1,3})\s+(.+?)\s+(\d{1,4})\s*$", line)
        if match is not None:
            entries.append(tuple(part.strip() for part in match.groups()))
    return entries


def _matching_table_of_contents_entry(
    title: str,
    entries: list[tuple[str, str, str]],
    start_index: int,
) -> tuple[str, str, str] | None:
    normalized_title = _compact_text(title)
    for entry in entries[start_index:]:
        if not normalized_title or normalized_title == _compact_text(entry[1]):
            return entry
    return None


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _promote_code_table_leaf_headers(table: dict[str, object]) -> None:
    columns = table.get("columns")
    header_rows = table.get("header_rows")
    rows = table.get("rows")
    if (
        not isinstance(columns, list)
        or len(columns) < 2
        or not isinstance(header_rows, list)
        or len(header_rows) != 1
        or not isinstance(rows, list)
        or not rows
    ):
        return

    labels = [
        _normalize_header_text(str(column.get("text", "")))
        for column in columns
        if isinstance(column, Mapping)
    ]
    if len(labels) != len(columns) or labels[0] != "질병코드" or any(labels[1:]):
        return

    first_row = rows[0]
    if not isinstance(first_row, Mapping):
        return
    first_cells = first_row.get("cells")
    if not isinstance(first_cells, list) or len(first_cells) != len(columns):
        return
    leaf_values = [str(cell.get("text", "")).strip() for cell in first_cells]
    if not leaf_values or not all(_looks_like_disease_code(value) for value in leaf_values):
        return

    for index, column in enumerate(columns):
        if isinstance(column, Mapping):
            column["text"] = f"질병코드 / {leaf_values[index]}"

    header_rows[0] = {
        "index": 1,
        "cells": [
            {
                "column_id": "c1",
                "text": "질병코드",
                "rowspan": 1,
                "colspan": len(columns),
                "children": [],
            }
        ],
    }
    header_rows.append(
        {
            "index": 2,
            "cells": [
                _simple_cell(str(column.get("id", f"c{index + 1}")), leaf_values[index])
                for index, column in enumerate(columns)
                if isinstance(column, Mapping)
            ],
        }
    )

    table["rows"] = [
        _reindexed_row(row, index + 1)
        for index, row in enumerate(rows[1:])
        if isinstance(row, Mapping)
    ]


def _looks_like_disease_code(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]\d{2}(?:\.\d+)?(?:~[A-Z]?\d{2}(?:\.\d+)?)?", value))


def _promote_ultrasound_code_matrix(table: dict[str, object]) -> None:
    columns = table.get("columns")
    rows = table.get("rows")
    if (
        not isinstance(columns, list)
        or [str(column.get("text", "")) for column in columns if isinstance(column, Mapping)]
        != ["구분", "EDI코드"]
        or not isinstance(rows, list)
        or len(rows) < 3
    ):
        return

    row_cells = [_row_cell_texts(row) for row in rows]
    if len(row_cells[0]) != 2 or set(_edi_codes(row_cells[0][1])) != {"EB401", "EB402"}:
        return

    promoted_rows: list[dict[str, object]] = []
    for cells in row_cells:
        if len(cells) != 2:
            return
        promoted = _ultrasound_code_rows(cells[0], cells[1])
        if promoted is None:
            return
        for group, label, code in promoted:
            promoted_rows.append(
                {
                    "index": len(promoted_rows) + 1,
                    "cells": [
                        _simple_cell("c1", group),
                        _simple_cell("c2", label),
                        _simple_cell("c3", code),
                    ],
                }
            )

    if not promoted_rows:
        return

    table["columns"] = [
        {"id": "c1", "text": "구분"},
        {"id": "c2", "text": "구분"},
        {"id": "c3", "text": "EDI코드"},
    ]
    table["header_rows"] = [
        {
            "index": 1,
            "cells": [
                {
                    "column_id": "c1",
                    "text": "구분",
                    "rowspan": 1,
                    "colspan": 2,
                    "children": [],
                },
                _simple_cell("c3", "EDI코드"),
            ],
        },
    ]
    table["rows"] = _collapse_leading_empty_rowspan_cells(promoted_rows, "c1")


def _collapse_leading_empty_rowspan_cells(
    rows: list[dict[str, object]],
    column_id: str,
) -> list[dict[str, object]]:
    active: dict[str, object] | None = None
    for row in rows:
        cells = row.get("cells")
        if not isinstance(cells, list):
            active = None
            continue
        target_index = next(
            (
                index
                for index, cell in enumerate(cells)
                if isinstance(cell, Mapping) and cell.get("column_id") == column_id
            ),
            None,
        )
        if target_index is None:
            active = None
            continue
        cell = cells[target_index]
        if not isinstance(cell, Mapping):
            active = None
            continue
        if str(cell.get("text", "")).strip():
            active = cell
            continue
        if active is None:
            continue
        active["rowspan"] = int(active.get("rowspan", 1)) + 1
        del cells[target_index]
    return rows


def _row_cell_texts(row: object) -> list[str]:
    if not isinstance(row, Mapping):
        return []
    cells = row.get("cells")
    if not isinstance(cells, list):
        return []
    return [
        str(cell.get("text", "")).strip()
        for cell in cells
        if isinstance(cell, Mapping)
    ]


def _ultrasound_code_rows(
    label_text: str,
    code_text: str,
) -> list[tuple[str, str, str]] | None:
    codes = _edi_codes(code_text)
    if len(codes) < 2:
        return None
    group_word = _ultrasound_group_word(label_text)
    if not group_word:
        return None

    labels = _known_ultrasound_labels(codes)
    if labels is None:
        labels = _ultrasound_labels_from_text(label_text, group_word, len(codes))
    if len(labels) != len(codes) or any(not label for label in labels):
        return None

    rows: list[tuple[str, str, str]] = []
    for index, (label, code) in enumerate(zip(labels, codes, strict=True)):
        group = f"{group_word} 초음파" if index == 0 else ""
        rows.append((group, label, code))
    return rows


def _ultrasound_group_word(text: str) -> str:
    padded = f" {text} "
    for word in ("기본", "진단", "제한적", "특수"):
        if f" {word} " in padded:
            return word
    return ""


def _known_ultrasound_labels(codes: list[str]) -> list[str] | None:
    labels: list[str] = []
    for code in codes:
        label = _ULTRASOUND_EDI_LABELS.get(_base_ultrasound_edi_code(code))
        if label is None:
            return None
        labels.append(label)
    return labels


def _base_ultrasound_edi_code(code: str) -> str:
    return code[:-3] if code.endswith("001") else code


def _ultrasound_labels_from_text(
    label_text: str,
    group_word: str,
    count: int,
) -> list[str]:
    without_group = re.sub(
        rf"(^|\s){re.escape(group_word)}(?=\s|$)",
        " ",
        label_text,
        count=1,
    ).strip()
    without_separator = re.sub(
        r"\s+초음파\s+",
        " ",
        without_group,
        count=1,
    ).strip()
    return _split_ultrasound_label_tail(without_separator, count)


def _split_ultrasound_label_tail(text: str, count: int) -> list[str]:
    if count <= 0:
        return []
    tokens = text.split()
    if count == 1:
        return [text.strip()]
    if len(tokens) <= count:
        return tokens
    return [*tokens[: count - 1], " ".join(tokens[count - 1:])]


_ULTRASOUND_EDI_LABELS = {
    "EB411": "안구",
    "EB412": "안와",
    "EB414": "갑상선·부갑상선",
    "EB415": "갑상선·부갑상선 제외한 경부",
    "EB421": "유방·액와부-일반",
    "EB422": "흉벽, 흉막, 늑골 등",
    "EB423": "유방·액와부-정밀",
    "EB424": "자동유방초음파",
    "EB430": "선천성 심질환 경흉부",
    "EB431": "경흉부-단순",
    "EB432": "경흉부-일반",
    "EB433": "경흉부-전문",
    "EB434": "부하-약물부하",
    "EB435": "부하-운동부하",
    "EB436": "태아정밀",
    "EB441": "간·담낭·담도·비장·췌장(일반)",
    "EB442": "간·담낭·담도·비장·췌장(정밀)",
    "EB443": "충수",
    "EB444": "소장·대장",
    "EB445": "서혜부",
    "EB446": "직장·항문",
    "EB447": "항문",
    "EB448": "신장·부신·방광",
    "EB449": "신장·부신",
    "EB450": "방광",
    "EB610": "선천성 심질환 경식도",
    "EB611": "경식도",
    "EB612": "심장내",
}


def _edi_codes(text: str) -> list[str]:
    return re.findall(r"\bEB\d{3}(?:001)?\b", text)


def _expand_parallel_code_action_rows(table: dict[str, object]) -> None:
    columns = table.get("columns")
    rows = table.get("rows")
    if (
        not isinstance(columns, list)
        or [str(column.get("text", "")) for column in columns if isinstance(column, Mapping)]
        != ["분류", "코드", "행위명"]
        or not isinstance(rows, list)
    ):
        return

    expanded_rows: list[dict[str, object]] = []
    changed = False
    for row in rows:
        cells = _row_cell_texts(row)
        if len(cells) != 3:
            continue
        codes = _parallel_cell_parts(cells[1])
        actions = _parallel_cell_parts(cells[2])
        if len(codes) == len(actions) and len(codes) > 1:
            changed = True
            for index, (code, action) in enumerate(zip(codes, actions, strict=True)):
                expanded_rows.append(
                    {
                        "index": len(expanded_rows) + 1,
                        "cells": [
                            _simple_cell("c1", cells[0] if index == 0 else ""),
                            _simple_cell("c2", code),
                            _simple_cell("c3", action),
                        ],
                    }
                )
            continue
        if isinstance(row, Mapping):
            copied = dict(row)
            copied["index"] = len(expanded_rows) + 1
            expanded_rows.append(copied)

    if changed:
        table["rows"] = expanded_rows


def _parallel_cell_parts(text: str) -> list[str]:
    line_parts = [part.strip() for part in text.splitlines() if part.strip()]
    if len(line_parts) > 1:
        return line_parts
    return _slash_parts(text)


def _slash_parts(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\s*/\s*", text) if part.strip()]


def _reindexed_row(row: dict[str, object], index: int) -> dict[str, object]:
    copied = dict(row)
    copied["index"] = index
    return copied


def _is_wrapped_table_row(row: dict[str, object]) -> bool:
    cells = row.get("cells")
    if not isinstance(cells, list) or not cells:
        return False

    first = cells[0]
    if _cell_has_content(first):
        return False

    content_cells = [
        cell
        for cell in cells
        if isinstance(cell, Mapping) and _cell_has_content(cell)
    ]
    if not content_cells:
        return False
    if len(content_cells) == 1:
        return True

    second = cells[1] if len(cells) > 1 else None
    if not _cell_has_text(second):
        return False
    return not _starts_new_table_subrow(str(second.get("text", "")))


def _append_wrapped_row_cells(
    target_row: dict[str, object],
    wrapped_row: dict[str, object],
) -> None:
    target_cells = target_row.get("cells")
    wrapped_cells = wrapped_row.get("cells")
    if not isinstance(target_cells, list) or not isinstance(wrapped_cells, list):
        return

    target_by_column = {
        str(cell.get("column_id")): cell
        for cell in target_cells
        if isinstance(cell, Mapping)
    }
    for wrapped_cell in wrapped_cells:
        if not isinstance(wrapped_cell, Mapping) or not _cell_has_content(wrapped_cell):
            continue
        target_cell = target_by_column.get(str(wrapped_cell.get("column_id")))
        if not isinstance(target_cell, Mapping):
            continue
        wrapped_text = str(wrapped_cell.get("text", "")).strip()
        if wrapped_text:
            target_text = str(target_cell.get("text", "")).strip()
            target_cell["text"] = (
                f"{target_text}\n{wrapped_text}"
                if target_text
                else wrapped_text
            )
        children = wrapped_cell.get("children")
        if isinstance(children, list) and children:
            target_children = target_cell.get("children")
            if isinstance(target_children, list):
                target_children.extend(children)
                target_cell["children"] = _merge_nested_table_children(target_children)
            else:
                target_cell["children"] = _merge_nested_table_children(list(children))


def _cell_has_content(cell: object) -> bool:
    if not isinstance(cell, Mapping):
        return False
    return _cell_has_text(cell) or bool(cell.get("children"))


def _cell_has_text(cell: object) -> bool:
    return isinstance(cell, Mapping) and bool(str(cell.get("text", "")).strip())


def _starts_new_table_subrow(text: str) -> bool:
    normalized = text.lstrip()
    return bool(re.match(r"(?:[-ㆍ•·]|[oO]\s|<)", normalized))


def _pad_row(row: list[str], column_count: int) -> list[str]:
    return [*row, *([""] * max(0, column_count - len(row)))]


def _header_depth(rows: list[list[str]]) -> int:
    if len(rows) < 2:
        return 1
    first = rows[0]
    second = rows[1]
    if not any(not cell.strip() for cell in first):
        return 1
    first_count = _non_empty_cell_count(first)
    second_count = _non_empty_cell_count(second)
    if second_count < 2:
        return 1
    if first_count < 2 and not _is_single_group_header_row(first):
        return 1
    if not _is_concise_header_row(second):
        return 1
    return 2


def _is_single_group_header_row(row: list[str]) -> bool:
    return (
        len(row) >= 3
        and _non_empty_cell_count(row) == 1
        and _is_concise_header_row(row)
    )


def _non_empty_cell_count(row: list[str]) -> int:
    return sum(1 for cell in row if cell.strip())


def _is_concise_header_row(row: list[str], max_cell_chars: int = 32) -> bool:
    values = [cell.strip() for cell in row if cell.strip()]
    return bool(values) and all(len(value) <= max_cell_chars for value in values)


def _columns_from_header_rows(
    header_rows: list[list[str]],
    column_count: int,
) -> list[dict[str, str]]:
    group_labels = (
        _forward_fill_header_row(header_rows[0], column_count)
        if len(header_rows) > 1
        else _pad_row(header_rows[0], column_count)
    )
    columns: list[dict[str, str]] = []
    for index in range(column_count):
        parts: list[str] = []
        for row_index, row in enumerate(header_rows):
            value = _normalize_header_text(
                group_labels[index] if row_index == 0 and len(header_rows) > 1 else row[index]
            )
            if value and value not in parts:
                parts.append(value)
        columns.append({"id": f"c{index + 1}", "text": " / ".join(parts)})
    return columns


def _forward_fill_header_row(row: list[str], column_count: int) -> list[str]:
    result: list[str] = []
    current = ""
    for value in _pad_row(row, column_count):
        normalized = _normalize_header_text(value)
        if normalized:
            current = normalized
        result.append(current)
    return result


def _normalize_header_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _table_cell_spans(
    table: object,
    *,
    row_count: int,
    column_count: int,
) -> _TableCellSpans:
    cell_rows = [
        _pad_row_cells(_row_cells(table, row_index), column_count)
        for row_index in range(row_count)
    ]
    column_boundaries = _table_column_boundaries(cell_rows, column_count)
    row_bottoms = [_row_bottom(cells) for cells in cell_rows]
    spans: dict[tuple[int, int], tuple[int, int]] = {}
    covered: set[tuple[int, int]] = set()
    for row_index, row in enumerate(cell_rows):
        for column_index, bbox in enumerate(row):
            if (row_index, column_index) in covered:
                continue
            if bbox is None:
                spans[(row_index, column_index)] = (1, 1)
                continue
            colspan = _pdf_cell_colspan(
                row,
                column_index,
                column_count,
                bbox,
                column_boundaries,
            )
            rowspan = _pdf_cell_rowspan(
                cell_rows,
                row_bottoms,
                row_index,
                column_index,
                colspan,
                bbox,
            )
            spans[(row_index, column_index)] = (rowspan, colspan)
            for covered_row in range(row_index, row_index + rowspan):
                for covered_column in range(column_index, column_index + colspan):
                    if (covered_row, covered_column) == (row_index, column_index):
                        continue
                    covered_bbox = cell_rows[covered_row][covered_column]
                    if _slot_is_covered_by_cell(covered_bbox, bbox):
                        covered.add((covered_row, covered_column))
    return _TableCellSpans(spans=spans, covered=covered)


def _pad_row_cells(
    row: list[tuple[float, float, float, float] | None],
    column_count: int,
) -> list[tuple[float, float, float, float] | None]:
    return [*row[:column_count], *([None] * max(0, column_count - len(row)))]


def _row_bottom(
    cells: list[tuple[float, float, float, float] | None],
) -> float | None:
    bottoms = [cell[3] for cell in cells if cell is not None]
    return max(bottoms) if bottoms else None


def _table_column_boundaries(
    cell_rows: list[list[tuple[float, float, float, float] | None]],
    column_count: int,
) -> list[float]:
    positions = [
        position
        for row in cell_rows
        for cell in row
        if cell is not None
        for position in (cell[0], cell[2])
    ]
    boundaries = _cluster_positions(positions)
    if len(boundaries) < column_count + 1:
        return []
    return boundaries


def _cluster_positions(
    positions: list[float],
    *,
    tolerance: float = 2.0,
) -> list[float]:
    clustered: list[float] = []
    for position in sorted(positions):
        if not clustered or abs(position - clustered[-1]) > tolerance:
            clustered.append(position)
            continue
        clustered[-1] = (clustered[-1] + position) / 2
    return clustered


def _pdf_cell_colspan(
    row: list[tuple[float, float, float, float] | None],
    column_index: int,
    column_count: int,
    bbox: tuple[float, float, float, float],
    column_boundaries: list[float],
) -> int:
    colspan = 1
    while column_index + colspan < column_count and _slot_is_covered_by_cell(
        row[column_index + colspan],
        bbox,
    ):
        if column_boundaries and not _cell_reaches_column_end(
            bbox,
            column_boundaries,
            column_index + colspan,
        ):
            break
        colspan += 1
    return colspan


def _cell_reaches_column_end(
    bbox: tuple[float, float, float, float],
    column_boundaries: list[float],
    column_index: int,
    *,
    tolerance: float = 2.0,
) -> bool:
    boundary_index = column_index + 1
    return (
        boundary_index >= len(column_boundaries)
        or bbox[2] >= column_boundaries[boundary_index] - tolerance
    )


def _pdf_cell_rowspan(
    cell_rows: list[list[tuple[float, float, float, float] | None]],
    row_bottoms: list[float | None],
    row_index: int,
    column_index: int,
    colspan: int,
    bbox: tuple[float, float, float, float],
    *,
    tolerance: float = 2.0,
) -> int:
    rowspan = 1
    for next_row in range(row_index + 1, len(cell_rows)):
        row_bottom = row_bottoms[next_row]
        if row_bottom is None or bbox[3] < row_bottom - tolerance:
            break
        if not all(
            _slot_is_covered_by_cell(cell_rows[next_row][covered_column], bbox)
            for covered_column in range(column_index, column_index + colspan)
        ):
            break
        rowspan += 1
    return rowspan


def _slot_is_covered_by_cell(
    slot: tuple[float, float, float, float] | None,
    bbox: tuple[float, float, float, float],
) -> bool:
    return slot is None or _bbox_near_equal(slot, bbox)


def _row_evidence_cells(
    page: object,
    tables: list[object],
    table_idx: int,
    *,
    row_index: int,
    raw_row: list[str],
    columns: list[dict[str, str]],
    child_map: dict[tuple[int, int], list[int]],
    cell_spans: _TableCellSpans,
    nested: _NestedResolution,
    seen: set[int],
    cell_image_children: dict[int, dict[tuple[int, int], list[dict[str, object]]]],
    image_child_map: dict[tuple[int, int], list[dict[str, object]]],
) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    row_cells = _row_cells(tables[table_idx], row_index)
    for column_index, column in enumerate(columns):
        if (row_index, column_index) in cell_spans.covered:
            continue
        rowspan, colspan = cell_spans.spans.get((row_index, column_index), (1, 1))
        text = raw_row[column_index] if column_index < len(raw_row) else ""
        child_indices = child_map.get((row_index, column_index), [])
        table_children = [
            child
            for child in (
                _nested_table_child(
                    page,
                    tables,
                    child_idx,
                    nested,
                    seen,
                    cell_image_children=cell_image_children,
                )
                for child_idx in child_indices
            )
            if child is not None
        ]
        children = [
            *table_children,
            *image_child_map.get((row_index, column_index), []),
        ]
        children = _merge_nested_table_children(children)
        if table_children:
            cell_bbox = row_cells[column_index] if column_index < len(row_cells) else None
            if cell_bbox is not None:
                text = _rebuild_cell_text(
                    page,
                    cell_bbox,
                    [tables[child_idx] for child_idx in child_indices],
                )
            else:
                text = ""
        cells.append(
            {
                "column_id": column["id"],
                "text": text,
                "rowspan": rowspan,
                "colspan": colspan,
                "children": children,
            }
        )
    return cells


def _merge_nested_table_children(
    children: list[dict[str, object]],
) -> list[dict[str, object]]:
    if len(children) < 2:
        return children

    merged: list[dict[str, object]] = []
    for child in children:
        if merged and _can_merge_nested_table_child(merged[-1], child):
            _append_nested_child_rows(merged[-1]["content"], child["content"])
            continue
        merged.append(child)
    return merged


def _can_merge_nested_table_child(
    previous: dict[str, object],
    current: dict[str, object],
) -> bool:
    if (
        previous.get("type", previous.get("kind")) != "table"
        or current.get("type", current.get("kind")) != "table"
    ):
        return False
    previous_content = previous.get("content")
    current_content = current.get("content")
    if not isinstance(previous_content, Mapping) or not isinstance(current_content, Mapping):
        return False
    previous_columns = previous_content.get("columns")
    current_columns = current_content.get("columns")
    if (
        _table_column_signature(previous_content)
        != _table_column_signature(current_content)
        and not _looks_like_nested_table_data_continuation(
            previous_content,
            current_content,
        )
    ):
        return False
    return (
        isinstance(previous_columns, list)
        and isinstance(current_columns, list)
        and len(previous_columns) == len(current_columns)
        and len(previous_columns) > 1
    )


def _looks_like_nested_table_data_continuation(
    previous: dict[str, object],
    current: dict[str, object],
) -> bool:
    previous_columns = previous.get("columns")
    current_columns = current.get("columns")
    if not isinstance(previous_columns, list) or not isinstance(current_columns, list):
        return False
    if len(previous_columns) != len(current_columns) or len(previous_columns) <= 1:
        return False
    labels = [
        str(column.get("text", "")).strip()
        for column in current_columns
        if isinstance(column, Mapping)
    ]
    if len(labels) != len(current_columns):
        return False
    data_like = sum(_looks_like_pdf_table_data_value(label) for label in labels)
    return data_like >= max(2, len(labels) // 2)


def _looks_like_pdf_table_data_value(text: str) -> bool:
    stripped = text.strip()
    if not stripped or _CJK.search(stripped):
        return False
    if len(stripped) > 40:
        return False
    return bool(re.search(r"\d", stripped))


def _append_nested_child_rows(
    target: dict[str, object],
    continuation: dict[str, object],
) -> None:
    target_columns = target.get("columns")
    target_rows = target.get("rows")
    continuation_columns = continuation.get("columns")
    continuation_rows = continuation.get("rows")
    if (
        not isinstance(target_columns, list)
        or not isinstance(target_rows, list)
        or not isinstance(continuation_columns, list)
        or not isinstance(continuation_rows, list)
    ):
        return

    if _table_column_signature(target) != _table_column_signature(continuation):
        target_rows.append(
            {
                "index": len(target_rows) + 1,
                "cells": [
                    _simple_cell(
                        str(target_columns[index].get("id", f"c{index + 1}")),
                        str(column.get("text", "")),
                    )
                    for index, column in enumerate(continuation_columns)
                    if isinstance(column, Mapping)
                ],
            }
        )

    for row in continuation_rows:
        if not isinstance(row, Mapping):
            continue
        copied = dict(row)
        copied["index"] = len(target_rows) + 1
        target_rows.append(copied)
    _promote_code_table_leaf_headers(target)
    _promote_ultrasound_code_matrix(target)
    _expand_parallel_code_action_rows(target)


def _nested_table_child(
    page: object,
    tables: list[object],
    table_idx: int,
    nested: _NestedResolution,
    seen: set[int],
    *,
    cell_image_children: dict[int, dict[tuple[int, int], list[dict[str, object]]]] | None = None,
) -> dict[str, object] | None:
    structured = _structured_table_from_pdf_table(
        page,
        tables,
        table_idx,
        nested,
        seen,
        cell_image_children=cell_image_children,
    )
    if structured is None:
        return None
    return {
        "type": "table",
        "format": "structured_table",
        "content": structured,
    }


def _table_rows(table: object) -> list[list[str]]:
    try:
        rows = table.extract() or []
    except Exception:
        return []
    result: list[list[str]] = []
    for row in rows:
        if row is None:
            continue
        result.append([_clean_cell(cell) for cell in row])
    return result


def _trim_empty_trailing_columns(rows: list[list[str]]) -> list[list[str]]:
    trimmed = [list(row) for row in rows]
    while True:
        column_count = max((len(row) for row in trimmed), default=0)
        if column_count <= 1:
            return trimmed
        last_index = column_count - 1
        if any(last_index < len(row) and str(row[last_index]).strip() for row in trimmed):
            return trimmed
        trimmed = [
            row[:last_index] if last_index < len(row) else list(row)
            for row in trimmed
        ]


def _row_cells(table: object, row_index: int) -> list[tuple[float, float, float, float] | None]:
    rows = getattr(table, "rows", [])
    if row_index >= len(rows):
        return []
    return list(getattr(rows[row_index], "cells", []) or [])


def _title_table_text(table: dict[str, object]) -> str | None:
    if table["rows"]:
        return None
    header_rows = table.get("header_rows")
    if not isinstance(header_rows, list) or len(header_rows) != 1:
        return None
    cells = header_rows[0].get("cells")
    if not isinstance(cells, list) or not cells:
        return None
    if len(cells) > 3:
        return None
    parts: list[str] = []
    for cell in cells:
        if not isinstance(cell, Mapping) or cell.get("children"):
            return None
        text = str(cell.get("text", "")).strip()
        if text:
            parts.append(text)
    if not parts:
        return None
    text = _join_pdf_text_fragments(parts).strip()
    if len(cells) > 1 and len(text) > 40:
        return None
    return text or None


def _single_line_header_only_table_text(
    pdf_table: object,
    table: dict[str, object],
) -> str | None:
    if table["rows"]:
        return None
    bbox = getattr(pdf_table, "bbox", None)
    if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
        return None
    if float(bbox[3]) - float(bbox[1]) > 24:
        return None
    header_rows = table.get("header_rows")
    if not isinstance(header_rows, list) or len(header_rows) != 1:
        return None
    cells = header_rows[0].get("cells")
    if not isinstance(cells, list) or not cells or len(cells) > 3:
        return None
    parts: list[str] = []
    for cell in cells:
        if not isinstance(cell, Mapping) or cell.get("children"):
            return None
        text = str(cell.get("text", "")).strip()
        if text:
            parts.append(text)
    if not parts:
        return None
    return _join_pdf_text_fragments(parts).strip() or None


def _is_empty_single_cell_table_artifact(table: dict[str, object]) -> bool:
    columns = table.get("columns")
    if not isinstance(columns, list) or len(columns) != 1:
        return False
    all_rows: list[object] = []
    header_rows = table.get("header_rows")
    rows = table.get("rows")
    if isinstance(header_rows, list):
        all_rows.extend(header_rows)
    if isinstance(rows, list):
        all_rows.extend(rows)
    if len(all_rows) != 1:
        return False
    row = all_rows[0]
    if not isinstance(row, Mapping):
        return False
    cells = row.get("cells")
    if not isinstance(cells, list) or len(cells) != 1:
        return False
    cell = cells[0]
    if not isinstance(cell, Mapping):
        return False
    return not str(cell.get("text", "")).strip() and not cell.get("children")


def _join_pdf_text_fragments(parts: list[str]) -> str:
    if not parts:
        return ""
    out = parts[0].strip()
    for part in parts[1:]:
        current = part.strip()
        if not current:
            continue
        if _should_join_pdf_text_fragment(out, current):
            out = out.rstrip() + current.lstrip()
        else:
            out = out.rstrip() + " " + current.lstrip()
    return out


def _should_join_pdf_text_fragment(previous: str, current: str) -> bool:
    previous_token = previous.rstrip().rsplit(" ", 1)[-1]
    current_token = current.lstrip().split(" ", 1)[0]
    return (
        bool(previous_token)
        and bool(current_token)
        and len(current_token) <= 1
        and _CJK.search(current_token[0]) is not None
        and re.search(r"[0-9가-힣]$", previous_token) is not None
    )


def _resolve_nested_tables(tables: list[object]) -> _NestedResolution:
    contained: dict[int, tuple[int, int, int, float]] = {}
    for sub_idx, sub_table in enumerate(tables):
        sub_bbox = sub_table.bbox
        best: tuple[int, int, int, float] | None = None
        for parent_idx, parent_table in enumerate(tables):
            if parent_idx == sub_idx:
                continue
            parent_bbox = parent_table.bbox
            if _bbox_near_equal(parent_bbox, sub_bbox):
                continue
            if _bbox_area(parent_bbox) <= _bbox_area(sub_bbox):
                continue
            for row_idx, row in enumerate(getattr(parent_table, "rows", [])):
                for col_idx, cell in enumerate(getattr(row, "cells", []) or []):
                    if cell is None:
                        continue
                    if _bbox_in_cell(sub_bbox, cell):
                        area = _bbox_area(cell)
                        if best is None or area < best[3]:
                            best = (parent_idx, row_idx, col_idx, area)
        if best is not None:
            contained[sub_idx] = best

    children: dict[int, dict[tuple[int, int], list[int]]] = {}
    for sub_idx, (parent_idx, row_idx, col_idx, _area) in contained.items():
        children.setdefault(parent_idx, {}).setdefault((row_idx, col_idx), []).append(
            sub_idx
        )
    for cell_map in children.values():
        for child_indices in cell_map.values():
            child_indices.sort(key=lambda idx: float(tables[idx].bbox[1]))

    return _NestedResolution(suppressed=set(contained), children=children)


def _rebuild_cell_text(
    page: object,
    cell_bbox: tuple[float, float, float, float],
    child_tables: list[object],
) -> str:
    x0, top, x1, bottom = cell_bbox
    parts: list[str] = []
    y_cursor = top
    for child in sorted(child_tables, key=lambda table: float(table.bbox[1])):
        child_top = float(child.bbox[1])
        child_bottom = float(child.bbox[3])
        band = _crop_text(page, x0, y_cursor, x1, child_top)
        if band:
            parts.append(band)
        y_cursor = max(y_cursor, child_bottom)
    tail = _crop_text(page, x0, y_cursor, x1, bottom)
    if tail:
        parts.append(tail)
    return " ".join(parts).strip()


def _crop_text(page: object, x0: float, top: float, x1: float, bottom: float) -> str:
    if bottom - top <= 2 or x1 - x0 <= 2:
        return ""
    try:
        crop = page.crop((x0, top, x1, bottom))
        try:
            text = crop.extract_text(x_tolerance=3, y_tolerance=3) or ""
        except TypeError:
            text = crop.extract_text() or ""
    except Exception:
        return ""
    return _clean_text(text)


def _clean_text(text: str | None) -> str:
    stripped = _strip_page_numbers(text or "")
    return "\n".join(line.strip() for line in stripped.splitlines() if line.strip()).strip()


def _is_pdf_artifact_text(text: str) -> bool:
    return bool(re.fullmatch(r"INSID[A-Za-z0-9_:.-]+", text.strip()))


def _strip_page_numbers(text: str) -> str:
    return _PAGE_NUM_RE.sub("", text)


def _join_lines(text: str) -> str:
    parts = text.split("\n")
    if len(parts) == 1:
        return text
    out = parts[0]
    for part in parts[1:]:
        if out and _CJK.search(out[-1]) and part and _CJK.search(part[0]):
            out = out + part
        else:
            out = out.rstrip() + " " + part.lstrip()
    return out


def _join_cell_lines(text: str) -> str:
    parts = text.split("\n")
    if len(parts) == 1:
        return text
    out = parts[0]
    for part in parts[1:]:
        if _is_short_line_continuation(out, part):
            out = out.rstrip() + part.lstrip()
        else:
            out = out.rstrip() + " " + part.lstrip()
    return out


def _is_short_line_continuation(previous: str, current: str) -> bool:
    previous_token = previous.rstrip().rsplit(" ", 1)[-1]
    current_token = current.lstrip().split(" ", 1)[0]
    return (
        bool(previous_token)
        and bool(current_token)
        and _CJK.search(previous_token[-1]) is not None
        and _CJK.search(current_token[0]) is not None
        and (len(previous_token) <= 1 or len(current_token) <= 1)
    )


def _clean_cell(cell: object) -> str:
    if cell is None:
        return ""
    text = _join_cell_lines(str(cell))
    text = re.sub(
        r"(?<![가-힣])([가-힣])( [가-힣])+(?![가-힣])",
        lambda match: match.group(0).replace(" ", ""),
        text,
    )
    return text.strip()


def _is_scanned_page(page: object, min_text_chars: int = 30) -> bool:
    text_chars = 0
    for char in getattr(page, "chars", []):
        if str(char.get("text", "")).strip():
            text_chars += 1
            if text_chars >= min_text_chars:
                return False

    page_width = float(getattr(page, "width", 0.0) or 0.0)
    page_height = float(getattr(page, "height", 0.0) or 0.0)
    for img in getattr(page, "images", []):
        img_width = float(img.get("x1", 0.0) - img.get("x0", 0.0))
        img_height = float(img.get("y1", 0.0) - img.get("y0", 0.0))
        if (
            page_width
            and page_height
            and img_width / page_width > 0.7
            and img_height / page_height > 0.7
        ):
            return True
    return False


def _ocr_fallback_reason(
    page: object,
    *,
    allow_degraded_native: bool,
) -> str | None:
    if _is_scanned_page(page):
        return "scanned"
    if allow_degraded_native and _has_degraded_native_text(page):
        return "degraded_native_text"
    return None


def _has_degraded_native_text(page: object) -> bool:
    if not getattr(page, "images", None):
        return False
    text = "".join(
        str(char.get("text", ""))
        for char in getattr(page, "chars", [])
        if str(char.get("text", "")).strip()
    )
    total = len(text)
    if total < 15 or total > 250:
        return False
    cjk_count = sum(1 for char in text if _CJK.search(char))
    latin_count = sum(1 for char in text if char.isascii() and char.isalpha())
    punctuation_count = sum(1 for char in text if not char.isalnum())
    return (
        cjk_count / total < 0.05
        and latin_count / total < 0.10
        and punctuation_count / total >= 0.45
    )


def _ocr_pages(
    scanned: list[tuple[int, bytes]],
    data: bytes,
    max_workers: int | None,
    ocr_fn: Callable[[bytes, int], str] | None,
    ocr_llm: PdfOcrConfig | None,
) -> _OcrResults:
    if not scanned:
        return _OcrResults()

    def run_ocr(png: bytes, page_idx: int) -> str:
        if ocr_fn is not None:
            return ocr_fn(png, page_idx)
        if ocr_llm is not None:
            return _ocr_page_with_vision(data, png, page_idx, ocr_llm)
        return _ocr_page(data, png, page_idx)

    if max_workers is None or max_workers <= 1 or len(scanned) == 1:
        results = _OcrResults()
        for page_idx, png in scanned:
            _record_ocr_result(results, page_idx, lambda: run_ocr(png, page_idx))
        return results

    results = _OcrResults()
    workers = min(max_workers, len(scanned))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {
            executor.submit(run_ocr, png, page_idx): page_idx
            for page_idx, png in scanned
        }
        for future in as_completed(future_to_idx):
            page_idx = future_to_idx[future]
            _record_ocr_result(results, page_idx, future.result)
    results.failed_pages.sort(key=lambda failure: int(failure["page"]))
    return results


def _record_ocr_result(
    results: _OcrResults,
    page_idx: int,
    get_text: Callable[[], str],
) -> None:
    try:
        results[page_idx] = get_text() or ""
        if not results[page_idx].strip():
            results.failed_pages.append(
                {"page": page_idx, "stage": "ocr", "message": "empty OCR result"}
            )
    except Exception as exc:
        results[page_idx] = ""
        results.failed_pages.append(
            {"page": page_idx, "stage": "ocr", "message": str(exc)}
        )


def _ocr_warnings(failed_pages: list[dict[str, object]]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    for failure in failed_pages:
        page = int(failure.get("page", 0))
        warnings.append(
            {
                "type": "pdf_ocr_failed",
                "severity": "medium",
                "page": page + 1,
                "stage": failure.get("stage", "ocr"),
                "message": str(failure.get("message", "")),
            }
        )
    return warnings


def _ocr_text_segments(
    text: str,
    page_number: int,
) -> tuple[list[_Segment], list[dict[str, Any]]]:
    segments: list[_Segment] = []
    warnings: list[dict[str, Any]] = []
    part_index = 0
    for kind, payload in _ocr_text_parts(text):
        if kind in {"table", "table_pipe", "table_text"}:
            table = payload
            segments.append(
                _Segment(
                    top=part_index * 0.001,
                    bottom=part_index * 0.001,
                    kind="table",
                    payload=table,
                    page=page_number,
                    metadata={"ocr": True, "confidence": "medium"},
                )
            )
            table_type = "text table" if kind == "table_text" else "pipe table"
            warnings.append(
                {
                    "type": "pdf_ocr_table_inferred",
                    "severity": "low",
                    "page": page_number,
                    "message": (
                        "OCR text was converted to structured_table from a "
                        f"detected {table_type}."
                    ),
                }
            )
        else:
            segments.append(
                _Segment(
                    top=part_index * 0.001,
                    bottom=part_index * 0.001,
                    kind="text",
                    payload=str(payload),
                    page=page_number,
                    metadata={"ocr": True},
                )
            )
        part_index += 1
    return segments, warnings


def _ocr_text_parts(text: str) -> list[tuple[str, object]]:
    lines = text.splitlines()
    parts: list[tuple[str, object]] = []
    paragraph: list[str] = []
    index = 0

    def flush_paragraph() -> None:
        if not paragraph:
            return
        body = _clean_text("\n".join(paragraph))
        paragraph.clear()
        if body:
            parts.append(("text", body))

    while index < len(lines):
        if _looks_like_pipe_table_start(lines, index):
            flush_paragraph()
            table_lines = [lines[index], lines[index + 1]]
            index += 2
            while index < len(lines) and _is_pipe_table_row(lines[index]):
                table_lines.append(lines[index])
                index += 1
            table = _structured_table_from_pipe_lines(table_lines)
            if table is not None:
                parts.append(("table_pipe", table))
                continue
            paragraph.extend(table_lines)
            continue
        if _looks_like_aligned_table_start(lines, index):
            flush_paragraph()
            table_lines = [lines[index]]
            column_count = len(_split_aligned_table_row(lines[index]))
            index += 1
            while index < len(lines):
                row = _split_aligned_table_row(lines[index])
                if len(row) != column_count:
                    break
                table_lines.append(lines[index])
                index += 1
            table = _structured_table_from_aligned_lines(table_lines)
            if table is not None:
                parts.append(("table_text", table))
                continue
            paragraph.extend(table_lines)
            continue
        paragraph.append(lines[index])
        index += 1

    flush_paragraph()
    return parts or [("text", text)]


def _looks_like_pipe_table_start(lines: list[str], index: int) -> bool:
    return (
        index + 2 < len(lines)
        and _is_pipe_table_row(lines[index])
        and _is_pipe_separator_line(lines[index + 1])
        and _is_pipe_table_row(lines[index + 2])
    )


def _is_pipe_table_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 3


def _is_pipe_separator_line(line: str) -> bool:
    if not _is_pipe_table_row(line):
        return False
    cells = _split_pipe_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)


def _structured_table_from_pipe_lines(lines: list[str]) -> dict[str, object] | None:
    if len(lines) < 3:
        return None
    headers = _split_pipe_row(lines[0])
    rows = [_split_pipe_row(line) for line in lines[2:] if _is_pipe_table_row(line)]
    if not headers or not rows:
        return None
    columns = [
        table_column(f"c{index}", header)
        for index, header in enumerate(headers, start=1)
    ]
    structured_rows: list[dict[str, object]] = []
    for row in rows:
        cells = []
        for index, column in enumerate(columns):
            value = row[index] if index < len(row) else ""
            cells.append(_simple_cell(str(column["id"]), value))
        structured_rows.append(table_row(len(structured_rows) + 1, cells))
    return _structured_table_content(columns=columns, rows=structured_rows)


def _looks_like_aligned_table_start(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    header = _split_aligned_table_row(lines[index])
    first_row = _split_aligned_table_row(lines[index + 1])
    return (
        len(header) >= 2
        and len(header) == len(first_row)
        and all(cell.strip() for cell in header)
        and all(cell.strip() for cell in first_row)
    )


def _structured_table_from_aligned_lines(lines: list[str]) -> dict[str, object] | None:
    if len(lines) < 2:
        return None
    headers = _split_aligned_table_row(lines[0])
    rows = [_split_aligned_table_row(line) for line in lines[1:]]
    if not headers or not rows or any(len(row) != len(headers) for row in rows):
        return None
    columns = [
        table_column(f"c{index}", header)
        for index, header in enumerate(headers, start=1)
    ]
    structured_rows: list[dict[str, object]] = []
    for row in rows:
        cells = [
            _simple_cell(str(column["id"]), row[index])
            for index, column in enumerate(columns)
        ]
        structured_rows.append(table_row(len(structured_rows) + 1, cells))
    return _structured_table_content(columns=columns, rows=structured_rows)


def _split_aligned_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if not re.search(r"\t| {2,}", stripped):
        return []
    return [
        cell.strip()
        for cell in re.split(r"\t+| {2,}", stripped)
        if cell.strip()
    ]


def _split_pipe_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _ocr_page(data: bytes, png: bytes, page_idx: int, lang: str = "kor+eng") -> str:
    if png:
        text = _ocr_png(png, lang)
        if text:
            return text
    text = _ocr_via_pypdf(data, page_idx, lang)
    if text:
        return text
    return _ocr_via_pdf2image(data, page_idx, lang)


def _ocr_page_with_vision(
    data: bytes,
    png: bytes,
    page_idx: int,
    cfg: PdfOcrConfig,
) -> str:
    if png:
        text = _vision_ocr_png(png, cfg)
        if text:
            return text
    return _ocr_page(data, png, page_idx)


_VISION_OCR_PROMPT = """\
이 이미지는 스캔된 문서 페이지입니다.
이미지에서 텍스트와 표를 읽어 문서 구조를 보존해 추출해 주세요.

지침:
- 원문의 줄바꿈과 문단 구조를 최대한 유지
- 표는 가능하면 Markdown pipe table 형식으로 출력
- 표가 아닌 본문은 일반 텍스트로 출력
- 텍스트 내용만 출력, 설명 없이"""


def _vision_ocr_png(png: bytes, cfg: PdfOcrConfig) -> str:
    b64 = base64.b64encode(png).decode("ascii")
    body = {
        "model": cfg.model,
        "temperature": cfg.temperature,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {"type": "text", "text": _VISION_OCR_PROMPT},
                ],
            }
        ],
    }
    req = request.Request(
        chat_completions_url(cfg.url),
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=cfg.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return _clean_vision_ocr_text(
            str(payload["choices"][0]["message"]["content"])
        )
    except Exception:
        return ""


def _clean_vision_ocr_text(text: str) -> str:
    cleaned = text.strip()
    lines = cleaned.splitlines()
    if (
        len(lines) >= 2
        and re.fullmatch(r"```[ \t]*[A-Za-z0-9_+.-]*[ \t]*", lines[0].strip())
        and lines[-1].strip() == "```"
    ):
        return "\n".join(lines[1:-1]).strip()
    return cleaned


def _ocr_png(png: bytes, lang: str) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return ""
    try:
        return pytesseract.image_to_string(Image.open(io.BytesIO(png)), lang=lang)
    except Exception:
        return ""


def _ocr_via_pypdf(data: bytes, page_idx: int, lang: str) -> str:
    try:
        import pytesseract
        from pypdf import PdfReader
    except ImportError:
        return ""
    try:
        reader = PdfReader(io.BytesIO(data))
        if page_idx >= len(reader.pages):
            return ""
        page_images = [file for file in reader.pages[page_idx].images if file.image is not None]
        if not page_images:
            return ""
        largest = max(page_images, key=lambda file: file.image.width * file.image.height)
        return pytesseract.image_to_string(largest.image, lang=lang)
    except Exception:
        return ""


def _ocr_via_pdf2image(data: bytes, page_idx: int, lang: str) -> str:
    try:
        import pytesseract
        from pdf2image import convert_from_bytes
    except ImportError:
        return ""
    try:
        pages = convert_from_bytes(
            data,
            first_page=page_idx + 1,
            last_page=page_idx + 1,
            dpi=200,
        )
        if pages:
            return pytesseract.image_to_string(pages[0], lang=lang)
    except Exception:
        return ""
    return ""


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


def _structured_diagram_from_pdf(
    page: object,
    bbox: tuple[float, float, float, float],
) -> dict[str, object]:
    node_rects = [
        rect_bbox
        for rect in getattr(page, "rects", [])
        if (rect_bbox := _pdf_shape_bbox(rect)) is not None
        and _bbox_in_cell(rect_bbox, bbox, tol=2.0)
        and rect_bbox[2] - rect_bbox[0] >= 10
        and rect_bbox[3] - rect_bbox[1] >= 10
    ]
    nodes: list[dict[str, object]] = []
    for index, rect_bbox in enumerate(sorted(node_rects, key=lambda item: (item[1], item[0])), start=1):
        text = _crop_text(page, *rect_bbox)
        if not text:
            continue
        nodes.append(
            {
                "id": f"n{len(nodes) + 1}",
                "shape_type": "rect",
                "text": text,
                "bbox": _pdf_bbox_payload(rect_bbox),
                "metadata": {"source": "pdf_vector_rect"},
            }
        )

    connectors = _pdf_diagram_connectors(page, bbox)
    return _structured_diagram_content(
        nodes=nodes,
        edges=_infer_pdf_edges(nodes, connectors),
        connectors=connectors,
    )


_TABLE_LIKE_DIAGRAM_LABELS = {
    "연번",
    "질의",
    "답변",
    "현행",
    "현행",
    "개정",
    "비고",
    "항목",
    "제목",
    "세부인정사항",
    "구분",
    "EDI코드",
    "코드",
    "부위",
    "분류",
    "번호",
    "행위명",
    "수가",
    "본인부담률",
}


def _should_skip_pdf_diagram(diagram: dict[str, object]) -> bool:
    nodes = diagram.get("nodes")
    if not isinstance(nodes, list):
        return True
    if len(nodes) < 2:
        return True
    connectors = diagram.get("connectors")
    if _looks_like_pdf_text_line_boxes(nodes, connectors):
        return True
    labels = [
        _normalize_diagram_label(str(node.get("text", "")))
        for node in nodes
        if isinstance(node, Mapping)
    ]
    table_like = sum(label in _TABLE_LIKE_DIAGRAM_LABELS for label in labels)
    return table_like >= 3 and table_like / max(len(labels), 1) >= 0.5


def _looks_like_pdf_text_line_boxes(nodes: list[object], connectors: object) -> bool:
    if connectors:
        return False
    if len(nodes) < 3:
        return False
    long_text_line_count = 0
    for node in nodes:
        if not isinstance(node, Mapping):
            return False
        text = str(node.get("text", "")).strip()
        bbox = node.get("bbox")
        if not text or not isinstance(bbox, Mapping):
            return False
        try:
            width = float(bbox.get("width", 0.0))
            height = float(bbox.get("height", 0.0))
        except (TypeError, ValueError):
            return False
        if height <= 0:
            return False
        if width >= 120 and height <= 24 and width / height >= 8 and len(text) >= 12:
            long_text_line_count += 1
    return long_text_line_count == len(nodes)


def _normalize_diagram_label(text: str) -> str:
    return re.sub(r"\s+", "", text.strip())


def _pdf_diagram_connectors(
    page: object,
    bbox: tuple[float, float, float, float],
) -> list[dict[str, object]]:
    connectors: list[dict[str, object]] = []
    shapes = [
        ("line", item)
        for item in getattr(page, "lines", [])
    ] + [
        ("curve", item)
        for item in getattr(page, "curves", [])
    ]
    for kind, shape in shapes:
        shape_bbox = _pdf_shape_bbox(shape)
        if shape_bbox is None or not _bbox_in_cell(shape_bbox, bbox, tol=3.0):
            continue
        width = abs(shape_bbox[2] - shape_bbox[0])
        height = abs(shape_bbox[3] - shape_bbox[1])
        if max(width, height) < 10:
            continue
        connectors.append(
            {
                "id": f"c{len(connectors) + 1}",
                "type": kind,
                "bbox": _pdf_bbox_payload(shape_bbox),
                "points": _pdf_connector_points(shape, shape_bbox),
                "arrow": bool(shape.get("arrow")) if isinstance(shape, Mapping) else False,
                "metadata": {"source": f"pdf_vector_{kind}"},
            }
        )
    return connectors


def _infer_pdf_edges(
    nodes: list[dict[str, object]],
    connectors: list[dict[str, object]],
) -> list[dict[str, object]]:
    bbox_nodes = [
        (str(node.get("id", "")), bbox)
        for node in nodes
        if (bbox := _diagram_node_bbox(node)) is not None
    ]
    if len(bbox_nodes) < 2:
        return []

    edges: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for connector in connectors:
        points = connector.get("points")
        if not isinstance(points, list) or len(points) < 2:
            continue
        start = _diagram_point(points[0])
        end = _diagram_point(points[-1])
        if start is None or end is None:
            continue
        from_id = _nearest_node_id(start, bbox_nodes)
        to_id = _nearest_node_id(end, bbox_nodes)
        if from_id is None or to_id is None or from_id == to_id:
            continue
        connector_id = str(connector.get("id", ""))
        key = (from_id, to_id, connector_id)
        if key in seen:
            continue
        seen.add(key)
        edges.append(
            {
                "from": from_id,
                "to": to_id,
                "type": "arrow" if connector.get("arrow") else "line",
                "label": "",
                "confidence": "inferred_geometry",
                "connector_id": connector_id,
            }
        )
    return edges


def _pdf_shape_bbox(shape: object) -> tuple[float, float, float, float] | None:
    if not isinstance(shape, Mapping):
        return None
    try:
        x0 = float(shape.get("x0", 0.0))
        x1 = float(shape.get("x1", x0))
        top = float(shape.get("top", shape.get("y0", 0.0)))
        bottom = float(shape.get("bottom", shape.get("y1", top)))
    except (TypeError, ValueError):
        return None
    return (min(x0, x1), min(top, bottom), max(x0, x1), max(top, bottom))


def _pdf_bbox_payload(bbox: tuple[float, float, float, float]) -> dict[str, object]:
    x0, top, x1, bottom = bbox
    return {
        "x": round(x0, 3),
        "y": round(top, 3),
        "width": round(x1 - x0, 3),
        "height": round(bottom - top, 3),
        "unit": "pt",
    }


def _pdf_connector_points(
    shape: object,
    bbox: tuple[float, float, float, float],
) -> list[dict[str, float]]:
    if isinstance(shape, Mapping):
        try:
            return [
                {
                    "x": float(shape.get("x0", bbox[0])),
                    "y": float(shape.get("top", shape.get("y0", bbox[1]))),
                },
                {
                    "x": float(shape.get("x1", bbox[2])),
                    "y": float(shape.get("bottom", shape.get("y1", bbox[3]))),
                },
            ]
        except (TypeError, ValueError):
            pass
    x0, top, x1, bottom = bbox
    return [{"x": x0, "y": top}, {"x": x1, "y": bottom}]


def _diagram_node_bbox(node: dict[str, object]) -> dict[str, float] | None:
    bbox = node.get("bbox")
    if not isinstance(bbox, Mapping):
        return None
    try:
        x = float(bbox["x"])
        y = float(bbox["y"])
        width = float(bbox["width"])
        height = float(bbox["height"])
    except (KeyError, TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return {"x": x, "y": y, "width": width, "height": height}


def _diagram_point(point: object) -> dict[str, float] | None:
    if not isinstance(point, Mapping):
        return None
    try:
        return {"x": float(point["x"]), "y": float(point["y"])}
    except (KeyError, TypeError, ValueError):
        return None


def _nearest_node_id(
    point: dict[str, float],
    bbox_nodes: list[tuple[str, dict[str, float]]],
) -> str | None:
    best_id: str | None = None
    best_distance: float | None = None
    for node_id, bbox in bbox_nodes:
        distance = _point_bbox_distance_squared(point, bbox)
        if best_distance is None or distance < best_distance:
            best_id = node_id
            best_distance = distance
    return best_id


def _point_bbox_distance_squared(
    point: dict[str, float],
    bbox: dict[str, float],
) -> float:
    min_x = bbox["x"]
    max_x = bbox["x"] + bbox["width"]
    min_y = bbox["y"]
    max_y = bbox["y"] + bbox["height"]
    dx = max(min_x - point["x"], 0.0, point["x"] - max_x)
    dy = max(min_y - point["y"], 0.0, point["y"] - max_y)
    return dx * dx + dy * dy


def _diagram_source_text(diagram: dict[str, object]) -> str:
    nodes = diagram.get("nodes", [])
    if not isinstance(nodes, list):
        return ""
    lines = [
        text
        for text in (
            str(node.get("text", "")).strip()
            for node in nodes
            if isinstance(node, Mapping)
        )
        if text
    ]
    edge_lines = _diagram_edge_source_lines(diagram.get("edges", []))
    if edge_lines:
        lines.append("relations:")
        lines.extend(edge_lines)
    return "\n".join(lines)


def _diagram_edge_source_lines(edges: object) -> list[str]:
    if not isinstance(edges, list):
        return []
    lines: list[str] = []
    for edge in edges:
        if not isinstance(edge, Mapping):
            continue
        from_id = str(edge.get("from", "")).strip()
        to_id = str(edge.get("to", "")).strip()
        if not from_id or not to_id:
            continue
        label = str(edge.get("label", "")).strip()
        line = f"{from_id} -> {to_id}"
        if label:
            line = f"{line}: {label}"
        lines.append(line)
    return lines


def _detect_diagram_bboxes(
    page: object,
    table_bboxes: list[tuple[float, float, float, float]],
    *,
    container_bbox: tuple[float, float, float, float] | None = None,
    page_shapes: list[_PdfDiagramShape] | None = None,
) -> list[tuple[float, tuple[float, float, float, float]]]:
    shapes: list[_PdfDiagramShape] = []
    for x0, top, x1, bottom, kind in page_shapes or _pdf_page_diagram_shapes(page):
        bbox = (x0, top, x1, bottom)
        if kind == "rect" and ((x1 - x0) < 5 or (bottom - top) < 5):
            continue
        if kind != "rect" and max(x1 - x0, bottom - top) < 10:
            continue
        if container_bbox is not None and not _bbox_in_cell(bbox, container_bbox, tol=2.0):
            continue
        if any(
            x0 < tx1 and x1 > tx0 and top < tbottom and bottom > ttop
            for tx0, ttop, tx1, tbottom in table_bboxes
        ):
            continue
        shapes.append((x0, top, x1, bottom, kind))

    if not shapes:
        return []

    shapes.sort(key=lambda shape: shape[1])
    clusters: list[list[tuple[float, float, float, float, str]]] = [[shapes[0]]]
    for shape in shapes[1:]:
        previous_bottom = max(item[3] for item in clusters[-1])
        if shape[1] - previous_bottom <= 30:
            clusters[-1].append(shape)
        else:
            clusters.append([shape])

    results: list[tuple[float, tuple[float, float, float, float]]] = []
    if container_bbox is None:
        left = 0.0
        top_bound = 0.0
        right = float(getattr(page, "width", 0.0))
        bottom_bound = float(getattr(page, "height", 0.0))
    else:
        left, top_bound, right, bottom_bound = container_bbox
    for cluster in clusters:
        rect_count = sum(1 for item in cluster if item[4] == "rect")
        connector_count = sum(1 for item in cluster if item[4] != "rect")
        if rect_count < 2:
            continue
        if rect_count < 3 and connector_count == 0:
            continue
        x0 = max(left, min(rect[0] for rect in cluster) - 10)
        top = max(top_bound, min(rect[1] for rect in cluster) - 10)
        x1 = min(right, max(rect[2] for rect in cluster) + 10)
        bottom = min(bottom_bound, max(rect[3] for rect in cluster) + 10)
        area = max(0.0, x1 - x0) * max(0.0, bottom - top)
        text = _crop_text(page, x0, top, x1, bottom)
        if area > 0 and len(text) / area > 0.1:
            continue
        results.append((top, (x0, top, x1, bottom)))
    return results


def _pdf_page_diagram_shapes(page: object) -> list[_PdfDiagramShape]:
    shapes: list[_PdfDiagramShape] = []
    for kind, items in (
        ("rect", getattr(page, "rects", [])),
        ("line", getattr(page, "lines", [])),
        ("curve", getattr(page, "curves", [])),
    ):
        for shape in items:
            bbox = _pdf_shape_bbox(shape)
            if bbox is None:
                continue
            shapes.append((*bbox, kind))
    return shapes


def _render_page_to_png(
    data: bytes,
    page_idx: int,
    bbox: tuple[float, float, float, float],
    scale: float = _DEFAULT_RENDER_SCALE,
) -> bytes:
    try:
        import fitz
    except ImportError as exc:
        raise ImportError(
            "PDF page rendering requires pymupdf. Install the PDF extraction "
            "dependencies before parsing scanned pages or diagrams."
        ) from exc

    with fitz.open(stream=data, filetype="pdf") as doc:
        page = doc.load_page(page_idx)
        clip = fitz.Rect(*bbox)
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip)
        return pix.tobytes("png")


def _bbox_in_cell(
    sub_bbox: tuple[float, float, float, float],
    cell_bbox: tuple[float, float, float, float],
    tol: float = 2.0,
) -> bool:
    sub_x0, sub_top, sub_x1, sub_bottom = sub_bbox
    cell_x0, cell_top, cell_x1, cell_bottom = cell_bbox
    return (
        sub_x0 >= cell_x0 - tol
        and sub_x1 <= cell_x1 + tol
        and sub_top >= cell_top - tol
        and sub_bottom <= cell_bottom + tol
    )


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _bbox_near_equal(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
    margin: float = 3.0,
) -> bool:
    return all(abs(first[index] - second[index]) <= margin for index in range(4))


def _table_source_text(table: dict[str, object]) -> str:
    columns = table["columns"]
    rows = table["rows"]
    column_text = _build_column_source_labels(columns, _column_source_label, rows)
    lines: list[str] = []
    if columns:
        lines.append(f"table: {len(columns)} columns")
    for header_row in table.get("header_rows", []):
        cells = _table_source_cells(
            header_row["cells"],
            column_text,
            use_header_labels=False,
        )
        if cells:
            lines.append(f"header {header_row['index']}: " + "; ".join(cells))
    for row in rows:
        cells = _table_source_cells(
            row["cells"],
            column_text,
            use_header_labels=True,
        )
        if cells:
            lines.append(f"row {row['index']}: " + "; ".join(cells))
    return "\n".join(lines)


def _column_source_label(column: dict[str, object]) -> str:
    text = str(column["text"]).strip()
    return text or _column_coordinate_label(str(column["id"]))


def _table_source_cells(
    cells: list[dict[str, object]],
    column_text: dict[str, str],
    *,
    use_header_labels: bool,
) -> list[str]:
    result: list[str] = []
    for cell in cells:
        header = _cell_source_label(
            cell,
            column_text,
            use_header_labels=use_header_labels,
        )
        value = str(cell["text"])
        child_texts = [
            "nested table: " + _inline_table_source(child["content"])
            for child in cell["children"]
            if child.get("type", child.get("kind")) == "table"
        ]
        image_texts = [
            f"image: {child['content']['asset_id']}"
            for child in cell["children"]
            if child.get("type", child.get("kind")) == "image"
        ]
        diagram_texts = [
            "diagram: " + _inline_diagram_source(child["content"])
            for child in cell["children"]
            if child.get("type", child.get("kind")) == "diagram"
        ]
        combined = "; ".join(
            part for part in [value, *child_texts, *image_texts, *diagram_texts] if part
        )
        if combined:
            result.append(f"{header}: {combined}")
    return result


def _inline_diagram_source(diagram: object) -> str:
    if not isinstance(diagram, Mapping):
        return ""
    nodes = diagram.get("nodes")
    labels = [
        text
        for text in (
            str(node.get("text", "")).strip()
            for node in nodes
            if isinstance(node, Mapping)
        )
        if text
    ] if isinstance(nodes, list) else []
    if labels:
        return " / ".join(labels)
    asset_id = str(diagram.get("asset_id", "")).strip()
    return f"image: {asset_id}" if asset_id else ""


def _cell_source_label(
    cell: dict[str, object],
    column_text: dict[str, str],
    *,
    use_header_labels: bool,
) -> str:
    column_id = str(cell["column_id"])
    colspan = int(cell.get("colspan", 1))
    if use_header_labels:
        labels = [
            column_text.get(f"c{column_index}", f"col {column_index}")
            for column_index in range(
                _column_id_number(column_id),
                _column_id_number(column_id) + colspan,
            )
        ]
        common_prefix = _common_semantic_header_prefix(labels)
        if common_prefix is not None:
            return common_prefix
        group_label = _semantic_column_group_label(labels)
        if group_label is not None:
            return group_label
        if len(set(labels)) == 1 and _is_semantic_column_label(labels[0]):
            return labels[0]
        if colspan == 1 and _is_semantic_column_label(labels[0]):
            return labels[0]
    return _cell_coordinate_label(column_id, colspan)


def _cell_coordinate_label(column_id: str, colspan: int) -> str:
    start = _column_id_number(column_id)
    if colspan <= 1:
        return _column_coordinate_label(column_id)
    return f"cols {start}-{start + colspan - 1}"


def _column_coordinate_label(column_id: str) -> str:
    return f"col {_column_id_number(column_id)}"


def _column_id_number(column_id: str) -> int:
    try:
        return max(1, int(column_id.removeprefix("c")))
    except ValueError:
        return 1


def _inline_table_source(table: dict[str, object]) -> str:
    source = _table_source_text(table)
    return source.replace("\n", " / ")
