from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ....models import PendingAsset
from ...schema import structured_diagram as _structured_diagram_content
from .embedded_images import PdfEmbeddedImageExtractor
from .geometry import _bbox_in_cell
from .models import NestedResolution as _NestedResolution
from .models import PdfImage as _PdfImage
from .models import Segment as _Segment
from .table_extraction import PdfTableExtractor

_DEFAULT_RENDER_SCALE = 2.0
_PdfDiagramShape = tuple[float, float, float, float, str]
_RenderPage = Callable[
    [bytes, int, tuple[float, float, float, float], float],
    bytes,
]
_TABLE_LIKE_DIAGRAM_LABELS = {
    "연번", "질의", "답변", "현행", "개정", "비고", "항목", "제목",
    "세부인정사항", "구분", "EDI코드", "코드", "부위", "분류", "번호",
    "행위명", "수가", "본인부담률",
}
_IMAGE_EXTRACTOR = PdfEmbeddedImageExtractor()
_TABLE_EXTRACTOR = PdfTableExtractor()
_append_pdf_image_asset = _IMAGE_EXTRACTOR.append_asset
_clean_text = _TABLE_EXTRACTOR.clean_text
_crop_text = _TABLE_EXTRACTOR.crop_text


@dataclass(frozen=True)
class PdfDiagramExtractor:
    """Extracts PDF vector diagrams and table-cell diagram children."""

    def table_cell_children(
        self,
        data: bytes,
        page: object,
        page_idx: int,
        tables: list[object],
        nested: _NestedResolution,
        assets: list[PendingAsset],
        warnings: list[dict[str, Any]],
        *,
        shape_bbox: Callable[[object], tuple[float, float, float, float] | None] | None = None,
        render_page: _RenderPage | None = None,
    ) -> dict[int, dict[tuple[int, int], list[dict[str, object]]]]:
        return _table_cell_diagram_children(
            data,
            page,
            page_idx,
            tables,
            nested,
            assets,
            warnings,
            shape_bbox=shape_bbox,
            render_page=render_page,
        )

    def segments(
        self,
        data: bytes,
        page: object,
        page_idx: int,
        table_bboxes: list[tuple[float, float, float, float]],
        assets: list[PendingAsset],
        warnings: list[dict[str, Any]],
        *,
        render_page: _RenderPage | None = None,
    ) -> list[_Segment]:
        return _diagram_segments(
            data,
            page,
            page_idx,
            table_bboxes,
            assets,
            warnings,
            render_page=render_page,
        )

    def source_text(self, diagram: dict[str, object]) -> str:
        return _diagram_source_text(diagram)

    def shape_bbox(
        self,
        shape: object,
    ) -> tuple[float, float, float, float] | None:
        return _pdf_shape_bbox(shape)

    def render_page(
        self,
        data: bytes,
        page_idx: int,
        bbox: tuple[float, float, float, float],
        scale: float = _DEFAULT_RENDER_SCALE,
    ) -> bytes:
        return _render_page_to_png(data, page_idx, bbox, scale)


def _table_cell_diagram_children(
    data: bytes,
    page: object,
    page_idx: int,
    tables: list[object],
    nested: _NestedResolution,
    assets: list[PendingAsset],
    warnings: list[dict[str, Any]],
    *,
    shape_bbox: Callable[
        [object], tuple[float, float, float, float] | None
    ] | None = None,
    render_page: _RenderPage | None = None,
) -> dict[int, dict[tuple[int, int], list[dict[str, object]]]]:
    cell_children: dict[int, dict[tuple[int, int], list[dict[str, object]]]] = {}
    page_shapes = _pdf_page_diagram_shapes(page, shape_bbox=shape_bbox)
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
                        render_page=render_page,
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
    *,
    render_page: _RenderPage | None = None,
) -> dict[str, object] | None:
    structured = _structured_diagram_from_pdf(page, bbox)
    source_text = _diagram_source_text(structured)
    if source_text and _should_skip_pdf_diagram(structured):
        return None

    asset_id: str | None = None
    render = render_page or _render_page_to_png
    try:
        asset_id = _append_pdf_image_asset(
            assets,
            _PdfImage(
                data=render(data, page_idx, bbox, _DEFAULT_RENDER_SCALE),
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


def _diagram_segments(
    data: bytes,
    page: object,
    page_idx: int,
    table_bboxes: list[tuple[float, float, float, float]],
    assets: list[PendingAsset],
    warnings: list[dict[str, Any]],
    *,
    render_page: _RenderPage | None = None,
) -> list[_Segment]:
    segments: list[_Segment] = []
    render = render_page or _render_page_to_png
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
                    data=render(data, page_idx, bbox, _DEFAULT_RENDER_SCALE),
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
    source_shapes = (
        page_shapes
        if page_shapes is not None
        else _pdf_page_diagram_shapes(page)
    )
    for x0, top, x1, bottom, kind in source_shapes:
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


def _pdf_page_diagram_shapes(
    page: object,
    *,
    shape_bbox: Callable[
        [object], tuple[float, float, float, float] | None
    ] | None = None,
) -> list[_PdfDiagramShape]:
    shapes: list[_PdfDiagramShape] = []
    bbox_from_shape = shape_bbox or _pdf_shape_bbox
    for kind, items in (
        ("rect", getattr(page, "rects", [])),
        ("line", getattr(page, "lines", [])),
        ("curve", getattr(page, "curves", [])),
    ):
        for shape in items:
            bbox = bbox_from_shape(shape)
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
