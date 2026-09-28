from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

from ....llm import LlmConfig
from ....models import PendingAsset
from ...backend import ParsedDocument
from ...ocr import OcrFn, OcrGateway, OcrOutput, VisionOcr
from .diagram import PdfDiagramExtractor
from .embedded_images import PdfEmbeddedImageExtractor
from .models import NestedResolution as _NestedResolution
from .models import Segment as _Segment
from .ocr import (
    ocr_page_with_vision as _ocr_page_with_vision,
    ocr_fallback_reason as _ocr_fallback_reason,
    ocr_warnings as _ocr_warnings,
    run_ocr_pages as _ocr_pages,
)
from .ocr_text import PdfOcrTextParser
from .segment_processing import PdfSegmentProcessor
from .source_projection import PdfTableSourceProjector
from .table_extraction import PdfTableExtractor
from .table_normalization import PdfTableNormalizer

_SCANNED_OCR_RENDER_SCALE = 3.0
_PDF_DIAGRAM_EXTRACTOR = PdfDiagramExtractor()
_PDF_IMAGE_EXTRACTOR = PdfEmbeddedImageExtractor()
_PDF_OCR_TEXT_PARSER = PdfOcrTextParser()
_PDF_SEGMENT_PROCESSOR = PdfSegmentProcessor()
_PDF_TABLE_EXTRACTOR = PdfTableExtractor()
_PDF_TABLE_NORMALIZER = PdfTableNormalizer()
_PDF_TABLE_SOURCE_PROJECTOR = PdfTableSourceProjector()

_cell_has_content = _PDF_TABLE_NORMALIZER.cell_has_content
_cell_source_label = _PDF_TABLE_SOURCE_PROJECTOR.cell_label
_clean_cell = _PDF_TABLE_EXTRACTOR.clean_cell
_clean_text = _PDF_TABLE_EXTRACTOR.clean_text
_expand_parallel_code_action_rows = _PDF_TABLE_NORMALIZER.expand_parallel_rows
_expand_text_segments = _PDF_SEGMENT_PROCESSOR.expand_text
_extract_page_images = _PDF_IMAGE_EXTRACTOR.extract_page
_find_tables = _PDF_TABLE_EXTRACTOR.find
_image_segments_and_cell_children = _PDF_IMAGE_EXTRACTOR.split_segments_and_cell_children
_merge_cell_children = _PDF_IMAGE_EXTRACTOR.merge_cell_children
_merge_continuation_tables = _PDF_SEGMENT_PROCESSOR.merge_continuation_tables
_needs_pymupdf_blank_cell_fallback = _PDF_TABLE_EXTRACTOR.needs_text_fallback
_ocr_text_segments = _PDF_OCR_TEXT_PARSER.parse
_page_segments_ordered = _PDF_SEGMENT_PROCESSOR.order_page
_pdf_reader = _PDF_IMAGE_EXTRACTOR.reader
_pdf_shape_bbox = _PDF_DIAGRAM_EXTRACTOR.shape_bbox
_promote_ultrasound_code_matrix = _PDF_TABLE_NORMALIZER.promote_ultrasound_codes
_pymupdf_table_rows_by_pdfplumber_index = _PDF_TABLE_EXTRACTOR.fallback_rows
_render_page_to_png = _PDF_DIAGRAM_EXTRACTOR.render_page
_resolve_nested_tables = _PDF_TABLE_EXTRACTOR.resolve_nested
_segments_to_units = _PDF_SEGMENT_PROCESSOR.to_units
_simple_cell = _PDF_TABLE_NORMALIZER.simple_cell
_table_source_text = _PDF_TABLE_SOURCE_PROJECTOR.project


def _diagram_segments(
    data: bytes,
    page: object,
    page_idx: int,
    table_bboxes: list[tuple[float, float, float, float]],
    assets: list[PendingAsset],
    warnings: list[dict[str, Any]],
) -> list[_Segment]:
    return _PDF_DIAGRAM_EXTRACTOR.segments(
        data,
        page,
        page_idx,
        table_bboxes,
        assets,
        warnings,
        render_page=_render_page_to_png,
    )


def _table_cell_diagram_children(
    data: bytes,
    page: object,
    page_idx: int,
    tables: list[object],
    nested: _NestedResolution,
    assets: list[PendingAsset],
    warnings: list[dict[str, Any]],
) -> dict[int, dict[tuple[int, int], list[dict[str, object]]]]:
    return _PDF_DIAGRAM_EXTRACTOR.table_cell_children(
        data,
        page,
        page_idx,
        tables,
        nested,
        assets,
        warnings,
        shape_bbox=_pdf_shape_bbox,
        render_page=_render_page_to_png,
    )


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


@dataclass
class PdfBackend:
    supported_suffixes = (".pdf",)
    max_ocr_workers: int = 4
    ocr_fn: OcrFn | None = None
    ocr_llm: LlmConfig | None = None
    ocr_gateway: OcrGateway | None = None

    def __post_init__(self) -> None:
        configured = sum(
            option is not None
            for option in (self.ocr_fn, self.ocr_llm, self.ocr_gateway)
        )
        if configured > 1:
            raise ValueError("configure only one of ocr_fn, ocr_llm, or ocr_gateway")

    def _configured_ocr_gateway(self, data: bytes) -> OcrGateway | None:
        if self.ocr_gateway is not None:
            return self.ocr_gateway
        if self.ocr_fn is not None:
            return self.ocr_fn
        if self.ocr_llm is None:
            return None
        vision_ocr = VisionOcr(self.ocr_llm)

        def vision_with_local_fallback(image: bytes, page_idx: int) -> OcrOutput:
            return _ocr_page_with_vision(data, image, page_idx, vision_ocr)

        return vision_with_local_fallback

    def parse(self, data: bytes, suffix: str) -> ParsedDocument:
        try:
            import pdfplumber
        except ImportError as exc:
            raise NotImplementedError(
                "PDF extraction requires pdfplumber. Install the PDF extraction "
                "dependencies before parsing .pdf files."
            ) from exc

        ocr_gateway = self._configured_ocr_gateway(data)
        assets: list[PendingAsset] = []
        warnings: list[dict[str, Any]] = []

        with pdfplumber.open(io.BytesIO(data)) as pdf:
            page_segments: list[list[_Segment]] = [[] for _ in pdf.pages]
            scanned: list[tuple[int, bytes]] = []
            pdf_reader: object | None = None

            for page_idx, page in enumerate(pdf.pages):
                ocr_fallback_reason = _ocr_fallback_reason(
                    page,
                    allow_degraded_native=ocr_gateway is not None,
                )
                if ocr_fallback_reason is not None:
                    png = (
                        _render_scanned_page_for_ocr(
                            data,
                            page_idx,
                            page,
                            warnings,
                        )
                        if ocr_gateway is not None
                        else b""
                    )
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
                table_text_fallbacks = (
                    _pymupdf_table_rows_by_pdfplumber_index(data, page_idx, tables)
                    if _needs_pymupdf_blank_cell_fallback(tables)
                    else {}
                )
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
                image_segments, cell_image_children = (
                    _image_segments_and_cell_children(
                        img_items,
                        tables,
                        assets,
                        page_idx,
                        warnings,
                    )
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
                        table_text_fallbacks,
                    )
                )

            ocr_by_page = _ocr_pages(
                scanned,
                data,
                self.max_ocr_workers,
                ocr_gateway,
            )
            for page_idx, text in ocr_by_page.items():
                if text.strip():
                    segments, ocr_parse_warnings = _ocr_text_segments(
                        text,
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
