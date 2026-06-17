from __future__ import annotations

import io
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable

from ....models import Evidence, EvidenceUnit, PendingAsset, SourceEvidence
from ...backend import ParsedDocument


_PAGE_NUM_RE = re.compile(r"(?m)^\s*(?:-\s*)?\d+\s*(?:-\s*)?$")
_CJK = re.compile(r"[가-힣一-鿿㐀-䶿]")
_MIN_IMAGE_AREA_PT2 = 2500


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


@dataclass(frozen=True)
class _NestedResolution:
    suppressed: set[int]
    children: dict[int, dict[tuple[int, int], list[int]]]


class _OcrResults(dict[int, str]):
    def __init__(self) -> None:
        super().__init__()
        self.failed_pages: list[dict[str, object]] = []


@dataclass
class PdfBackend:
    supported_suffixes = (".pdf",)
    max_ocr_workers: int = 4
    ocr_fn: Callable[[bytes, int], str] | None = None

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
                if _is_scanned_page(page):
                    png = _render_scanned_page_for_ocr(
                        data,
                        page_idx,
                        page,
                        warnings,
                    ) if self.ocr_fn is not None else b""
                    scanned.append((page_idx, png))
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
                for top, bbox in _detect_diagram_bboxes(page, table_bboxes):
                    try:
                        img_items.append(
                            (
                                top,
                                _PdfImage(
                                    data=_render_page_to_png(data, page_idx, bbox),
                                    mime="image/png",
                                    ext="png",
                                    is_diagram=True,
                                    metadata={"source": "diagram", "bbox": bbox},
                                ),
                            )
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

                image_segments: list[_Segment] = []
                for top, item in sorted(img_items, key=lambda candidate: candidate[0]):
                    asset_id = f"img-{len(assets) + 1:04d}"
                    metadata = dict(getattr(item, "metadata", {}) or {})
                    metadata.setdefault("page", page_idx + 1)
                    if getattr(item, "is_diagram", False):
                        metadata["is_diagram"] = True
                    assets.append(
                        PendingAsset(
                            id=asset_id,
                            kind="image",
                            data=bytes(getattr(item, "data")),
                            mime=str(getattr(item, "mime")),
                            ext=str(getattr(item, "ext")),
                            metadata=metadata,
                        )
                    )
                    image_segments.append(
                        _Segment(
                            top=float(top),
                            bottom=float(top) + 1.0,
                            kind="image",
                            payload={"asset_id": asset_id, "caption": None},
                            page=page_idx + 1,
                        )
                    )

                page_segments[page_idx].extend(
                    _page_segments_ordered(page, page_idx + 1, tables, image_segments)
                )

            ocr_by_page = _ocr_pages(
                scanned,
                data,
                self.max_ocr_workers,
                self.ocr_fn,
            )
            for page_idx, text in ocr_by_page.items():
                cleaned = _clean_text(text)
                if cleaned:
                    page_segments[page_idx].append(
                        _Segment(
                            top=0.0,
                            bottom=0.0,
                            kind="text",
                            payload=cleaned,
                            page=page_idx + 1,
                        )
                    )
            warnings.extend(_ocr_warnings(ocr_by_page.failed_pages))

        return ParsedDocument(
            units=_segments_to_units(page_segments),
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


def _page_segments_ordered(
    page: object,
    page_number: int,
    tables: list[object],
    image_segments: list[_Segment],
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
        )
        if structured is None:
            continue
        text_box = _single_cell_text_table_text(structured)
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
                if not text:
                    continue
                units.append(
                    EvidenceUnit(
                        id=f"b{block_index}",
                        type="text",
                        source=SourceEvidence(kind="text", text=text),
                        evidence=Evidence(kind="text", format="plain", content=text),
                        metadata={
                            "common": {
                                "chunk_kind": "text",
                                "section_path": [],
                                "display_format": "plain",
                            },
                            "pdf": {"page": segment.page},
                        },
                    )
                )
                block_index += 1
                continue
            if segment.kind == "table":
                table = segment.payload
                table_id = f"t{table_index}"
                headers = [str(column["text"]) for column in table["columns"]]
                units.append(
                    EvidenceUnit(
                        id=f"b{block_index}",
                        type="table",
                        source=SourceEvidence(
                            kind="table",
                            text=_table_source_text(table),
                        ),
                        evidence=Evidence(
                            kind="table",
                            format="structured_table",
                            content=table,
                        ),
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
                            "pdf": {"page": segment.page},
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
                        source=SourceEvidence(
                            kind="image",
                            text=f"image: {asset_id}",
                        ),
                        evidence=Evidence(
                            kind="image",
                            format="asset_ref",
                            content=dict(segment.payload),
                        ),
                        metadata={
                            "common": {
                                "chunk_kind": "image",
                                "section_path": [],
                                "display_format": "image",
                            },
                            "asset": {"asset_id": asset_id},
                            "pdf": {"page": segment.page},
                        },
                    )
                )
                block_index += 1

    return units


def _structured_table_from_pdf_table(
    page: object,
    tables: list[object],
    table_idx: int,
    nested: _NestedResolution,
    seen: set[int],
) -> dict[str, object] | None:
    if table_idx in seen:
        return None
    table = tables[table_idx]
    raw_rows = _table_rows(table)
    if not raw_rows:
        return None
    column_count = max((len(row) for row in raw_rows), default=0)
    if column_count == 0:
        return {"caption": None, "columns": [], "rows": []}

    columns = [
        {
            "id": f"c{index}",
            "text": raw_rows[0][index - 1] if index - 1 < len(raw_rows[0]) else "",
        }
        for index in range(1, column_count + 1)
    ]
    child_map = nested.children.get(table_idx, {})
    header_cells = _row_evidence_cells(
        page,
        tables,
        table_idx,
        row_index=0,
        raw_row=raw_rows[0],
        columns=columns,
        child_map=child_map,
        nested=nested,
        seen=seen | {table_idx},
    )
    rows: list[dict[str, object]] = []
    for raw_index, raw_row in enumerate(raw_rows[1:], start=1):
        cells = _row_evidence_cells(
            page,
            tables,
            table_idx,
            row_index=raw_index,
            raw_row=raw_row,
            columns=columns,
            child_map=child_map,
            nested=nested,
            seen=seen | {table_idx},
        )
        if not any(str(cell["text"]).strip() or cell["children"] for cell in cells):
            continue
        rows.append({"index": len(rows) + 1, "cells": cells})

    result: dict[str, object] = {
        "caption": None,
        "columns": columns,
        "rows": rows,
    }
    if header_cells:
        result["header_rows"] = [{"index": 1, "cells": header_cells}]
    return result


def _row_evidence_cells(
    page: object,
    tables: list[object],
    table_idx: int,
    *,
    row_index: int,
    raw_row: list[str],
    columns: list[dict[str, str]],
    child_map: dict[tuple[int, int], list[int]],
    nested: _NestedResolution,
    seen: set[int],
) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    row_cells = _row_cells(tables[table_idx], row_index)
    for column_index, column in enumerate(columns):
        text = raw_row[column_index] if column_index < len(raw_row) else ""
        child_indices = child_map.get((row_index, column_index), [])
        children = [
            child
            for child in (
                _nested_table_child(page, tables, child_idx, nested, seen)
                for child_idx in child_indices
            )
            if child is not None
        ]
        if children:
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
                "rowspan": 1,
                "colspan": 1,
                "children": children,
            }
        )
    return cells


def _nested_table_child(
    page: object,
    tables: list[object],
    table_idx: int,
    nested: _NestedResolution,
    seen: set[int],
) -> dict[str, object] | None:
    structured = _structured_table_from_pdf_table(page, tables, table_idx, nested, seen)
    if structured is None:
        return None
    return {
        "kind": "table",
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


def _row_cells(table: object, row_index: int) -> list[tuple[float, float, float, float] | None]:
    rows = getattr(table, "rows", [])
    if row_index >= len(rows):
        return []
    return list(getattr(rows[row_index], "cells", []) or [])


def _single_cell_text_table_text(table: dict[str, object]) -> str | None:
    if table["rows"]:
        return None
    header_rows = table.get("header_rows")
    if not isinstance(header_rows, list) or len(header_rows) != 1:
        return None
    cells = header_rows[0].get("cells")
    if not isinstance(cells, list) or len(cells) != 1:
        return None
    cell = cells[0]
    if not isinstance(cell, dict) or cell.get("children"):
        return None
    text = str(cell.get("text", "")).strip()
    return text or None


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


def _clean_cell(cell: object) -> str:
    if cell is None:
        return ""
    text = _join_lines(str(cell))
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


def _ocr_pages(
    scanned: list[tuple[int, bytes]],
    data: bytes,
    max_workers: int | None,
    ocr_fn: Callable[[bytes, int], str] | None,
) -> _OcrResults:
    if not scanned:
        return _OcrResults()

    def run_ocr(png: bytes, page_idx: int) -> str:
        if ocr_fn is not None:
            return ocr_fn(png, page_idx)
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


def _ocr_page(data: bytes, png: bytes, page_idx: int, lang: str = "kor+eng") -> str:
    if png:
        text = _ocr_png(png, lang)
        if text:
            return text
    text = _ocr_via_pypdf(data, page_idx, lang)
    if text:
        return text
    return _ocr_via_pdf2image(data, page_idx, lang)


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
        item = _pil_to_pdf_image(
            image,
            metadata={"source": "embedded", "name": name, "index_hint": image_index},
        )
        results.append((float(image_info.get("top", 0.0)), item))
        image_index += 1
    return sorted(results, key=lambda item: item[0])


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


def _detect_diagram_bboxes(
    page: object,
    table_bboxes: list[tuple[float, float, float, float]],
) -> list[tuple[float, tuple[float, float, float, float]]]:
    rects: list[tuple[float, float, float, float]] = []
    for rect in getattr(page, "rects", []):
        x0 = float(rect["x0"])
        top = float(rect["top"])
        x1 = float(rect["x1"])
        bottom = float(rect["bottom"])
        if (x1 - x0) < 5 or (bottom - top) < 5:
            continue
        if any(
            x0 < tx1 and x1 > tx0 and top < tbottom and bottom > ttop
            for tx0, ttop, tx1, tbottom in table_bboxes
        ):
            continue
        rects.append((x0, top, x1, bottom))

    if not rects:
        return []

    rects.sort(key=lambda rect: rect[1])
    clusters: list[list[tuple[float, float, float, float]]] = [[rects[0]]]
    for rect in rects[1:]:
        previous_bottom = max(item[3] for item in clusters[-1])
        if rect[1] - previous_bottom <= 20:
            clusters[-1].append(rect)
        else:
            clusters.append([rect])

    results: list[tuple[float, tuple[float, float, float, float]]] = []
    page_width = float(getattr(page, "width", 0.0))
    page_height = float(getattr(page, "height", 0.0))
    for cluster in clusters:
        if len(cluster) < 3:
            continue
        x0 = max(0.0, min(rect[0] for rect in cluster) - 10)
        top = max(0.0, min(rect[1] for rect in cluster) - 10)
        x1 = min(page_width, max(rect[2] for rect in cluster) + 10)
        bottom = min(page_height, max(rect[3] for rect in cluster) + 10)
        area = max(0.0, x1 - x0) * max(0.0, bottom - top)
        text = _crop_text(page, x0, top, x1, bottom)
        if area > 0 and len(text) / area > 0.1:
            continue
        results.append((top, (x0, top, x1, bottom)))
    return results


def _render_page_to_png(
    data: bytes,
    page_idx: int,
    bbox: tuple[float, float, float, float],
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
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), clip=clip)
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
    column_text = {
        str(column["id"]): _column_source_label(column)
        for column in columns
    }
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
            if child.get("kind") == "table"
        ]
        combined = "; ".join(part for part in [value, *child_texts] if part)
        if combined:
            result.append(f"{header}: {combined}")
    return result


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
        if len(set(labels)) == 1 and _is_semantic_column_label(labels[0]):
            return labels[0]
        if colspan == 1 and _is_semantic_column_label(labels[0]):
            return labels[0]
    return _cell_coordinate_label(column_id, colspan)


def _is_semantic_column_label(label: str) -> bool:
    return bool(label) and not label.startswith("col ")


def _common_semantic_header_prefix(labels: list[str]) -> str | None:
    if not labels or not all(_is_semantic_column_label(label) for label in labels):
        return None
    split_labels = [label.split(" / ") for label in labels]
    prefix: list[str] = []
    for parts in zip(*split_labels):
        if len(set(parts)) != 1:
            break
        prefix.append(parts[0])
    if not prefix:
        return None
    return " / ".join(prefix)


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
