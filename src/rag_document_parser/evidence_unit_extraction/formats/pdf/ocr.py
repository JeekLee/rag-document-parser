from __future__ import annotations

import io
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from ...ocr import OcrGateway, OcrOutput, OcrResult

_CJK = re.compile(r"[가-힣一-鿿㐀-䶿]")


class OcrBatchResults(dict[int, str]):
    def __init__(self) -> None:
        super().__init__()
        self.failed_pages: list[dict[str, object]] = []


def run_ocr_pages(
    scanned: list[tuple[int, bytes]],
    data: bytes,
    max_workers: int | None,
    ocr_gateway: OcrGateway | None,
) -> OcrBatchResults:
    if not scanned:
        return OcrBatchResults()

    def run_ocr(image: bytes, page_idx: int) -> OcrOutput:
        if ocr_gateway is not None:
            return ocr_gateway(image, page_idx)
        return ocr_page(data, image, page_idx)

    if max_workers is None or max_workers <= 1 or len(scanned) == 1:
        results = OcrBatchResults()
        for page_idx, image in scanned:
            _record_ocr_result(results, page_idx, lambda: run_ocr(image, page_idx))
        return results

    results = OcrBatchResults()
    workers = min(max_workers, len(scanned))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {
            executor.submit(run_ocr, image, page_idx): page_idx
            for page_idx, image in scanned
        }
        for future in as_completed(future_to_idx):
            page_idx = future_to_idx[future]
            _record_ocr_result(results, page_idx, future.result)
    results.failed_pages.sort(key=lambda failure: int(failure["page"]))
    return results


def _record_ocr_result(
    results: OcrBatchResults,
    page_idx: int,
    get_text: Callable[[], OcrOutput],
) -> None:
    try:
        raw_result = get_text()
        if isinstance(raw_result, OcrResult):
            if raw_result.status == "extracted":
                results[page_idx] = raw_result.text
                return
            results[page_idx] = ""
            results.failed_pages.append(
                {
                    "page": page_idx,
                    "stage": "ocr",
                    "status": raw_result.status,
                    "reason": raw_result.reason,
                }
            )
            return
        results[page_idx] = raw_result or ""
        if not results[page_idx].strip():
            results.failed_pages.append(
                {"page": page_idx, "stage": "ocr", "message": "empty OCR result"}
            )
    except Exception as exc:
        results[page_idx] = ""
        results.failed_pages.append(
            {"page": page_idx, "stage": "ocr", "message": str(exc)}
        )


def ocr_warnings(failed_pages: list[dict[str, object]]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    for failure in failed_pages:
        page = int(failure.get("page", 0))
        status = failure.get("status")
        if status in {"no_text", "uncertain"}:
            warning_type = "pdf_ocr_empty" if status == "no_text" else "pdf_ocr_uncertain"
            warnings.append(
                {
                    "type": warning_type,
                    "severity": "low" if status == "no_text" else "medium",
                    "page": page + 1,
                    "stage": failure.get("stage", "ocr"),
                    "reason": str(failure.get("reason", "")),
                    "message": (
                        "Structured OCR reported no readable text."
                        if status == "no_text"
                        else "Structured OCR could not determine whether text is readable."
                    ),
                }
            )
            continue
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


def ocr_fallback_reason(
    page: object,
    *,
    allow_degraded_native: bool,
) -> str | None:
    if _is_scanned_page(page):
        return "scanned"
    if allow_degraded_native and _has_degraded_native_text(page):
        return "degraded_native_text"
    return None


def _is_scanned_page(page: object, min_text_chars: int = 30) -> bool:
    text_chars = 0
    for char in getattr(page, "chars", []):
        if str(char.get("text", "")).strip():
            text_chars += 1
            if text_chars >= min_text_chars:
                return False

    page_width = float(getattr(page, "width", 0.0) or 0.0)
    page_height = float(getattr(page, "height", 0.0) or 0.0)
    for image in getattr(page, "images", []):
        image_width = float(image.get("x1", 0.0) - image.get("x0", 0.0))
        image_height = float(image.get("y1", 0.0) - image.get("y0", 0.0))
        if (
            page_width
            and page_height
            and image_width / page_width > 0.7
            and image_height / page_height > 0.7
        ):
            return True
    return False


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


def ocr_page(data: bytes, png: bytes, page_idx: int, lang: str = "kor+eng") -> str:
    if png:
        text = _ocr_png(png, lang)
        if text:
            return text
    text = _ocr_via_pypdf(data, page_idx, lang)
    if text:
        return text
    return _ocr_via_pdf2image(data, page_idx, lang)


def ocr_page_with_vision(
    data: bytes,
    png: bytes,
    page_idx: int,
    ocr_gateway: OcrGateway,
) -> OcrOutput:
    if png:
        try:
            result = vision_ocr_png(png, ocr_gateway, page_idx)
        except Exception:
            result = None
        if isinstance(result, OcrResult):
            if result.status != "uncertain":
                return result
            fallback_text = ocr_page(data, png, page_idx)
            return fallback_text or result
        if result:
            return result
    return ocr_page(data, png, page_idx)


def vision_ocr_png(
    png: bytes,
    ocr_gateway: OcrGateway,
    page_idx: int,
) -> OcrResult:
    result = ocr_gateway(png, page_idx)
    if not isinstance(result, OcrResult):
        text = result or ""
        if not text.strip():
            return OcrResult(status="no_text", text="", reason="empty OCR result")
        result = OcrResult(status="extracted", text=text, reason="legacy OCR result")
    if result.status != "extracted":
        return result
    cleaned = clean_vision_ocr_text(result.text)
    if not cleaned:
        return OcrResult(
            status="uncertain",
            text="",
            reason="structured OCR returned empty extracted text",
        )
    return OcrResult(status="extracted", text=cleaned, reason=result.reason)


def clean_vision_ocr_text(text: str) -> str:
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
