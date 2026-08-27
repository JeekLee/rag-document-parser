from __future__ import annotations

from .backend import DocumentBackend, ParsedDocument
from .ocr import OcrGateway, OcrResult, VisionOcr
from .registry import default_backends

__all__ = [
    "DocumentBackend",
    "OcrResult",
    "OcrGateway",
    "ParsedDocument",
    "VisionOcr",
    "default_backends",
]
