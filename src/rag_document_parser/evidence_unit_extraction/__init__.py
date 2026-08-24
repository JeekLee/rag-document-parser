from __future__ import annotations

from .backend import DocumentBackend, ParsedDocument
from .ocr import OcrResult, VisionOcr
from .registry import default_backends

__all__ = [
    "DocumentBackend",
    "OcrResult",
    "ParsedDocument",
    "VisionOcr",
    "default_backends",
]
