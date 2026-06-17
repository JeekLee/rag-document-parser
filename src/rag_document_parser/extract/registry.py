from __future__ import annotations

from .backend import DocumentBackend
from .formats.hwpx import HwpxBackend
from .formats.markdown import MarkdownBackend


def default_backends() -> dict[str, DocumentBackend]:
    markdown_backend = MarkdownBackend()
    hwpx_backend = HwpxBackend()
    return {
        ".hwpx": hwpx_backend,
        ".markdown": markdown_backend,
        ".md": markdown_backend,
        ".txt": markdown_backend,
    }
