from __future__ import annotations

from .backend import DocumentBackend
from .formats.hwp5 import Hwp5Backend
from .formats.html import HtmlBackend
from .formats.hwpx import HwpxBackend
from .formats.markdown import MarkdownBackend
from .formats.pdf import PdfBackend


def default_backends() -> dict[str, DocumentBackend]:
    markdown_backend = MarkdownBackend()
    hwp5_backend = Hwp5Backend()
    html_backend = HtmlBackend()
    hwpx_backend = HwpxBackend()
    pdf_backend = PdfBackend()
    return {
        ".hwp": hwp5_backend,
        ".html": html_backend,
        ".htm": html_backend,
        ".hwpx": hwpx_backend,
        ".markdown": markdown_backend,
        ".md": markdown_backend,
        ".pdf": pdf_backend,
        ".txt": markdown_backend,
    }
