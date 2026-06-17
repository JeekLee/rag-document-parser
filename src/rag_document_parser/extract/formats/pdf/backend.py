from __future__ import annotations

from ...backend import ParsedDocument


class PdfBackend:
    supported_suffixes = (".pdf",)

    def parse(self, data: bytes, suffix: str) -> ParsedDocument:
        raise NotImplementedError("PDF extraction is not implemented yet")
