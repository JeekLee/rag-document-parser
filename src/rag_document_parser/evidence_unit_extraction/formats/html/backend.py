from __future__ import annotations

from dataclasses import dataclass

from ...backend import ParsedDocument
from .document_parser import HtmlDocumentParser


@dataclass
class HtmlBackend:
    supported_suffixes = (".html", ".htm")

    def parse(self, data: bytes, suffix: str) -> ParsedDocument:
        return HtmlDocumentParser().parse(data, suffix)
