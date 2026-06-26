from __future__ import annotations

from dataclasses import dataclass

from ...backend import ParsedDocument


@dataclass
class HtmlBackend:
    supported_suffixes = (".html", ".htm")

    def parse(self, data: bytes, suffix: str) -> ParsedDocument:
        return ParsedDocument(units=[])
