from __future__ import annotations

from ...backend import ParsedDocument


class Hwp5Backend:
    supported_suffixes = (".hwp",)

    def parse(self, data: bytes, suffix: str) -> ParsedDocument:
        raise NotImplementedError("HWP5 extraction is not implemented yet")
