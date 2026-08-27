from __future__ import annotations

import io
from dataclasses import dataclass

from ...backend import ParsedDocument
from ...ocr import OcrFn
from .binary_reader import BinEntry as _BinEntry
from .binary_reader import Hwp5BinaryReader
from .parsed import ParsedBlocks as _ParsedBlocks
from .section_parser import Hwp5SectionParser
from .source_projection import Hwp5TableSourceProjector
from .table_extraction import Cell as _Cell
from .table_extraction import Hwp5TableBuilder

_HWP5_BINARY_READER = Hwp5BinaryReader()
_HWP5_SECTION_PARSER = Hwp5SectionParser()
_HWP5_SOURCE_PROJECTOR = Hwp5TableSourceProjector()
_HWP5_TABLE_BUILDER = Hwp5TableBuilder()

# Preserve the backend's historical private helpers for downstream callers while
# delegating their implementation to focused services.
_decode_stream = _HWP5_BINARY_READER.decode_stream
_detect_mime = _HWP5_BINARY_READER.detect_mime
_iter_records = _HWP5_BINARY_READER.iter_records
_load_bin_data = _HWP5_BINARY_READER.load_bin_data
_mime_to_ext = _HWP5_BINARY_READER.mime_to_ext
_parse_section = _HWP5_SECTION_PARSER.parse
_structured_table = _HWP5_TABLE_BUILDER.build
_table_source_text = _HWP5_SOURCE_PROJECTOR.project


@dataclass(frozen=True)
class Hwp5Backend:
    supported_suffixes = (".hwp",)
    ocr_fn: OcrFn | None = None

    def parse(self, data: bytes, suffix: str) -> ParsedDocument:
        try:
            import olefile
        except (ImportError, ModuleNotFoundError) as exc:
            raise NotImplementedError(
                "HWP5 extraction requires the optional 'olefile' dependency."
            ) from exc

        ole = olefile.OleFileIO(io.BytesIO(data))
        try:
            flags = _HWP5_BINARY_READER.read_flags(ole)
            compressed = bool(flags & 0x1)
            bin_entries = _HWP5_BINARY_READER.parse_bin_entries(ole, compressed)
            bin_streams = _HWP5_BINARY_READER.load_bin_data(ole, compressed)
            parsed = _ParsedBlocks()
            for stream_name in _HWP5_BINARY_READER.section_streams(ole):
                raw = ole.openstream(stream_name).read()
                section = _HWP5_SECTION_PARSER.parse(
                    _HWP5_BINARY_READER.decode_stream(raw, compressed),
                    bin_entries=bin_entries,
                    bin_streams=bin_streams,
                    asset_offset=len(parsed.assets),
                )
                parsed.extend(section)
        finally:
            ole.close()

        return parsed.to_document(ocr_fn=self.ocr_fn)
