from __future__ import annotations

import re
import struct
import zlib
from dataclasses import dataclass

_TAG_BIN_DATA = 0x12


@dataclass(frozen=True)
class Hwp5BinaryReader:
    """Reads HWP5 OLE streams, compression flags, and BinData records."""

    def read_flags(self, ole: object) -> int:
        return _read_flags(ole)

    def section_streams(self, ole: object) -> list[str]:
        return _section_streams(ole)

    def parse_bin_entries(
        self,
        ole: object,
        compressed: bool,
    ) -> dict[int, BinEntry]:
        return _parse_doc_info_bin_data(ole, compressed)

    def load_bin_data(
        self,
        ole: object,
        compressed: bool,
    ) -> dict[int, tuple[bytes, str]]:
        return _load_bin_data(ole, compressed)

    def decode_stream(self, raw: bytes, compressed: bool) -> bytes:
        return _decode_stream(raw, compressed)

    def iter_records(self, data: bytes):
        return _iter_records(data)

    def detect_mime(self, data: bytes) -> str:
        return _detect_mime(data)

    def mime_to_ext(self, mime: str) -> str:
        return _mime_to_ext(mime)


@dataclass(frozen=True)
class BinEntry:
    storage_id: int
    ext: str


def _iter_records(data: bytes):
    offset = 0
    length = len(data)
    while offset + 4 <= length:
        header = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        tag_id = header & 0x3FF
        level = (header >> 10) & 0x3FF
        size = (header >> 20) & 0xFFF
        if size == 0xFFF:
            if offset + 4 > length:
                break
            size = struct.unpack_from("<I", data, offset)[0]
            offset += 4
        payload = data[offset : offset + size]
        offset += size
        yield tag_id, level, payload


def _decode_stream(raw: bytes, compressed: bool) -> bytes:
    if not compressed:
        return raw
    try:
        return zlib.decompress(raw, -15)
    except zlib.error:
        return zlib.decompress(raw)


def _decode_bin_data_stream(raw: bytes, compressed: bool) -> bytes:
    if not compressed:
        return raw
    try:
        return _decode_stream(raw, compressed=True)
    except zlib.error:
        # Some HWP5 files mix deflated BinData streams with raw image bytes.
        return raw


def _read_flags(ole: object) -> int:
    data = ole.openstream("FileHeader").read()
    if len(data) < 40:
        return 0
    return struct.unpack_from("<I", data, 36)[0]


def _section_streams(ole: object) -> list[str]:
    streams = [
        "/".join(entry)
        for entry in ole.listdir(streams=True)
        if len(entry) == 2
        and entry[0] == "BodyText"
        and entry[1].startswith("Section")
    ]
    return sorted(
        streams,
        key=lambda stream: int(re.search(r"\d+", stream.split("/")[1]).group()),
    )


def _parse_doc_info_bin_data(ole: object, compressed: bool) -> dict[int, BinEntry]:
    if not ole.exists("DocInfo"):
        return {}
    raw = ole.openstream("DocInfo").read()
    data = _decode_stream(raw, compressed)

    entries: dict[int, BinEntry] = {}
    sequence = 1
    for tag_id, _level, payload in _iter_records(data):
        if tag_id != _TAG_BIN_DATA:
            continue
        if len(payload) < 4:
            sequence += 1
            continue
        attr = struct.unpack_from("<H", payload, 0)[0]
        data_type = attr & 0x0F
        if data_type in (1, 2):
            storage_id = struct.unpack_from("<H", payload, 2)[0]
            ext, _ = _read_hwp_string(payload, 4)
            entries[sequence] = BinEntry(storage_id=storage_id, ext=ext.lower())
        sequence += 1
    return entries


def _load_bin_data(ole: object, compressed: bool) -> dict[int, tuple[bytes, str]]:
    result: dict[int, tuple[bytes, str]] = {}
    for entry in ole.listdir(streams=True):
        if entry[0] != "BinData":
            continue
        name = entry[1]
        match = re.match(r"BIN([0-9A-Fa-f]{4})\.(\w+)$", name, re.IGNORECASE)
        if not match:
            continue
        stream_id = int(match.group(1), 16)
        ext = match.group(2).lower()
        try:
            raw = ole.openstream(f"BinData/{name}").read()
            result[stream_id] = (_decode_bin_data_stream(raw, compressed), ext)
        except Exception:
            continue
    return result


def _read_hwp_string(data: bytes, offset: int) -> tuple[str, int]:
    if offset + 2 > len(data):
        return "", offset
    length = struct.unpack_from("<H", data, offset)[0]
    offset += 2
    byte_length = length * 2
    if offset + byte_length > len(data):
        return "", offset + byte_length
    text = data[offset : offset + byte_length].decode("utf-16-le", errors="replace")
    return text, offset + byte_length


def _detect_mime(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] in (b"GIF8", b"GIF9"):
        return "image/gif"
    if data[:2] == b"BM":
        return "image/bmp"
    return "application/octet-stream"


def _mime_to_ext(mime: str) -> str:
    return {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/gif": "gif",
        "image/bmp": "bmp",
    }.get(mime, "bin")
