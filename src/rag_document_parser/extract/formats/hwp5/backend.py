from __future__ import annotations

import io
import re
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any

from ....models import EvidenceUnit, PendingAsset, SourceEvidence
from ...backend import ParsedDocument
from ...table_source import (
    build_column_source_labels as _build_column_source_labels,
    common_semantic_header_prefix as _common_semantic_header_prefix,
    is_semantic_column_label as _is_semantic_column_label,
)

_TAG_BIN_DATA = 0x12
_TAG_PARA_TEXT = 0x43
_TAG_CTRL_HEADER = 0x47
_TAG_LIST_HEADER = 0x48
_TAG_SHAPE_PICTURE = 0x55

_CTRL_TABLE = b" lbt"
_CTRL_GSO = b" osg"
_PICTURE_BIN_DATA_ID_OFFSET = 71


@dataclass(frozen=True)
class Hwp5Backend:
    supported_suffixes = (".hwp",)

    def parse(self, data: bytes, suffix: str) -> ParsedDocument:
        try:
            import olefile
        except (ImportError, ModuleNotFoundError) as exc:
            raise NotImplementedError(
                "HWP5 extraction requires the optional 'olefile' dependency."
            ) from exc

        ole = olefile.OleFileIO(io.BytesIO(data))
        try:
            flags = _read_flags(ole)
            compressed = bool(flags & 0x1)
            bin_entries = _parse_doc_info_bin_data(ole, compressed)
            bin_streams = _load_bin_data(ole, compressed)
            parsed = _ParsedBlocks()
            for stream_name in _section_streams(ole):
                raw = ole.openstream(stream_name).read()
                section = _parse_section(
                    _decode_stream(raw, compressed),
                    bin_entries=bin_entries,
                    bin_streams=bin_streams,
                    asset_offset=len(parsed.assets),
                )
                parsed.extend(section)
        finally:
            ole.close()

        return parsed.to_document()


@dataclass(frozen=True)
class _BinEntry:
    storage_id: int
    ext: str


@dataclass
class _Cell:
    text: str = ""
    children: list[dict[str, object]] = field(default_factory=list)
    col_addr: int | None = None


@dataclass
class _TextBlock:
    text: str


@dataclass
class _TableBlock:
    rows: list[list[_Cell]]


@dataclass
class _ImageBlock:
    asset_id: str


@dataclass
class _ParsedBlocks:
    blocks: list[_TextBlock | _TableBlock | _ImageBlock] = field(default_factory=list)
    assets: list[PendingAsset] = field(default_factory=list)
    saw_drawing: bool = False
    missing_image_count: int = 0

    def extend(self, other: _ParsedBlocks) -> None:
        self.blocks.extend(other.blocks)
        self.assets.extend(other.assets)
        self.saw_drawing = self.saw_drawing or other.saw_drawing
        self.missing_image_count += other.missing_image_count

    def to_document(self) -> ParsedDocument:
        return _to_document(self)


@dataclass
class _TableCtx:
    ctrl_level: int
    rows: list[list[_Cell]] = field(default_factory=list)
    current_row: list[_Cell] = field(default_factory=list)
    current_cell_parts: list[str] = field(default_factory=list)
    current_cell_children: list[dict[str, object]] = field(default_factory=list)
    in_cell: bool = False
    row_addr: int = -1
    col_addr: int = -1


def _to_document(parsed: _ParsedBlocks) -> ParsedDocument:
    units: list[EvidenceUnit] = []
    block_index = 1
    table_index = 1

    for block in parsed.blocks:
        if isinstance(block, _TextBlock):
            text = _clean_text(block.text)
            if not text:
                continue
            units.append(
                EvidenceUnit(
                    id=f"b{block_index}",
                    type="text",
                    format="plain",
                    source=SourceEvidence(kind="text", text=text),
                    content=text,
                    metadata={
                        "common": {
                            "chunk_kind": "text",
                            "section_path": [],
                            "display_format": "plain",
                        }
                    },
                )
            )
            block_index += 1
            continue

        if isinstance(block, _ImageBlock):
            units.append(
                EvidenceUnit(
                    id=f"b{block_index}",
                    type="image",
                    format="asset_ref",
                    source=SourceEvidence(kind="image", text=f"image: {block.asset_id}"),
                    content={"asset_id": block.asset_id, "caption": None},
                    metadata={
                        "common": {
                            "chunk_kind": "image",
                            "section_path": [],
                            "display_format": "image",
                        },
                        "asset": {"asset_id": block.asset_id},
                    },
                )
            )
            block_index += 1
            continue

        structured = _structured_table(block.rows)
        if not _table_has_content(block.rows):
            continue
        table_id = f"t{table_index}"
        units.append(
            EvidenceUnit(
                id=f"b{block_index}",
                type="table",
                format="structured_table",
                source=SourceEvidence(kind="table", text=_table_source_text(structured)),
                content=structured,
                metadata={
                    "common": {
                        "chunk_kind": "table",
                        "section_path": [],
                        "display_format": "structured_table",
                    },
                    "table": {
                        "table_id": table_id,
                        "headers": [
                            str(column["text"]) for column in structured["columns"]
                        ],
                        "row_count": len(structured["rows"]),
                    },
                },
            )
        )
        block_index += 1
        table_index += 1

    warnings: list[dict[str, Any]] = []
    if parsed.saw_drawing:
        warnings.append(
            {
                "type": "hwp5_drawing_structure_unsupported",
                "severity": "medium",
                "message": (
                    "HWP5 drawing object text was extracted, but geometry and "
                    "connector structure are not represented."
                ),
            }
        )
    if parsed.missing_image_count:
        warnings.append(
            {
                "type": "hwp5_images_missing",
                "severity": "medium",
                "message": (
                    f"{parsed.missing_image_count} HWP5 image reference(s) could not "
                    "be resolved from BinData streams."
                ),
            }
        )

    return ParsedDocument(
        units=units,
        assets=parsed.assets,
        quality_warnings=warnings,
    )


def _parse_section(
    data: bytes,
    bin_entries: dict[int, _BinEntry] | None = None,
    bin_streams: dict[int, tuple[bytes, str]] | None = None,
    asset_offset: int = 0,
) -> _ParsedBlocks:
    parsed = _ParsedBlocks()
    table_stack: list[_TableCtx] = []
    bin_entries = bin_entries or {}
    bin_streams = bin_streams or {}

    in_gso = False
    gso_level = -1
    gso_text_parts: list[str] = []

    def close_gso() -> None:
        nonlocal in_gso, gso_level, gso_text_parts
        text = _clean_text(" ".join(gso_text_parts))
        if text:
            parsed.blocks.append(_TextBlock(text))
            parsed.saw_drawing = True
        in_gso = False
        gso_level = -1
        gso_text_parts = []

    def close_current_cell(ctx: _TableCtx) -> None:
        if not ctx.in_cell:
            return
        ctx.current_row.append(
            _Cell(
                text=_clean_text(" ".join(ctx.current_cell_parts)),
                children=list(ctx.current_cell_children),
                col_addr=ctx.col_addr,
            )
        )
        ctx.current_cell_parts = []
        ctx.current_cell_children = []

    def close_top_table() -> None:
        ctx = table_stack.pop()
        close_current_cell(ctx)
        if ctx.current_row:
            ctx.rows.append(ctx.current_row)

        if not table_stack:
            single_text = _single_cell_table_text(ctx.rows)
            if single_text is not None:
                parsed.blocks.append(_TextBlock(single_text))
            elif _table_has_content(ctx.rows):
                parsed.blocks.append(_TableBlock(ctx.rows))
            return

        if _table_has_content(ctx.rows):
            table_stack[-1].current_cell_children.append(
                {
                    "type": "table",
                    "format": "structured_table",
                    "content": _structured_table(ctx.rows),
                }
            )

    for tag_id, level, payload in _iter_records(data):
        while table_stack and level <= table_stack[-1].ctrl_level:
            close_top_table()

        if in_gso and level <= gso_level and tag_id != _TAG_CTRL_HEADER:
            close_gso()

        if tag_id == _TAG_CTRL_HEADER:
            ctrl = payload[:4] if len(payload) >= 4 else b""
            if ctrl == _CTRL_GSO:
                if in_gso:
                    close_gso()
                in_gso = True
                gso_level = level
                gso_text_parts = []
            elif ctrl == _CTRL_TABLE:
                table_stack.append(_TableCtx(ctrl_level=level))
            continue

        if in_gso and tag_id == _TAG_SHAPE_PICTURE:
            image = _image_child_from_picture(
                payload,
                parsed,
                bin_entries,
                bin_streams,
                asset_offset,
            )
            if image is None:
                parsed.missing_image_count += 1
            elif table_stack and table_stack[-1].in_cell:
                table_stack[-1].current_cell_children.append(image)
            else:
                parsed.blocks.append(_ImageBlock(str(image["content"]["asset_id"])))
            in_gso = False
            gso_level = -1
            gso_text_parts = []
            continue

        if (
            table_stack
            and tag_id == _TAG_LIST_HEADER
            and level == table_stack[-1].ctrl_level + 1
        ):
            top = table_stack[-1]
            col_addr = struct.unpack_from("<H", payload, 8)[0] if len(payload) >= 10 else 0
            row_addr = struct.unpack_from("<H", payload, 10)[0] if len(payload) >= 12 else 0
            if top.in_cell:
                close_current_cell(top)
                if row_addr != top.row_addr:
                    top.rows.append(top.current_row)
                    top.current_row = []
            top.row_addr = row_addr
            top.col_addr = col_addr
            top.in_cell = True
            continue

        if tag_id != _TAG_PARA_TEXT:
            continue

        text = _para_text_from_payload(payload)
        if not text:
            continue
        if table_stack and table_stack[-1].in_cell:
            table_stack[-1].current_cell_parts.append(text)
        elif in_gso:
            gso_text_parts.append(text)
        else:
            parsed.blocks.append(_TextBlock(text))

    while table_stack:
        close_top_table()
    if in_gso:
        close_gso()

    return parsed


def _image_child_from_picture(
    payload: bytes,
    parsed: _ParsedBlocks,
    bin_entries: dict[int, _BinEntry],
    bin_streams: dict[int, tuple[bytes, str]],
    asset_offset: int,
) -> dict[str, object] | None:
    if len(payload) < _PICTURE_BIN_DATA_ID_OFFSET + 2:
        return None
    bin_data_id = struct.unpack_from("<H", payload, _PICTURE_BIN_DATA_ID_OFFSET)[0]
    entry = bin_entries.get(bin_data_id)
    stream_data = bin_streams.get(entry.storage_id) if entry is not None else None
    if stream_data is None:
        stream_data = bin_streams.get(bin_data_id)
    if stream_data is None:
        return None

    raw_data, ext_from_name = stream_data
    mime = _detect_mime(raw_data)
    ext = _mime_to_ext(mime) if mime != "application/octet-stream" else ext_from_name
    asset_id = f"img-{asset_offset + len(parsed.assets) + 1:04d}"
    parsed.assets.append(
        PendingAsset(
            id=asset_id,
            kind="image",
            data=raw_data,
            mime=mime,
            ext=ext,
        )
    )
    return {
        "type": "image",
        "format": "asset_ref",
        "content": {"asset_id": asset_id, "caption": None},
    }


def _structured_table(rows: list[list[_Cell]]) -> dict[str, object]:
    normalized = _normalize_rows(rows)
    if not normalized:
        return {"caption": None, "columns": [], "rows": []}

    column_count = max(len(row) for row in normalized)
    header = _pad_row(normalized[0], column_count)
    columns = [
        {"id": f"c{index}", "text": header[index - 1].text}
        for index in range(1, column_count + 1)
    ]
    header_rows = [{"index": 1, "cells": _evidence_cells(header, columns)}]

    data_rows: list[dict[str, object]] = []
    for row in normalized[1:]:
        padded = _pad_row(row, column_count)
        if not any(cell.text.strip() or cell.children for cell in padded):
            continue
        data_rows.append(
            {
                "index": len(data_rows) + 1,
                "cells": _evidence_cells(padded, columns),
            }
        )

    return {
        "caption": None,
        "columns": columns,
        "rows": data_rows,
        "header_rows": header_rows,
    }


def _evidence_cells(
    row: list[_Cell],
    columns: list[dict[str, str]],
) -> list[dict[str, object]]:
    return [
        {
            "column_id": column["id"],
            "text": row[index].text,
            "rowspan": 1,
            "colspan": 1,
            "children": list(row[index].children),
        }
        for index, column in enumerate(columns)
    ]


def _normalize_rows(rows: list[list[_Cell]]) -> list[list[_Cell]]:
    sparse_rows: list[list[tuple[int, _Cell]]] = []
    column_count = 0
    for row in rows:
        sparse_row: list[tuple[int, _Cell]] = []
        fallback_col = 0
        for cell in row:
            col_addr = cell.col_addr if cell.col_addr is not None and cell.col_addr >= 0 else fallback_col
            cleaned = _Cell(
                text=_clean_text(cell.text),
                children=list(cell.children),
                col_addr=col_addr,
            )
            sparse_row.append((col_addr, cleaned))
            column_count = max(column_count, col_addr + 1)
            fallback_col = col_addr + 1
        if any(cell.text or cell.children for _, cell in sparse_row):
            sparse_rows.append(sparse_row)

    normalized: list[list[_Cell]] = []
    for sparse_row in sparse_rows:
        dense_row = [_Cell(col_addr=index) for index in range(column_count)]
        for col_addr, cell in sparse_row:
            if 0 <= col_addr < column_count:
                dense_row[col_addr] = cell
        normalized.append(dense_row)
    return normalized


def _pad_row(row: list[_Cell], column_count: int) -> list[_Cell]:
    return row + [_Cell() for _ in range(column_count - len(row))]


def _single_cell_table_text(rows: list[list[_Cell]]) -> str | None:
    normalized = _normalize_rows(rows)
    if len(normalized) != 1 or len(normalized[0]) != 1:
        return None
    cell = normalized[0][0]
    if cell.children:
        return None
    text = cell.text.strip()
    return text or None


def _table_has_content(rows: list[list[_Cell]]) -> bool:
    return any(cell.text.strip() or cell.children for row in rows for cell in row)


def _table_source_text(table: dict[str, object]) -> str:
    columns = table["columns"]
    rows = table["rows"]
    column_text = _build_column_source_labels(columns, _column_source_label, rows)
    lines: list[str] = []
    if columns:
        lines.append(f"table: {len(columns)} columns")
    for header_row in table.get("header_rows", []):
        cells = _table_source_cells(
            header_row["cells"],
            column_text,
            use_header_labels=False,
        )
        if cells:
            lines.append(f"header {header_row['index']}: " + "; ".join(cells))
    for row in rows:
        cells = _table_source_cells(
            row["cells"],
            column_text,
            use_header_labels=True,
        )
        if cells:
            lines.append(f"row {row['index']}: " + "; ".join(cells))
    return "\n".join(lines)


def _table_source_cells(
    cells: list[dict[str, object]],
    column_text: dict[str, str],
    *,
    use_header_labels: bool,
) -> list[str]:
    result: list[str] = []
    for cell in cells:
        header = _cell_source_label(
            cell,
            column_text,
            use_header_labels=use_header_labels,
        )
        value = str(cell["text"])
        child_texts = [
            "nested table: " + _inline_table_source(child["content"])
            for child in cell["children"]
            if child.get("type", child.get("kind")) == "table"
        ]
        image_texts = [
            f"image: {child['content']['asset_id']}"
            for child in cell["children"]
            if child.get("type", child.get("kind")) == "image"
        ]
        combined = "; ".join(
            part for part in [value, *child_texts, *image_texts] if part
        )
        if combined:
            result.append(f"{header}: {combined}")
    return result


def _cell_source_label(
    cell: dict[str, object],
    column_text: dict[str, str],
    *,
    use_header_labels: bool,
) -> str:
    column_id = str(cell["column_id"])
    colspan = int(cell.get("colspan", 1))
    if use_header_labels:
        labels = [
            column_text.get(f"c{column_index}", f"col {column_index}")
            for column_index in range(
                _column_id_number(column_id),
                _column_id_number(column_id) + colspan,
            )
        ]
        common_prefix = _common_semantic_header_prefix(labels)
        if common_prefix is not None:
            return common_prefix
        if len(set(labels)) == 1 and _is_semantic_column_label(labels[0]):
            return labels[0]
        if colspan == 1 and _is_semantic_column_label(labels[0]):
            return labels[0]
    return _cell_coordinate_label(column_id, colspan)


def _column_source_label(column: dict[str, object]) -> str:
    text = str(column["text"]).strip()
    return text or _column_coordinate_label(str(column["id"]))


def _cell_coordinate_label(column_id: str, colspan: int) -> str:
    start = _column_id_number(column_id)
    if colspan <= 1:
        return _column_coordinate_label(column_id)
    return f"cols {start}-{start + colspan - 1}"


def _column_coordinate_label(column_id: str) -> str:
    return f"col {_column_id_number(column_id)}"


def _column_id_number(column_id: str) -> int:
    try:
        return max(1, int(column_id.removeprefix("c")))
    except ValueError:
        return 1


def _inline_table_source(table: dict[str, object]) -> str:
    source = _table_source_text(table)
    return source.replace("\n", " / ")


def _para_text_from_payload(payload: bytes) -> str:
    if len(payload) % 2 != 0:
        payload = payload[:-1]
    try:
        chars = list(payload.decode("utf-16-le"))
    except UnicodeDecodeError:
        return ""

    result: list[str] = []
    index = 0
    while index < len(chars):
        char = chars[index]
        if "\x01" <= char <= "\x1f":
            index += 8
            continue
        result.append(char)
        index += 1
    return _clean_text("".join(result))


def _clean_text(text: str) -> str:
    text = "".join(char for char in text if char == "\n" or char > "\x1f")
    return re.sub(r"[ \t\r\f\v]+", " ", text).strip()


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


def _parse_doc_info_bin_data(ole: object, compressed: bool) -> dict[int, _BinEntry]:
    if not ole.exists("DocInfo"):
        return {}
    raw = ole.openstream("DocInfo").read()
    data = _decode_stream(raw, compressed)

    entries: dict[int, _BinEntry] = {}
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
            entries[sequence] = _BinEntry(storage_id=storage_id, ext=ext.lower())
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
            result[stream_id] = (_decode_stream(raw, compressed), ext)
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
