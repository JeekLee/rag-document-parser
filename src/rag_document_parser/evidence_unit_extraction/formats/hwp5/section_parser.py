from __future__ import annotations

import struct
from dataclasses import dataclass, field

from ....models import PendingAsset
from .binary_reader import BinEntry as _BinEntry
from .binary_reader import Hwp5BinaryReader
from .blocks import DrawingLineBlock as _DrawingLineBlock
from .blocks import ImageBlock as _ImageBlock
from .blocks import TableBlock as _TableBlock
from .blocks import TextBlock as _TextBlock
from .diagram import Hwp5DiagramBuilder
from .parsed import ParsedBlocks as _ParsedBlocks
from .table_extraction import Cell as _Cell
from .table_extraction import Hwp5TableBuilder
from .text import clean_text as _clean_text

_TAG_PARA_TEXT = 0x43
_TAG_CTRL_HEADER = 0x47
_TAG_LIST_HEADER = 0x48
_TAG_SHAPE_COMPONENT = 0x4C
_TAG_TABLE_BODY = 0x4D
_TAG_SHAPE_COMPONENT_LINE = 0x4E
_TAG_SHAPE_PICTURE = 0x55

_CTRL_TABLE = b" lbt"
_CTRL_GSO = b" osg"
_PICTURE_BIN_DATA_ID_OFFSET = 71
_SHAPE_CTRL_TYPES = {
    b"cer$": "rectangle",
    b"lle$": "ellipse",
    b"txt$": "textbox",
    b"prg$": "group",
    b"noc$": "container",
    b"lop$": "polygon",
    b"cip$": "picture",
    b"nil$": "line",
    b"loc$": "connector",
    b"cra$": "arc",
    b"ruc$": "curve",
}
_HWP5_BINARY_READER = Hwp5BinaryReader()
_HWP5_DIAGRAM_BUILDER = Hwp5DiagramBuilder()
_HWP5_TABLE_BUILDER = Hwp5TableBuilder()
_bbox_int = _HWP5_DIAGRAM_BUILDER.bbox_int
_ctrl_id_text = _HWP5_DIAGRAM_BUILDER.ctrl_id_text
_detect_mime = _HWP5_BINARY_READER.detect_mime
_iter_records = _HWP5_BINARY_READER.iter_records
_mime_to_ext = _HWP5_BINARY_READER.mime_to_ext
_diagram_node_from_text_block = _HWP5_DIAGRAM_BUILDER.node_from_text_block
_single_cell_table_text = _HWP5_TABLE_BUILDER.single_cell_text
_structured_connector = _HWP5_DIAGRAM_BUILDER.connector
_structured_diagram = _HWP5_DIAGRAM_BUILDER.build
_structured_table = _HWP5_TABLE_BUILDER.build
_table_has_content = _HWP5_TABLE_BUILDER.has_content
_unsupported_drawing_shape_warning = _HWP5_DIAGRAM_BUILDER.unsupported_shape_warning


@dataclass(frozen=True)
class Hwp5SectionParser:
    """Parses decoded HWP5 section records into typed intermediate blocks."""

    def parse(
        self,
        data: bytes,
        bin_entries: dict[int, _BinEntry] | None = None,
        bin_streams: dict[int, tuple[bytes, str]] | None = None,
        asset_offset: int = 0,
    ) -> _ParsedBlocks:
        return _parse_section(
            data,
            bin_entries=bin_entries,
            bin_streams=bin_streams,
            asset_offset=asset_offset,
        )


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
    rowspan: int = 1
    colspan: int = 1
    row_count: int | None = None
    column_count: int | None = None

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
    gso_bbox: dict[str, int | str] | None = None
    gso_shape_type: str | None = None
    gso_line_payload: bytes | None = None
    gso_instance_id: int | None = None
    gso_ctrl_id: str | None = None
    gso_metadata: dict[str, object] = {}
    gso_container_stack: list[tuple[int, int | None, str | None]] = []

    def gso_parent_metadata(level: int) -> dict[str, object]:
        for container_level, instance_id, ctrl_id in reversed(gso_container_stack):
            if container_level >= level:
                continue
            metadata: dict[str, object] = {}
            if instance_id is not None:
                metadata["parent_instance_id"] = instance_id
            if ctrl_id is not None:
                metadata["parent_ctrl_id"] = ctrl_id
            return metadata
        return {}

    def append_gso_text_block(text_block: _TextBlock) -> None:
        if table_stack and table_stack[-1].in_cell:
            table_stack[-1].current_cell_children.append(
                {
                    "type": "diagram",
                    "format": "structured_diagram",
                    "content": _structured_diagram(
                        text_block.text,
                        nodes=[_diagram_node_from_text_block(1, text_block)],
                        connectors=[],
                    ),
                }
            )
        else:
            parsed.blocks.append(text_block)
        parsed.saw_drawing = True

    def preserve_active_container(next_level: int) -> None:
        nonlocal gso_level, gso_text_parts, gso_bbox, gso_shape_type
        nonlocal gso_line_payload, gso_instance_id, gso_ctrl_id
        nonlocal gso_metadata
        container_level = gso_level
        container_instance_id = gso_instance_id
        container_ctrl_id = gso_ctrl_id
        text = _clean_text(" ".join(gso_text_parts))
        metadata = {**gso_metadata, "container": True}
        append_gso_text_block(
            _TextBlock(
                text,
                origin="drawing",
                bbox=gso_bbox,
                shape_type=gso_shape_type or "container",
                instance_id=gso_instance_id,
                ctrl_id=gso_ctrl_id,
                metadata=metadata,
            )
        )
        gso_container_stack.append(
            (container_level, container_instance_id, container_ctrl_id)
        )
        gso_level = next_level
        gso_text_parts = []
        gso_bbox = None
        gso_shape_type = None
        gso_line_payload = None
        gso_instance_id = None
        gso_ctrl_id = None
        gso_metadata = gso_parent_metadata(next_level)

    def close_gso() -> None:
        nonlocal in_gso, gso_level, gso_text_parts, gso_bbox
        nonlocal gso_shape_type, gso_line_payload, gso_instance_id, gso_ctrl_id
        nonlocal gso_metadata
        text = _clean_text(" ".join(gso_text_parts))
        if gso_shape_type in {"line", "connector"}:
            line_block = _DrawingLineBlock(
                bbox=gso_bbox,
                points=_line_points_from_bbox(gso_bbox, gso_line_payload),
                arrow=_line_has_arrow(gso_line_payload),
                ctrl_id=gso_ctrl_id,
                instance_id=gso_instance_id,
                payload=gso_line_payload,
                metadata=dict(gso_metadata),
            )
            if table_stack and table_stack[-1].in_cell:
                table_stack[-1].current_cell_children.append(
                    {
                        "type": "diagram",
                        "format": "structured_diagram",
                        "content": _structured_diagram(
                            "",
                            nodes=[],
                            connectors=[_structured_connector(1, line_block)],
                        ),
                    }
                )
            else:
                parsed.blocks.append(line_block)
            parsed.saw_drawing = True
        elif (
            text
            or gso_bbox is not None
            or (gso_shape_type is not None and gso_shape_type != "picture")
        ):
            shape_type = gso_shape_type or ("textbox" if text else "shape")
            text_block = _TextBlock(
                text,
                origin="drawing",
                bbox=gso_bbox,
                shape_type=shape_type,
                instance_id=gso_instance_id,
                ctrl_id=gso_ctrl_id,
                metadata=dict(gso_metadata),
            )
            append_gso_text_block(text_block)
        in_gso = False
        gso_level = -1
        gso_text_parts = []
        gso_bbox = None
        gso_shape_type = None
        gso_line_payload = None
        gso_instance_id = None
        gso_ctrl_id = None
        gso_metadata = {}

    def close_current_cell(ctx: _TableCtx) -> None:
        if not ctx.in_cell:
            return
        ctx.current_row.append(
            _Cell(
                text=_clean_text("\n".join(ctx.current_cell_parts)),
                children=list(ctx.current_cell_children),
                row_addr=ctx.row_addr,
                col_addr=ctx.col_addr,
                rowspan=ctx.rowspan,
                colspan=ctx.colspan,
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
            single_text = _single_cell_table_text(
                ctx.rows,
                row_count=ctx.row_count,
                column_count=ctx.column_count,
            )
            if single_text is not None:
                parsed.blocks.append(_TextBlock(single_text))
            elif _table_has_content(ctx.rows):
                parsed.blocks.append(
                    _TableBlock(
                        ctx.rows,
                        row_count=ctx.row_count,
                        column_count=ctx.column_count,
                    )
                )
            return

        if _table_has_content(ctx.rows):
            table_stack[-1].current_cell_children.append(
                {
                    "type": "table",
                    "format": "structured_table",
                    "content": _structured_table(
                        ctx.rows,
                        row_count=ctx.row_count,
                        column_count=ctx.column_count,
                    ),
                }
            )

    for tag_id, level, payload in _iter_records(data):
        if in_gso and level <= gso_level and tag_id != _TAG_CTRL_HEADER:
            close_gso()
        while gso_container_stack and level <= gso_container_stack[-1][0]:
            gso_container_stack.pop()

        while table_stack and level <= table_stack[-1].ctrl_level:
            close_top_table()

        if tag_id == _TAG_CTRL_HEADER:
            ctrl = payload[:4] if len(payload) >= 4 else b""
            if ctrl == _CTRL_GSO:
                if in_gso:
                    close_gso()
                in_gso = True
                gso_level = level
                gso_text_parts = []
                gso_bbox = _gso_bbox_from_ctrl_header(payload)
                gso_shape_type = None
                gso_line_payload = None
                gso_instance_id = _gso_instance_id(payload)
                gso_ctrl_id = _ctrl_id_text(ctrl)
                gso_metadata = gso_parent_metadata(level)
            elif ctrl in _SHAPE_CTRL_TYPES:
                shape_type = _SHAPE_CTRL_TYPES[ctrl]
                if in_gso and level <= gso_level:
                    close_gso()
                elif (
                    in_gso
                    and gso_shape_type in {"group", "container"}
                    and level > gso_level
                ):
                    preserve_active_container(level)
                if not in_gso:
                    in_gso = True
                    gso_level = level
                    gso_text_parts = []
                    gso_line_payload = None
                    gso_metadata = gso_parent_metadata(level)
                gso_bbox = _shape_bbox_from_ctrl_header(payload) or gso_bbox
                gso_shape_type = shape_type
                gso_instance_id = _gso_instance_id(payload)
                gso_ctrl_id = _ctrl_id_text(ctrl)
            elif ctrl == _CTRL_TABLE:
                table_stack.append(_TableCtx(ctrl_level=level))
            continue

        if (
            table_stack
            and tag_id == _TAG_TABLE_BODY
            and level == table_stack[-1].ctrl_level + 1
        ):
            top = table_stack[-1]
            if len(payload) >= 8:
                top.row_count = max(0, struct.unpack_from("<H", payload, 4)[0])
                top.column_count = max(0, struct.unpack_from("<H", payload, 6)[0])
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
                parsed.blocks.append(
                    _ImageBlock(
                        str(image["content"]["asset_id"]),
                        metadata=dict(image.get("metadata", {})),
                    )
                )
            in_gso = False
            gso_level = -1
            gso_text_parts = []
            gso_bbox = None
            gso_shape_type = None
            gso_line_payload = None
            gso_instance_id = None
            gso_ctrl_id = None
            continue

        if in_gso and tag_id == _TAG_SHAPE_COMPONENT:
            ctrl = payload[:4] if len(payload) >= 4 else b""
            if ctrl in _SHAPE_CTRL_TYPES:
                gso_shape_type = _SHAPE_CTRL_TYPES[ctrl]
                gso_ctrl_id = _ctrl_id_text(ctrl)
            elif ctrl:
                parsed.quality_warnings.append(
                    _unsupported_drawing_shape_warning(ctrl, tag_id, level)
                )
            continue

        if in_gso and tag_id == _TAG_SHAPE_COMPONENT_LINE:
            gso_line_payload = payload
            continue

        if (
            table_stack
            and tag_id == _TAG_LIST_HEADER
            and level == table_stack[-1].ctrl_level + 1
        ):
            top = table_stack[-1]
            col_addr = struct.unpack_from("<H", payload, 8)[0] if len(payload) >= 10 else 0
            row_addr = struct.unpack_from("<H", payload, 10)[0] if len(payload) >= 12 else 0
            raw_colspan = (
                struct.unpack_from("<H", payload, 12)[0] if len(payload) >= 14 else 1
            )
            raw_rowspan = (
                struct.unpack_from("<H", payload, 14)[0] if len(payload) >= 16 else 1
            )
            if not _is_valid_table_cell_header(
                top,
                row_addr=row_addr,
                col_addr=col_addr,
                rowspan=raw_rowspan,
                colspan=raw_colspan,
            ):
                continue
            if top.in_cell:
                close_current_cell(top)
                if row_addr != top.row_addr:
                    top.rows.append(top.current_row)
                    top.current_row = []
            top.row_addr = row_addr
            top.col_addr = col_addr
            top.rowspan = raw_rowspan
            top.colspan = raw_colspan
            top.in_cell = True
            continue

        if tag_id != _TAG_PARA_TEXT:
            continue

        text = _para_text_from_payload(payload)
        if not text:
            continue
        if in_gso:
            gso_text_parts.append(text)
        elif table_stack and table_stack[-1].in_cell:
            table_stack[-1].current_cell_parts.append(text)
        else:
            parsed.blocks.append(_TextBlock(text))

    while table_stack:
        close_top_table()
    if in_gso:
        close_gso()

    return parsed


def _is_valid_table_cell_header(
    ctx: _TableCtx,
    *,
    row_addr: int,
    col_addr: int,
    rowspan: int,
    colspan: int,
) -> bool:
    if row_addr < 0 or col_addr < 0 or rowspan < 1 or colspan < 1:
        return False
    if ctx.row_count is not None and ctx.row_count > 0:
        if row_addr >= ctx.row_count or row_addr + rowspan > ctx.row_count:
            return False
    if ctx.column_count is not None and ctx.column_count > 0:
        if col_addr >= ctx.column_count or col_addr + colspan > ctx.column_count:
            return False
    return True


def _gso_bbox_from_ctrl_header(payload: bytes) -> dict[str, int | str] | None:
    if len(payload) < 24:
        return None
    x = struct.unpack_from("<I", payload, 8)[0]
    y = struct.unpack_from("<I", payload, 12)[0]
    width = struct.unpack_from("<I", payload, 16)[0]
    height = struct.unpack_from("<I", payload, 20)[0]
    if width == 0 and height == 0:
        return None
    return {
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "unit": "hwp",
    }


def _shape_bbox_from_ctrl_header(payload: bytes) -> dict[str, int | str] | None:
    if len(payload) < 20:
        return None
    y = struct.unpack_from("<I", payload, 4)[0]
    x = struct.unpack_from("<I", payload, 8)[0]
    width = struct.unpack_from("<I", payload, 12)[0]
    height = struct.unpack_from("<I", payload, 16)[0]
    if width == 0 and height == 0:
        return _gso_bbox_from_ctrl_header(payload)
    return {
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "unit": "hwp",
    }


def _gso_instance_id(payload: bytes) -> int | None:
    if len(payload) < 36:
        return None
    instance_id = struct.unpack_from("<I", payload, 32)[0]
    return instance_id or None


def _line_points_from_bbox(
    bbox: dict[str, int | str] | None,
    payload: bytes | None,
) -> list[dict[str, int]]:
    if bbox is None:
        if payload is not None and len(payload) >= 16:
            start_x, start_y, end_x, end_y = struct.unpack_from("<4i", payload)
            return [{"x": start_x, "y": start_y}, {"x": end_x, "y": end_y}]
        return []
    x = _bbox_int(bbox, "x")
    y = _bbox_int(bbox, "y")
    width = _bbox_int(bbox, "width")
    height = _bbox_int(bbox, "height")
    if width >= max(height, 1) * 3:
        y_mid = y + max(height, 1) // 2
        return [{"x": x, "y": y_mid}, {"x": x + width, "y": y_mid}]
    if height >= max(width, 1) * 3:
        x_mid = x + max(width, 1) // 2
        return [{"x": x_mid, "y": y}, {"x": x_mid, "y": y + height}]
    if payload is not None and len(payload) >= 16:
        start_x, start_y, end_x, end_y = struct.unpack_from("<4i", payload)
        if max(abs(start_x), abs(start_y), abs(end_x), abs(end_y)) >= 1000:
            return [
                {"x": x + start_x, "y": y + start_y},
                {"x": x + end_x, "y": y + end_y},
            ]
    return [{"x": x, "y": y}, {"x": x + width, "y": y + height}]


def _line_has_arrow(payload: bytes | None) -> bool:
    if payload is None or len(payload) < 20:
        return False
    link_type = struct.unpack_from("<I", payload, 16)[0]
    return link_type % 3 != 0


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
    stream_id = entry.storage_id if entry is not None else bin_data_id
    stream_data = bin_streams.get(stream_id)
    if stream_data is None:
        stream_id = bin_data_id
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
            metadata={
                "source": "hwp5_bindata",
                "bin_data_id": bin_data_id,
                "storage_id": entry.storage_id if entry is not None else None,
                "stream_id": stream_id,
                "doc_info_ext": entry.ext if entry is not None else None,
                "stream_ext": ext_from_name,
            },
        )
    )
    return {
        "type": "image",
        "format": "asset_ref",
        "content": {"asset_id": asset_id, "caption": None},
        "metadata": {
            "asset_id": asset_id,
            "source": "hwp5_bindata",
            "bin_data_id": bin_data_id,
            "storage_id": entry.storage_id if entry is not None else None,
            "stream_id": stream_id,
            "ext": ext,
        },
    }


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
