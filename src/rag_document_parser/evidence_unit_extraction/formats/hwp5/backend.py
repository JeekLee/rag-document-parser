from __future__ import annotations

import io
import re
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable

from ....models import EvidenceUnit, PendingAsset, SourceEvidence
from ...backend import ParsedDocument
from ...schema import (
    structured_diagram as _structured_diagram_content,
    structured_table as _structured_table_content,
)
from ...table_source import (
    build_column_source_labels as _build_column_source_labels,
    common_semantic_header_prefix as _common_semantic_header_prefix,
    is_semantic_column_label as _is_semantic_column_label,
)

_TAG_BIN_DATA = 0x12
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
_DIAGRAM_STEP_LABEL_RE = re.compile(
    r"^(?:[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]|\d+[.)])"
)
_SHAPE_CTRL_TYPES = {
    b"cer$": "rectangle",  # "$rec"
    b"lle$": "ellipse",  # "$ell"
    b"txt$": "textbox",  # "$txt"
    b"prg$": "group",  # "$grp"
    b"noc$": "container",  # "$con"
    b"lop$": "polygon",  # "$pol"
    b"nil$": "line",  # "$lin"
    b"loc$": "connector",  # "$col"
    b"cra$": "arc",  # "$arc"
    b"ruc$": "curve",  # "$cur"
}


@dataclass(frozen=True)
class Hwp5Backend:
    supported_suffixes = (".hwp",)
    ocr_fn: Callable[[bytes, int], str] | None = None

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

        return parsed.to_document(ocr_fn=self.ocr_fn)


@dataclass(frozen=True)
class _BinEntry:
    storage_id: int
    ext: str


@dataclass
class _Cell:
    text: str = ""
    children: list[dict[str, object]] = field(default_factory=list)
    row_addr: int | None = None
    col_addr: int | None = None
    rowspan: int = 1
    colspan: int = 1
    synthetic: bool = False


@dataclass
class _TextBlock:
    text: str
    origin: str = "body"
    bbox: dict[str, int | str] | None = None
    shape_type: str = "label"
    instance_id: int | None = None
    ctrl_id: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class _DiagramBlock:
    text: str
    nodes: list[dict[str, object]] = field(default_factory=list)
    bboxes: list[dict[str, int | str] | None] = field(default_factory=list)
    connectors: list[dict[str, object]] = field(default_factory=list)


@dataclass
class _DrawingLineBlock:
    bbox: dict[str, int | str] | None
    points: list[dict[str, int]]
    arrow: bool = False
    ctrl_id: str | None = None
    instance_id: int | None = None
    payload: bytes | None = None


@dataclass
class _TableBlock:
    rows: list[list[_Cell]]
    row_count: int | None = None
    column_count: int | None = None


@dataclass
class _ImageBlock:
    asset_id: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class _ParsedBlocks:
    blocks: list[
        _TextBlock | _DiagramBlock | _DrawingLineBlock | _TableBlock | _ImageBlock
    ] = field(default_factory=list)
    assets: list[PendingAsset] = field(default_factory=list)
    saw_drawing: bool = False
    missing_image_count: int = 0
    quality_warnings: list[dict[str, Any]] = field(default_factory=list)

    def extend(self, other: _ParsedBlocks) -> None:
        self.blocks.extend(other.blocks)
        self.assets.extend(other.assets)
        self.saw_drawing = self.saw_drawing or other.saw_drawing
        self.missing_image_count += other.missing_image_count
        self.quality_warnings.extend(other.quality_warnings)

    def to_document(
        self,
        ocr_fn: Callable[[bytes, int], str] | None = None,
    ) -> ParsedDocument:
        return _apply_ocr_fallback(_to_document(self), ocr_fn)


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


def _to_document(parsed: _ParsedBlocks) -> ParsedDocument:
    units: list[EvidenceUnit] = []
    warnings: list[dict[str, Any]] = list(parsed.quality_warnings)
    block_index = 1
    table_index = 1
    saw_structured_diagram = False

    for block in _coalesce_drawing_text_blocks(parsed.blocks):
        if isinstance(block, _TextBlock):
            text = _clean_text(block.text)
            if not text:
                continue
            chunk_kind = "drawing" if block.origin == "drawing" else "text"
            display_format = "drawing_text" if block.origin == "drawing" else "plain"
            units.append(
                EvidenceUnit(
                    id=f"b{block_index}",
                    type="text",
                    format="plain",
                    source=SourceEvidence(kind="text", text=text),
                    content=text,
                    metadata={
                        "common": {
                            "chunk_kind": chunk_kind,
                            "section_path": [],
                            "display_format": display_format,
                        }
                    },
                )
            )
            block_index += 1
            continue

        if isinstance(block, _DiagramBlock):
            structured = _structured_diagram(
                block.text,
                nodes=block.nodes,
                bboxes=block.bboxes,
                connectors=block.connectors,
            )
            source_text = _diagram_source_text(structured)
            if not source_text:
                continue
            saw_structured_diagram = True
            unresolved_connectors = _unresolved_connector_ids(structured)
            units.append(
                EvidenceUnit(
                    id=f"b{block_index}",
                    type="diagram",
                    format="structured_diagram",
                    source=SourceEvidence(kind="diagram", text=source_text),
                    content=structured,
                    metadata={
                        "common": {
                            "chunk_kind": "diagram",
                            "section_path": [],
                            "display_format": "structured_diagram",
                        },
                        "diagram": {
                            "node_count": len(structured["nodes"]),
                            "connector_count": len(structured["connectors"]),
                            "edge_count": len(structured["edges"]),
                        },
                    },
                )
            )
            if unresolved_connectors:
                warnings.append(
                    {
                        "type": "hwp5_diagram_connector_unresolved",
                        "severity": "medium",
                        "unit_id": f"b{block_index}",
                        "connector_ids": unresolved_connectors,
                        "message": (
                            "HWP5 drawing connector(s) were preserved, but could not "
                            "be mapped to diagram edges."
                        ),
                    }
                )
            block_index += 1
            continue

        if isinstance(block, _DrawingLineBlock):
            continue

        if isinstance(block, _ImageBlock):
            asset_metadata = {"asset_id": block.asset_id, **dict(block.metadata)}
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
                        "asset": asset_metadata,
                    },
                )
            )
            block_index += 1
            continue

        structured = _structured_table(
            block.rows,
            row_count=block.row_count,
            column_count=block.column_count,
        )
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

    if saw_structured_diagram:
        warnings.append(
            {
                "type": "hwp5_drawing_structure_partial",
                "severity": "medium",
                "message": (
                    "HWP5 drawing object text and some geometry were extracted as "
                    "structured diagram evidence, but unsupported or undocumented "
                    "drawing details may still be incomplete."
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


def _coalesce_drawing_text_blocks(
    blocks: list[
        _TextBlock | _DiagramBlock | _DrawingLineBlock | _TableBlock | _ImageBlock
    ],
) -> list[
    _TextBlock | _DiagramBlock | _DrawingLineBlock | _TableBlock | _ImageBlock
]:
    result: list[
        _TextBlock | _DiagramBlock | _DrawingLineBlock | _TableBlock | _ImageBlock
    ] = []
    index = 0
    while index < len(blocks):
        block = blocks[index]
        if not _is_drawing_text_block(block):
            result.append(block)
            index += 1
            continue

        cluster_end = index
        drawing_text_count = 1 if _drawing_block_has_text(block) else 0
        line_count = 0
        short_body_gap = 0
        scan = index + 1
        while scan < len(blocks):
            candidate = blocks[scan]
            if _is_drawing_text_block(candidate):
                if _drawing_block_has_text(candidate):
                    drawing_text_count += 1
                cluster_end = scan
                short_body_gap = 0
                scan += 1
                continue
            if isinstance(candidate, _DrawingLineBlock):
                line_count += 1
                cluster_end = scan
                short_body_gap = 0
                scan += 1
                continue
            if _is_short_body_text_block(candidate) and short_body_gap < 6:
                short_body_gap += 1
                scan += 1
                continue
            break

        if drawing_text_count < 2 and line_count == 0:
            if isinstance(block, _TextBlock):
                result.append(_TextBlock(block.text))
            index += 1
            continue

        prefix: list[_TextBlock] = []
        while (
            result
            and len(prefix) < 3
            and _is_short_body_text_block(result[-1])
        ):
            previous = result.pop()
            if isinstance(previous, _TextBlock):
                prefix.append(previous)
        prefix.reverse()

        text_blocks = [
            item
            for item in [*prefix, *blocks[index : cluster_end + 1]]
            if isinstance(item, _TextBlock)
        ]
        line_blocks = [
            item
            for item in blocks[index : cluster_end + 1]
            if isinstance(item, _DrawingLineBlock)
        ]
        result.append(
            _DiagramBlock(
                "\n".join(
                    text
                    for text in (_clean_text(item.text) for item in text_blocks)
                    if text
                ),
                nodes=[
                    _diagram_node_from_text_block(node_index, item)
                    for node_index, item in enumerate(text_blocks, start=1)
                    if _clean_text(item.text)
                    or item.bbox is not None
                    or item.shape_type not in {"label", ""}
                ],
                bboxes=[
                    item.bbox
                    for item in text_blocks
                    if _clean_text(item.text)
                ],
                connectors=[
                    _structured_connector(connector_index, item)
                    for connector_index, item in enumerate(line_blocks, start=1)
                ],
            )
        )
        index = cluster_end + 1

    return result


def _is_drawing_text_block(block: object) -> bool:
    return isinstance(block, _TextBlock) and block.origin == "drawing"


def _drawing_block_has_text(block: object) -> bool:
    return isinstance(block, _TextBlock) and bool(_clean_text(block.text))


def _is_short_body_text_block(block: object) -> bool:
    return (
        isinstance(block, _TextBlock)
        and block.origin == "body"
        and 0 < len(_clean_text(block.text)) <= 80
    )


def _structured_diagram(
    text: str,
    *,
    nodes: list[dict[str, object]] | None = None,
    bboxes: list[dict[str, int | str] | None] | None = None,
    connectors: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    diagram_nodes = (
        [_normalize_diagram_node(index, node) for index, node in enumerate(nodes, start=1)]
        if nodes is not None
        else _diagram_nodes_from_text(text, bboxes=bboxes)
    )
    connector_items = connectors or []
    return _structured_diagram_content(
        nodes=diagram_nodes,
        edges=_infer_connector_edges(diagram_nodes, connector_items),
        connectors=connector_items,
    )


def _diagram_nodes_from_text(
    text: str,
    *,
    bboxes: list[dict[str, int | str] | None] | None,
) -> list[dict[str, object]]:
    labels = [
        line
        for line in (_clean_text(part) for part in text.splitlines())
        if line
    ]
    return [
        {
            "id": f"n{index}",
            "shape_type": "label",
            "text": label,
            "bbox": bboxes[index - 1] if bboxes and index <= len(bboxes) else None,
            "metadata": {"source": "hwp5_drawing_text"},
        }
        for index, label in enumerate(labels, start=1)
    ]


def _diagram_node_from_text_block(
    index: int,
    block: _TextBlock,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "source": "hwp5_drawing_text" if block.origin == "drawing" else "hwp5_body_text"
    }
    metadata.update(block.metadata)
    if block.instance_id is not None:
        metadata["instance_id"] = block.instance_id
    if block.ctrl_id is not None:
        metadata["ctrl_id"] = block.ctrl_id
    return {
        "id": f"n{index}",
        "shape_type": block.shape_type or "label",
        "text": _clean_text(block.text),
        "bbox": block.bbox,
        "metadata": metadata,
    }


def _normalize_diagram_node(
    index: int,
    node: dict[str, object],
) -> dict[str, object]:
    normalized = dict(node)
    normalized.setdefault("id", f"n{index}")
    normalized.setdefault("shape_type", "label")
    normalized.setdefault("text", "")
    normalized.setdefault("bbox", None)
    normalized.setdefault("metadata", {})
    return normalized


def _structured_connector(
    index: int,
    line: _DrawingLineBlock,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "source": "hwp5_gso_line",
    }
    if line.ctrl_id is not None:
        metadata["ctrl_id"] = line.ctrl_id
    if line.instance_id is not None:
        metadata["instance_id"] = line.instance_id
    metadata.update(_line_payload_metadata(line.payload))
    return {
        "id": f"c{index}",
        "type": "arrow" if line.arrow else "line",
        "bbox": dict(line.bbox) if line.bbox is not None else None,
        "points": [dict(point) for point in line.points],
        "arrow": line.arrow,
        "metadata": metadata,
    }


def _line_payload_metadata(payload: bytes | None) -> dict[str, object]:
    if payload is None:
        return {}
    metadata: dict[str, object] = {"payload_bytes": len(payload)}
    if len(payload) >= 20:
        metadata["link_type"] = struct.unpack_from("<I", payload, 16)[0]
    if len(payload) >= 36:
        metadata["start_subject_id"] = struct.unpack_from("<I", payload, 20)[0]
        metadata["start_subject_index"] = struct.unpack_from("<I", payload, 24)[0]
        metadata["end_subject_id"] = struct.unpack_from("<I", payload, 28)[0]
        metadata["end_subject_index"] = struct.unpack_from("<I", payload, 32)[0]
    return metadata


def _infer_connector_edges(
    nodes: list[dict[str, object]],
    connectors: list[dict[str, object]],
) -> list[dict[str, object]]:
    bbox_nodes = [
        (str(node.get("id", "")), bbox)
        for node in nodes
        if (bbox := _diagram_node_bbox(node)) is not None
    ]
    edges: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    edge_labels = _diagram_connector_labels(nodes)
    for connector_index, connector in enumerate(connectors):
        connector_id = str(connector.get("id", ""))
        subject_edge = _edge_from_connector_subject_ids(
            connector,
            nodes,
            connector_index=connector_index,
            edge_labels=edge_labels,
        )
        if subject_edge is not None:
            key = (
                str(subject_edge["from"]),
                str(subject_edge["to"]),
                connector_id,
            )
            if key not in seen:
                seen.add(key)
                edges.append(subject_edge)
            continue
        points = connector.get("points")
        if len(bbox_nodes) < 2:
            continue
        if not isinstance(points, list) or len(points) < 2:
            continue
        start = _diagram_point(points[0])
        end = _diagram_point(points[1])
        if start is None or end is None:
            continue
        from_id = _nearest_node_id(start, bbox_nodes)
        to_id = _nearest_node_id(end, bbox_nodes)
        if from_id is None or to_id is None or from_id == to_id:
            continue
        key = (from_id, to_id, connector_id)
        if key in seen:
            continue
        seen.add(key)
        edges.append(
            {
                "from": from_id,
                "to": to_id,
                "type": "arrow" if connector.get("arrow") else "line",
                "label": (
                    edge_labels[connector_index]
                    if connector_index < len(edge_labels)
                    else ""
                ),
                "confidence": "inferred_geometry",
                "connector_id": connector_id,
            }
        )
    return edges


def _edge_from_connector_subject_ids(
    connector: dict[str, object],
    nodes: list[dict[str, object]],
    *,
    connector_index: int,
    edge_labels: list[str],
) -> dict[str, object] | None:
    metadata = connector.get("metadata")
    if not isinstance(metadata, dict):
        return None
    start_subject_id = metadata.get("start_subject_id")
    end_subject_id = metadata.get("end_subject_id")
    if start_subject_id in (None, 0) or end_subject_id in (None, 0):
        return None
    node_by_instance_id = _nodes_by_instance_id(nodes)
    from_id = node_by_instance_id.get(str(start_subject_id))
    to_id = node_by_instance_id.get(str(end_subject_id))
    if from_id is None or to_id is None or from_id == to_id:
        return None
    connector_id = str(connector.get("id", ""))
    return {
        "from": from_id,
        "to": to_id,
        "type": "arrow" if connector.get("arrow") else "line",
        "label": (
            edge_labels[connector_index]
            if connector_index < len(edge_labels)
            else ""
        ),
        "confidence": "parsed_subject_ids",
        "connector_id": connector_id,
    }


def _nodes_by_instance_id(nodes: list[dict[str, object]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for node in nodes:
        metadata = node.get("metadata")
        if not isinstance(metadata, dict):
            continue
        instance_id = metadata.get("instance_id")
        if instance_id in (None, ""):
            continue
        result[str(instance_id)] = str(node.get("id", ""))
    return result


def _diagram_connector_labels(nodes: list[dict[str, object]]) -> list[str]:
    labels: list[str] = []
    for node in nodes:
        if _diagram_node_bbox(node) is not None:
            continue
        text = str(node.get("text", "")).strip()
        if not text:
            continue
        if _is_diagram_step_label(text):
            labels.append(text)
            continue
        if (
            labels
            and not _is_diagram_section_heading(text)
            and not _is_diagram_note(text)
        ):
            labels[-1] = f"{labels[-1]}\n{text}"
    return labels


def _is_diagram_step_label(text: str) -> bool:
    return bool(_DIAGRAM_STEP_LABEL_RE.match(text.strip()))


def _is_diagram_section_heading(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith("<") and stripped.endswith(">")


def _is_diagram_note(text: str) -> bool:
    stripped = text.strip()
    return (
        (stripped.startswith("(") and stripped.endswith(")"))
        or (stripped.startswith("[") and stripped.endswith("]"))
    )


def _diagram_node_bbox(
    node: dict[str, object],
) -> dict[str, int] | None:
    bbox = node.get("bbox")
    if not isinstance(bbox, dict):
        return None
    x = _bbox_int(bbox, "x")
    y = _bbox_int(bbox, "y")
    width = _bbox_int(bbox, "width")
    height = _bbox_int(bbox, "height")
    if width <= 0 or height <= 0:
        return None
    return {"x": x, "y": y, "width": width, "height": height}


def _diagram_point(point: object) -> dict[str, int] | None:
    if not isinstance(point, dict):
        return None
    try:
        return {"x": int(point["x"]), "y": int(point["y"])}
    except (KeyError, TypeError, ValueError):
        return None


def _nearest_node_id(
    point: dict[str, int],
    bbox_nodes: list[tuple[str, dict[str, int]]],
) -> str | None:
    best_id: str | None = None
    best_distance: int | None = None
    for node_id, bbox in bbox_nodes:
        distance = _point_bbox_distance_squared(point, bbox)
        if best_distance is None or distance < best_distance:
            best_id = node_id
            best_distance = distance
    return best_id


def _point_bbox_distance_squared(
    point: dict[str, int],
    bbox: dict[str, int],
) -> int:
    min_x = bbox["x"]
    max_x = bbox["x"] + bbox["width"]
    min_y = bbox["y"]
    max_y = bbox["y"] + bbox["height"]
    dx = max(min_x - point["x"], 0, point["x"] - max_x)
    dy = max(min_y - point["y"], 0, point["y"] - max_y)
    return dx * dx + dy * dy


def _diagram_source_text(diagram: dict[str, object]) -> str:
    nodes = diagram.get("nodes", [])
    if not isinstance(nodes, list):
        return ""
    lines = [
        text
        for text in (
            str(node.get("text", "")).strip()
            for node in nodes
            if isinstance(node, dict)
        )
        if text
    ]
    edge_lines = _diagram_edge_source_lines(diagram.get("edges", []))
    if edge_lines:
        lines.append("relations:")
        lines.extend(edge_lines)
    return "\n".join(lines)


def _diagram_edge_source_lines(edges: object) -> list[str]:
    if not isinstance(edges, list):
        return []
    lines = []
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        from_id = str(edge.get("from", "")).strip()
        to_id = str(edge.get("to", "")).strip()
        if not from_id or not to_id:
            continue
        label = str(edge.get("label", "")).strip()
        line = f"{from_id} -> {to_id}"
        if label:
            line = f"{line}: {label}"
        lines.append(line)
    return lines


def _unresolved_connector_ids(diagram: dict[str, object]) -> list[str]:
    connectors = diagram.get("connectors", [])
    edges = diagram.get("edges", [])
    if not isinstance(connectors, list) or not isinstance(edges, list):
        return []
    resolved = {
        str(edge.get("connector_id", ""))
        for edge in edges
        if isinstance(edge, dict) and edge.get("connector_id")
    }
    return [
        str(connector.get("id", ""))
        for connector in connectors
        if isinstance(connector, dict)
        and connector.get("id")
        and str(connector.get("id")) not in resolved
    ]


def _apply_ocr_fallback(
    document: ParsedDocument,
    ocr_fn: Callable[[bytes, int], str] | None,
) -> ParsedDocument:
    if ocr_fn is None or not _should_ocr_hwp5(document):
        return document

    units = list(document.units)
    warnings = list(document.quality_warnings)
    next_index = len(units) + 1
    for image_index, asset in enumerate(document.assets):
        if asset.kind != "image":
            continue
        try:
            text = _clean_text(ocr_fn(asset.data, image_index) or "")
        except Exception as exc:
            warnings.append(_hwp5_ocr_warning(asset.id, str(exc)))
            continue
        if not text:
            warnings.append(_hwp5_ocr_warning(asset.id, "empty OCR result"))
            continue
        units.append(
            EvidenceUnit(
                id=f"b{next_index}",
                type="text",
                format="plain",
                source=SourceEvidence(kind="text", text=text),
                content=text,
                metadata={
                    "common": {
                        "chunk_kind": "ocr",
                        "section_path": [],
                        "display_format": "plain",
                    },
                    "ocr": {
                        "source": "hwp5_image",
                        "asset_id": asset.id,
                    },
                },
            )
        )
        next_index += 1
    return ParsedDocument(
        units=units,
        assets=document.assets,
        quality_warnings=warnings,
    )


def _should_ocr_hwp5(document: ParsedDocument) -> bool:
    if not document.assets:
        return False
    for unit in document.units:
        if unit.type == "image":
            continue
        if unit.source.text.strip():
            return False
    return True


def _hwp5_ocr_warning(asset_id: str, message: str) -> dict[str, Any]:
    return {
        "type": "hwp5_ocr_failed",
        "severity": "medium",
        "asset_id": asset_id,
        "stage": "ocr",
        "message": message,
    }


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

    def close_gso() -> None:
        nonlocal in_gso, gso_level, gso_text_parts, gso_bbox
        nonlocal gso_shape_type, gso_line_payload, gso_instance_id, gso_ctrl_id
        text = _clean_text(" ".join(gso_text_parts))
        if gso_shape_type in {"line", "connector"}:
            line_block = _DrawingLineBlock(
                    bbox=gso_bbox,
                    points=_line_points_from_bbox(gso_bbox, gso_line_payload),
                    arrow=_line_has_arrow(gso_line_payload),
                    ctrl_id=gso_ctrl_id,
                    instance_id=gso_instance_id,
                    payload=gso_line_payload,
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
        elif text or gso_bbox is not None or gso_shape_type is not None:
            shape_type = gso_shape_type or ("textbox" if text else "shape")
            text_block = _TextBlock(
                text,
                origin="drawing",
                bbox=gso_bbox,
                shape_type=shape_type,
                instance_id=gso_instance_id,
                ctrl_id=gso_ctrl_id,
            )
            if table_stack and table_stack[-1].in_cell:
                table_stack[-1].current_cell_children.append(
                    {
                        "type": "diagram",
                        "format": "structured_diagram",
                        "content": _structured_diagram(
                            text,
                            nodes=[_diagram_node_from_text_block(1, text_block)],
                            connectors=[],
                        ),
                    }
                )
            else:
                parsed.blocks.append(text_block)
            parsed.saw_drawing = True
        in_gso = False
        gso_level = -1
        gso_text_parts = []
        gso_bbox = None
        gso_shape_type = None
        gso_line_payload = None
        gso_instance_id = None
        gso_ctrl_id = None

    def close_current_cell(ctx: _TableCtx) -> None:
        if not ctx.in_cell:
            return
        ctx.current_row.append(
            _Cell(
                text=_clean_text(" ".join(ctx.current_cell_parts)),
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
            elif ctrl in _SHAPE_CTRL_TYPES:
                shape_type = _SHAPE_CTRL_TYPES[ctrl]
                if in_gso and level <= gso_level:
                    close_gso()
                if not in_gso:
                    in_gso = True
                    gso_level = level
                    gso_text_parts = []
                    gso_line_payload = None
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


def _ctrl_id_text(ctrl: bytes) -> str:
    try:
        return ctrl.decode("ascii", errors="replace")
    except UnicodeDecodeError:
        return ctrl.hex()


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


def _bbox_int(bbox: dict[str, int | str], key: str) -> int:
    try:
        return int(bbox[key])
    except (KeyError, TypeError, ValueError):
        return 0


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


def _structured_table(
    rows: list[list[_Cell]],
    *,
    row_count: int | None = None,
    column_count: int | None = None,
) -> dict[str, object]:
    normalized = _normalize_rows(
        rows,
        row_count=row_count,
        column_count=column_count,
    )
    if not normalized:
        return _structured_table_content(columns=[], rows=[])

    actual_column_count = _table_column_count(normalized, column_count)
    header_count = _header_row_count(normalized)
    header_raw_rows = normalized[:header_count]
    visible_header_raw_rows = [
        row for row in header_raw_rows if not _row_is_blank(row)
    ]
    data_raw_rows = normalized[header_count:]
    columns = _table_columns(actual_column_count, visible_header_raw_rows)
    omit_blank_cells = _should_omit_blank_cells(
        normalized,
        column_count=actual_column_count,
    )
    header_rows = [
        {
            "index": index,
            "cells": _evidence_cells(
                raw_cells,
                columns,
                omit_blank_cells=omit_blank_cells,
            ),
        }
        for index, raw_cells in enumerate(visible_header_raw_rows, start=1)
    ]

    data_rows: list[dict[str, object]] = []
    for row in data_raw_rows:
        if row_count is None and _row_is_blank(row):
            continue
        data_rows.append(
            {
                "index": len(data_rows) + 1,
                "cells": _evidence_cells(
                    row,
                    columns,
                    omit_blank_cells=omit_blank_cells,
                ),
            }
        )

    structured = _structured_table_content(
        columns=columns,
        rows=data_rows,
        header_rows=header_rows,
    )
    omitted_count = _omitted_blank_cell_count(
        normalized,
        omit=omit_blank_cells,
    )
    if omitted_count:
        structured["compact"] = {"omitted_blank_cells": omitted_count}
    return structured


def _evidence_cells(
    row: list[_Cell],
    columns: list[dict[str, str]],
    *,
    omit_blank_cells: bool = False,
) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    for cell in sorted(row, key=lambda item: item.col_addr or 0):
        if omit_blank_cells and not cell.text.strip() and not cell.children:
            continue
        column_index = cell.col_addr or 0
        column_id = (
            columns[column_index]["id"]
            if 0 <= column_index < len(columns)
            else f"c{column_index + 1}"
        )
        cells.append(
            {
                "column_id": column_id,
                "text": cell.text,
                "rowspan": cell.rowspan,
                "colspan": cell.colspan,
                "children": list(cell.children),
            }
        )
    return cells


def _normalize_rows(
    rows: list[list[_Cell]],
    *,
    row_count: int | None = None,
    column_count: int | None = None,
) -> list[list[_Cell]]:
    row_map: dict[int, list[_Cell]] = {}
    all_cells: list[_Cell] = []
    max_row_end = 0
    max_col_end = 0
    for fallback_row, row in enumerate(rows):
        fallback_col = 0
        for cell in row:
            row_addr = (
                cell.row_addr
                if cell.row_addr is not None and cell.row_addr >= 0
                else fallback_row
            )
            col_addr = (
                cell.col_addr
                if cell.col_addr is not None and cell.col_addr >= 0
                else fallback_col
            )
            cleaned = _Cell(
                text=_clean_text(cell.text),
                children=list(cell.children),
                row_addr=row_addr,
                col_addr=col_addr,
                rowspan=max(1, cell.rowspan),
                colspan=max(1, cell.colspan),
                synthetic=cell.synthetic,
            )
            row_map.setdefault(row_addr, []).append(cleaned)
            all_cells.append(cleaned)
            max_row_end = max(max_row_end, row_addr + cleaned.rowspan)
            max_col_end = max(max_col_end, col_addr + cleaned.colspan)
            fallback_col = col_addr + cleaned.colspan

    actual_column_count = max(column_count or 0, max_col_end)
    if actual_column_count <= 0:
        return []
    if row_count is not None:
        row_indexes = list(range(max(row_count, max_row_end)))
    else:
        row_indexes = sorted(row_map)

    normalized: list[list[_Cell]] = []
    for row_index in row_indexes:
        current_cells = row_map.get(row_index, [])
        covered = _covered_columns_from_rowspans(all_cells, row_index)
        occupied = set(covered)
        for cell in current_cells:
            start = cell.col_addr or 0
            occupied.update(range(start, min(actual_column_count, start + cell.colspan)))

        materialized = list(current_cells)
        for col_addr in range(actual_column_count):
            if col_addr in occupied:
                continue
            materialized.append(
                _Cell(row_addr=row_index, col_addr=col_addr, synthetic=True)
            )
        if materialized:
            normalized.append(sorted(materialized, key=lambda item: item.col_addr or 0))
    return normalized


def _should_omit_blank_cells(
    rows: list[list[_Cell]],
    *,
    column_count: int,
) -> bool:
    cells = [cell for row in rows for cell in row]
    if column_count < 10 or len(cells) < 50:
        return False
    blank_count = _omitted_blank_cell_count(rows, omit=True)
    return blank_count / len(cells) >= 0.5


def _omitted_blank_cell_count(
    rows: list[list[_Cell]],
    *,
    omit: bool,
) -> int:
    if not omit:
        return 0
    return sum(
        1
        for row in rows
        for cell in row
        if not cell.text.strip() and not cell.children
    )


def _covered_columns_from_rowspans(cells: list[_Cell], row_index: int) -> set[int]:
    covered: set[int] = set()
    for cell in cells:
        row_addr = cell.row_addr or 0
        if not row_addr < row_index < row_addr + cell.rowspan:
            continue
        col_addr = cell.col_addr or 0
        covered.update(range(col_addr, col_addr + cell.colspan))
    return covered


def _table_column_count(
    rows: list[list[_Cell]],
    declared_column_count: int | None,
) -> int:
    return max(
        declared_column_count or 0,
        max(
            (
                (cell.col_addr or 0) + cell.colspan
                for row in rows
                for cell in row
            ),
            default=0,
        ),
    )


def _table_columns(
    column_count: int,
    header_rows: list[list[_Cell]],
) -> list[dict[str, str]]:
    return [
        {
            "id": f"c{index}",
            "text": _column_header_text(header_rows, index - 1),
        }
        for index in range(1, column_count + 1)
    ]


def _column_header_text(
    header_rows: list[list[_Cell]],
    column_index: int,
) -> str:
    texts: list[str] = []
    last_header_row = len(header_rows) - 1
    for row_index, row in enumerate(header_rows):
        for cell in row:
            if not _header_cell_contributes_to_column(
                cell,
                column_index,
                row_index,
                last_header_row,
            ):
                continue
            text = cell.text.strip()
            if text and text not in texts:
                texts.append(text)
    return " / ".join(texts)


def _header_cell_contributes_to_column(
    cell: _Cell,
    column_index: int,
    row_index: int,
    last_header_row: int,
) -> bool:
    start = cell.col_addr or 0
    end = start + cell.colspan
    return column_index == start or (
        start < column_index < end and row_index < last_header_row
    )


def _header_row_count(rows: list[list[_Cell]]) -> int:
    first_row = rows[0]
    if any(cell.children for cell in first_row):
        return 0
    if len(rows) == 1:
        return 1
    count = 1
    header_row_end = _row_span_end(first_row)
    last_header_row = first_row
    while count < len(rows):
        row = rows[count]
        if _row_is_blank(row):
            if (
                count + 1 < len(rows)
                and _row_refines_previous_header(rows[count + 1], last_header_row)
            ):
                count += 1
                continue
            break
        row_start = _row_start(row)
        if (
            row_start < header_row_end
            or _row_refines_previous_header(row, last_header_row)
        ):
            count += 1
            header_row_end = max(header_row_end, _row_span_end(row))
            last_header_row = row
            continue
        break
    return count


def _row_refines_previous_header(
    row: list[_Cell],
    previous_row: list[_Cell],
) -> bool:
    if any(cell.children for cell in row) or _row_is_blank(row):
        return False
    groups = [
        cell
        for cell in previous_row
        if cell.colspan > 1 and cell.text.strip()
    ]
    if not groups:
        return False
    for group in groups:
        group_start = group.col_addr or 0
        group_end = group_start + group.colspan
        refiners = [
            cell
            for cell in row
            if group_start <= (cell.col_addr or 0)
            and (cell.col_addr or 0) + cell.colspan <= group_end
        ]
        if not any(cell.text.strip() for cell in refiners):
            return False
    return True


def _row_start(row: list[_Cell]) -> int:
    return min((cell.row_addr or 0 for cell in row), default=0)


def _row_span_end(row: list[_Cell]) -> int:
    return max(
        (
            (cell.row_addr or 0) + cell.rowspan
            for cell in row
        ),
        default=0,
    )


def _row_is_blank(row: list[_Cell]) -> bool:
    return not any(cell.text.strip() or cell.children for cell in row)


def _single_cell_table_text(
    rows: list[list[_Cell]],
    *,
    row_count: int | None = None,
    column_count: int | None = None,
) -> str | None:
    normalized = _normalize_rows(
        rows,
        row_count=row_count,
        column_count=column_count,
    )
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
    active_rowspans: list[tuple[int, int, int, dict[str, object]]] = []
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
        row_index = int(row["index"])
        row_cells = list(row["cells"])
        source_row_cells = _source_cells_with_rowspan_context(
            row_cells,
            active_rowspans,
            row_index,
        )
        cells = _table_source_cells(
            source_row_cells,
            column_text,
            use_header_labels=True,
        )
        if cells:
            lines.append(f"row {row['index']}: " + "; ".join(cells))
        active_rowspans = [
            span for span in active_rowspans if span[2] > row_index
        ]
        active_rowspans.extend(_rowspan_source_spans(row_cells, row_index))
    return "\n".join(lines)


def _source_cells_with_rowspan_context(
    cells: list[dict[str, object]],
    active_rowspans: list[tuple[int, int, int, dict[str, object]]],
    row_index: int,
) -> list[dict[str, object]]:
    occupied = [_cell_column_range(cell) for cell in cells]
    context_cells = [
        cell
        for start, end, last_row, cell in active_rowspans
        if last_row >= row_index
        and not any(
            _ranges_overlap(start, end, other_start, other_end)
            for other_start, other_end in occupied
        )
    ]
    return sorted(
        [*context_cells, *cells],
        key=lambda cell: _cell_column_range(cell)[0],
    )


def _rowspan_source_spans(
    cells: list[dict[str, object]],
    row_index: int,
) -> list[tuple[int, int, int, dict[str, object]]]:
    spans: list[tuple[int, int, int, dict[str, object]]] = []
    for cell in cells:
        rowspan = _positive_int(cell.get("rowspan"), default=1)
        if rowspan <= 1:
            continue
        start, end = _cell_column_range(cell)
        spans.append((start, end, row_index + rowspan - 1, cell))
    return spans


def _cell_column_range(cell: dict[str, object]) -> tuple[int, int]:
    column = _column_id_number(str(cell.get("column_id", "c1")))
    colspan = _positive_int(cell.get("colspan"), default=1)
    return column, column + colspan


def _ranges_overlap(
    start: int,
    end: int,
    other_start: int,
    other_end: int,
) -> bool:
    return start < other_end and other_start < end


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
        diagram_texts = [
            "nested diagram: " + diagram_source
            for child in cell["children"]
            if child.get("type", child.get("kind")) == "diagram"
            and (diagram_source := _inline_diagram_source(child["content"]))
        ]
        combined = "; ".join(
            part for part in [value, *child_texts, *image_texts, *diagram_texts] if part
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


def _positive_int(value: object, *, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _inline_table_source(table: dict[str, object]) -> str:
    source = _table_source_text(table)
    return source.replace("\n", " / ")


def _inline_diagram_source(diagram: dict[str, object]) -> str:
    source = _diagram_source_text(diagram)
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
