from __future__ import annotations

import re
import struct
from collections.abc import Mapping
from dataclasses import dataclass

from ...schema import structured_diagram as _structured_diagram_content
from .blocks import DiagramBlock, DrawingLineBlock, ImageBlock, TableBlock, TextBlock
from .text import clean_text as _clean_text

_DIAGRAM_STEP_LABEL_RE = re.compile(
    r"^(?:[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]|\d+[.)])"
)
_DIAGRAM_STEP_TOKEN_RE = re.compile(
    r"(?:[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]|\d+[.)])"
)


@dataclass(frozen=True)
class Hwp5DiagramBuilder:
    """Builds and projects structured diagrams from parsed HWP5 drawing blocks."""

    def coalesce(self, blocks: list[object]) -> list[object]:
        return _coalesce_drawing_text_blocks(blocks)

    def build(
        self,
        text: str,
        *,
        nodes: list[dict[str, object]] | None = None,
        bboxes: list[dict[str, int | str] | None] | None = None,
        connectors: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        return _structured_diagram(
            text,
            nodes=nodes,
            bboxes=bboxes,
            connectors=connectors,
        )

    def node_from_text_block(
        self,
        index: int,
        block: TextBlock,
    ) -> dict[str, object]:
        return _diagram_node_from_text_block(index, block)

    def connector(
        self,
        index: int,
        line: DrawingLineBlock,
    ) -> dict[str, object]:
        return _structured_connector(index, line)

    def source_text(self, diagram: dict[str, object]) -> str:
        return _diagram_source_text(diagram)

    def unresolved_connectors(
        self,
        diagram: dict[str, object],
    ) -> list[dict[str, object]]:
        return _unresolved_connector_details(diagram)

    def unsupported_shape_warning(
        self,
        ctrl: bytes,
        tag_id: int,
        level: int,
    ) -> dict[str, object]:
        return _unsupported_drawing_shape_warning(
            ctrl,
            tag_id=tag_id,
            level=level,
        )

    def ctrl_id_text(self, ctrl: bytes) -> str:
        return _ctrl_id_text(ctrl)

    def bbox_int(self, bbox: dict[str, int | str], key: str) -> int:
        return _bbox_int(bbox, key)


def _coalesce_drawing_text_blocks(
    blocks: list[
        TextBlock | DiagramBlock | DrawingLineBlock | TableBlock | ImageBlock
    ],
) -> list[
    TextBlock | DiagramBlock | DrawingLineBlock | TableBlock | ImageBlock
]:
    result: list[
        TextBlock | DiagramBlock | DrawingLineBlock | TableBlock | ImageBlock
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
            if isinstance(candidate, DrawingLineBlock):
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
            if isinstance(block, TextBlock):
                result.append(TextBlock(block.text))
            index += 1
            continue

        prefix: list[TextBlock] = []
        while (
            result
            and len(prefix) < 3
            and _is_short_body_text_block(result[-1])
        ):
            previous = result.pop()
            if isinstance(previous, TextBlock):
                prefix.append(previous)
        prefix.reverse()

        text_blocks = [
            item
            for item in [*prefix, *blocks[index : cluster_end + 1]]
            if isinstance(item, TextBlock)
        ]
        line_blocks = [
            item
            for item in blocks[index : cluster_end + 1]
            if isinstance(item, DrawingLineBlock)
        ]
        result.append(
            DiagramBlock(
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
    return isinstance(block, TextBlock) and block.origin == "drawing"


def _drawing_block_has_text(block: object) -> bool:
    return isinstance(block, TextBlock) and bool(_clean_text(block.text))


def _is_short_body_text_block(block: object) -> bool:
    return (
        isinstance(block, TextBlock)
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
    block: TextBlock,
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
    line: DrawingLineBlock,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "source": "hwp5_gso_line",
    }
    metadata.update(line.metadata)
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
    if len(payload) >= 16:
        start_x, start_y, end_x, end_y = struct.unpack_from("<4i", payload)
        metadata["raw_start_point"] = {"x": start_x, "y": start_y}
        metadata["raw_end_point"] = {"x": end_x, "y": end_y}
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
    if not isinstance(metadata, Mapping):
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
        if not isinstance(metadata, Mapping):
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
            labels.extend(_split_diagram_step_labels(text))
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


def _split_diagram_step_labels(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    matches = list(_DIAGRAM_STEP_TOKEN_RE.finditer(stripped))
    if not matches or matches[0].start() != 0:
        return [stripped]
    labels: list[str] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(stripped)
        label = stripped[start:end].strip()
        if label:
            labels.append(label)
    return labels or [stripped]


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
    if not isinstance(bbox, Mapping):
        return None
    x = _bbox_int(bbox, "x")
    y = _bbox_int(bbox, "y")
    width = _bbox_int(bbox, "width")
    height = _bbox_int(bbox, "height")
    if width <= 0 or height <= 0:
        return None
    return {"x": x, "y": y, "width": width, "height": height}


def _diagram_point(point: object) -> dict[str, int] | None:
    if not isinstance(point, Mapping):
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
            if isinstance(node, Mapping)
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
        if not isinstance(edge, Mapping):
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


def _unresolved_connector_details(
    diagram: dict[str, object],
) -> list[dict[str, object]]:
    connectors = diagram.get("connectors", [])
    edges = diagram.get("edges", [])
    nodes = diagram.get("nodes", [])
    if not isinstance(connectors, list) or not isinstance(edges, list):
        return []
    resolved = {
        str(edge.get("connector_id", ""))
        for edge in edges
        if isinstance(edge, Mapping) and edge.get("connector_id")
    }
    return [
        _connector_warning_details(
            connector,
            nodes if isinstance(nodes, list) else [],
        )
        for connector in connectors
        if isinstance(connector, Mapping)
        and connector.get("id")
        and str(connector.get("id")) not in resolved
    ]


def _connector_warning_details(
    connector: dict[str, object],
    nodes: list[object],
) -> dict[str, object]:
    payload = {
        "id": str(connector.get("id", "")),
        "type": str(connector.get("type", "")),
        "bbox": connector.get("bbox"),
        "points": connector.get("points") if isinstance(connector.get("points"), list) else [],
        "arrow": bool(connector.get("arrow")),
        "metadata": (
            dict(connector.get("metadata"))
            if isinstance(connector.get("metadata"), Mapping)
            else {}
        ),
        "resolution_failure": _connector_resolution_failure(connector, nodes),
    }
    return payload


def _connector_resolution_failure(
    connector: dict[str, object],
    nodes: list[object],
) -> str:
    node_dicts = [node for node in nodes if isinstance(node, Mapping)]
    metadata = connector.get("metadata")
    if isinstance(metadata, Mapping):
        start_subject_id = metadata.get("start_subject_id")
        end_subject_id = metadata.get("end_subject_id")
        if start_subject_id not in (None, 0) or end_subject_id not in (None, 0):
            node_by_instance_id = _nodes_by_instance_id(node_dicts)
            if (
                str(start_subject_id) not in node_by_instance_id
                or str(end_subject_id) not in node_by_instance_id
            ):
                return "subject_id_unmatched"
            if node_by_instance_id[str(start_subject_id)] == node_by_instance_id[str(end_subject_id)]:
                return "subject_ids_resolve_to_same_node"
    bbox_nodes = [
        (str(node.get("id", "")), bbox)
        for node in node_dicts
        if (bbox := _diagram_node_bbox(node)) is not None
    ]
    if len(bbox_nodes) < 2:
        return "insufficient_bbox_nodes"
    points = connector.get("points")
    if not isinstance(points, list) or len(points) < 2:
        return "invalid_points"
    start = _diagram_point(points[0])
    end = _diagram_point(points[1])
    if start is None or end is None:
        return "invalid_points"
    from_id = _nearest_node_id(start, bbox_nodes)
    to_id = _nearest_node_id(end, bbox_nodes)
    if from_id is not None and from_id == to_id:
        return "endpoints_resolve_to_same_node"
    return "edge_inference_unresolved"


def _unsupported_drawing_shape_warning(
    ctrl: bytes,
    tag_id: int,
    level: int,
) -> dict[str, Any]:
    return {
        "type": "hwp5_unsupported_drawing_shape",
        "severity": "medium",
        "ctrl_id": _ctrl_id_text(ctrl),
        "tag_id": tag_id,
        "level": level,
        "message": "Unsupported HWP5 drawing shape component was preserved as text.",
    }


def _ctrl_id_text(ctrl: bytes) -> str:
    try:
        return ctrl.decode("ascii", errors="replace")
    except UnicodeDecodeError:
        return ctrl.hex()


def _bbox_int(bbox: dict[str, int | str], key: str) -> int:
    try:
        return int(bbox[key])
    except (KeyError, TypeError, ValueError):
        return 0
