from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from xml.etree import ElementTree as ET

from ...schema import structured_diagram as _structured_diagram_content
from .xml_utils import clean_text as _clean_text
from .xml_utils import element_text as _element_text
from .xml_utils import local_name as _local_name
from .xml_utils import q as _q

_DRAWING_NODE_TAGS = {
    "arc", "container", "curve", "ellipse", "polygon", "rect",
    "roundRect", "shapeObject", "textBox", "textbox",
}
_DRAWING_CONNECTOR_TAGS = {"arc", "connectLine", "curve", "line", "polygon"}
_UNSUPPORTED_DRAWING_TAGS = {
    "button", "checkBtn", "comboBox", "edit", "equation", "listBox",
    "ole", "radio", "scrollBar", "textart", "video",
}
_DIAGRAM_STEP_LABEL_RE = re.compile(
    r"^(?:[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]|\d+[.)])"
)


@dataclass(frozen=True)
class DrawingResult:
    structured: dict[str, object] | None = None
    single_text: str | None = None


_DrawingResult = DrawingResult


@dataclass(frozen=True)
class HwpxDiagramBuilder:
    """Builds structured diagrams from HWPX drawing and table geometry."""

    def parse_paragraph(
        self,
        paragraph: ET.Element,
        warnings: list[dict[str, Any]],
    ) -> DrawingResult | None:
        return _paragraph_drawing(paragraph, warnings)

    def source_text(self, diagram: dict[str, object]) -> str:
        return _diagram_source_text(diagram)

    def table_flowchart(
        self,
        table: dict[str, object],
    ) -> dict[str, object] | None:
        return _table_flowchart_diagram(table)

    def without_flowchart_rows(
        self,
        table: dict[str, object],
    ) -> dict[str, object]:
        return _table_without_flowchart_rows(table)


def _paragraph_drawing(
    paragraph: ET.Element,
    warnings: list[dict[str, Any]],
) -> _DrawingResult | None:
    candidates = list(_drawing_candidates(paragraph))
    if not candidates:
        return None

    nodes: list[dict[str, object]] = []
    connectors: list[dict[str, object]] = []
    for element in candidates:
        local = _local_name(element.tag)
        if local in _UNSUPPORTED_DRAWING_TAGS:
            _warn_unsupported_drawing(warnings, local)
            continue
        text = _element_text(element)
        if local in _DRAWING_NODE_TAGS and text:
            nodes.append(
                {
                    "id": f"n{len(nodes) + 1}",
                    "shape_type": local,
                    "text": text,
                    "bbox": _bbox_from_element(element),
                    "metadata": {"source": "hwpx_drawing_text"},
                }
            )
            continue
        if local in _DRAWING_CONNECTOR_TAGS:
            connector = _structured_connector(len(connectors) + 1, element)
            if connector is not None:
                connectors.append(connector)
            continue
        if local not in _DRAWING_NODE_TAGS:
            continue

    if len(nodes) == 1 and not connectors:
        return _DrawingResult(single_text=str(nodes[0]["text"]))
    if not nodes:
        return None

    structured = _structured_diagram_content(
        nodes=nodes,
        edges=_infer_connector_edges(nodes, connectors),
        connectors=connectors,
    )
    return _DrawingResult(structured=structured)


def _drawing_candidates(paragraph: ET.Element) -> list[ET.Element]:
    candidates: list[ET.Element] = []
    for run in paragraph.findall(_q("run")):
        for child in list(run):
            candidates.extend(_drawing_candidates_from_element(child))
    return candidates


def _drawing_candidates_from_element(element: ET.Element) -> list[ET.Element]:
    local = _local_name(element.tag)
    if local in {"drawText", "pic", "tbl"}:
        return []
    if local in _DRAWING_CONNECTOR_TAGS or local in _UNSUPPORTED_DRAWING_TAGS:
        return [element]
    if local in _DRAWING_NODE_TAGS:
        text = _element_text(element)
        if text or local != "container":
            return [element]
        candidates: list[ET.Element] = []
        for child in list(element):
            candidates.extend(_drawing_candidates_from_element(child))
        return candidates

    candidates = []
    for child in list(element):
        candidates.extend(_drawing_candidates_from_element(child))
    return candidates


def _structured_connector(
    index: int,
    element: ET.Element,
) -> dict[str, object] | None:
    bbox = _bbox_from_element(element, allow_flat=True)
    if bbox is None:
        return None
    local = _local_name(element.tag)
    connector_type = "line" if local == "connectLine" else local
    return {
        "id": f"c{index}",
        "type": connector_type,
        "bbox": bbox,
        "points": _points_from_element(element) or _line_points_from_bbox(bbox),
        "arrow": _line_has_arrow(element),
        "metadata": {"source": f"hwpx_{connector_type}"},
    }


def _bbox_from_element(
    element: ET.Element,
    *,
    allow_flat: bool = False,
) -> dict[str, int | str] | None:
    pos = _first_descendant(element, "pos")
    size = _first_descendant(element, "sz")
    x = _int_attr(pos, ("x", "left")) if pos is not None else 0
    y = _int_attr(pos, ("y", "top")) if pos is not None else 0
    width = _int_attr(size, ("width", "w", "cx")) if size is not None else 0
    height = _int_attr(size, ("height", "h", "cy")) if size is not None else 0
    x = _int_attr(element, ("x", "left"), x)
    y = _int_attr(element, ("y", "top"), y)
    width = _int_attr(element, ("width", "w", "cx"), width)
    height = _int_attr(element, ("height", "h", "cy"), height)
    if allow_flat:
        if width <= 0 and height <= 0:
            return None
    elif width <= 0 or height <= 0:
        return None
    return {
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "unit": "hwpx",
    }


def _points_from_element(element: ET.Element) -> list[dict[str, int]]:
    points: list[dict[str, int]] = []
    for descendant in element.iter():
        local = _local_name(descendant.tag)
        if local not in {"pt", "point"}:
            continue
        point = _xy_from_attrs(descendant, ("x",), ("y",))
        if point is not None:
            points.append(point)
    if len(points) >= 2:
        return points

    start = _xy_from_attrs(
        element,
        ("x1", "startX", "fromX"),
        ("y1", "startY", "fromY"),
    )
    end = _xy_from_attrs(
        element,
        ("x2", "endX", "toX"),
        ("y2", "endY", "toY"),
    )
    if start is not None and end is not None:
        return [start, end]
    return []


def _xy_from_attrs(
    element: ET.Element,
    x_names: tuple[str, ...],
    y_names: tuple[str, ...],
) -> dict[str, int] | None:
    x = _optional_int_attr(element, x_names)
    y = _optional_int_attr(element, y_names)
    if x is None or y is None:
        return None
    return {"x": x, "y": y}


def _first_descendant(element: ET.Element, local_name: str) -> ET.Element | None:
    for descendant in element.iter():
        if descendant is element:
            continue
        if _local_name(descendant.tag) == local_name:
            return descendant
    return None


def _int_attr(
    element: ET.Element | None,
    names: tuple[str, ...],
    default: int = 0,
) -> int:
    value = _optional_int_attr(element, names)
    return default if value is None else value


def _optional_int_attr(
    element: ET.Element | None,
    names: tuple[str, ...],
) -> int | None:
    if element is None:
        return None
    for name in names:
        value = element.get(name)
        if value is None:
            continue
        try:
            return int(float(value))
        except ValueError:
            continue
    return None


def _line_points_from_bbox(bbox: dict[str, int | str]) -> list[dict[str, int]]:
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
    return [{"x": x, "y": y}, {"x": x + width, "y": y + height}]


def _line_has_arrow(element: ET.Element) -> bool:
    for item in element.iter():
        for attr, value in item.attrib.items():
            if "arrow" not in _local_name(attr).lower():
                continue
            normalized = value.strip().lower()
            if normalized not in {"", "0", "false", "none", "null"}:
                return True
    return False


def _infer_connector_edges(
    nodes: list[dict[str, object]],
    connectors: list[dict[str, object]],
) -> list[dict[str, object]]:
    bbox_nodes = [
        (str(node.get("id", "")), bbox)
        for node in nodes
        if (bbox := _diagram_node_bbox(node)) is not None
    ]
    if len(bbox_nodes) < 2:
        return []

    edges: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    edge_labels = _diagram_connector_labels(nodes)
    for connector_index, connector in enumerate(connectors):
        points = connector.get("points")
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
        connector_id = str(connector.get("id", ""))
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


def _diagram_node_bbox(node: dict[str, object]) -> dict[str, int] | None:
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


def _table_flowchart_diagram(table: dict[str, object]) -> dict[str, object] | None:
    rows = table.get("rows")
    if not isinstance(rows, list):
        return None

    flowchart_rows = _flowchart_rows(rows)
    if not flowchart_rows:
        return None

    min_row_addr = _min_flowchart_row_addr(flowchart_rows)
    nodes: list[dict[str, object]] = []
    label_nodes: list[dict[str, object]] = []
    connectors: list[dict[str, object]] = []
    seen_node_texts: set[str] = set()
    for row in flowchart_rows:
        if not isinstance(row, Mapping):
            continue
        cells = row.get("cells")
        if not isinstance(cells, list):
            continue
        for cell in cells:
            if not isinstance(cell, Mapping):
                continue
            text = _flowchart_cell_text(cell)
            if not text:
                continue
            if _is_flowchart_edge_label(text):
                label = _flowchart_edge_label(text)
                connectors.append(
                    _table_flowchart_connector(
                        len(connectors) + 1,
                        text,
                        label,
                        _table_cell_grid_bbox(cell, min_row_addr),
                    )
                )
                label_nodes.append(
                    {
                        "id": "",
                        "shape_type": "label",
                        "text": label,
                        "bbox": None,
                        "metadata": {
                            "source": "hwpx_table_flowchart_label",
                            "raw_label": text,
                        },
                    }
                )
                continue
            if text not in seen_node_texts:
                seen_node_texts.add(text)
                nodes.append(
                    {
                        "id": "",
                        "shape_type": "label",
                        "text": text,
                        "bbox": _table_cell_grid_bbox(cell, min_row_addr),
                        "metadata": {
                            "source": "hwpx_table_flowchart",
                            "role": (
                                "title" if _is_flowchart_title(text) else "node"
                            ),
                        },
                    }
                )

    if len(nodes) < 2:
        return None

    all_nodes = _assign_node_ids(nodes + label_nodes)
    edges = _infer_table_grid_edges(all_nodes, connectors)
    return _structured_diagram_content(nodes=all_nodes, edges=edges, connectors=connectors)


def _assign_node_ids(nodes: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            **node,
            "id": f"n{index}",
        }
        for index, node in enumerate(nodes, start=1)
    ]


def _min_flowchart_row_addr(rows: list[object]) -> int:
    row_addrs = [
        int(cell["row_addr"])
        for row in rows
        if isinstance(row, Mapping)
        for cell in row.get("cells", [])
        if isinstance(cell, Mapping) and "row_addr" in cell
    ]
    return min(row_addrs, default=0)


def _table_cell_grid_bbox(
    cell: dict[str, object],
    min_row_addr: int,
) -> dict[str, int | str]:
    return {
        "x": _int_value(cell.get("col_addr"), 0),
        "y": max(0, _int_value(cell.get("row_addr"), min_row_addr) - min_row_addr),
        "width": _int_value(cell.get("colspan"), 1),
        "height": _int_value(cell.get("rowspan"), 1),
        "unit": "hwpx_table_grid",
    }


def _int_value(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _table_flowchart_connector(
    index: int,
    raw_label: str,
    label: str,
    bbox: dict[str, int | str],
) -> dict[str, object]:
    return {
        "id": f"c{index}",
        "type": "arrow",
        "bbox": bbox,
        "points": _table_flowchart_connector_points(raw_label, bbox),
        "arrow": True,
        "metadata": {
            "source": "hwpx_table_flowchart",
            "label": label,
            "raw_label": raw_label,
        },
    }


def _table_flowchart_connector_points(
    raw_label: str,
    bbox: dict[str, int | str],
) -> list[dict[str, float | int]]:
    x = _int_value(bbox.get("x"), 0)
    y = _int_value(bbox.get("y"), 0)
    width = _int_value(bbox.get("width"), 1)
    height = _int_value(bbox.get("height"), 1)
    x_mid = x + width / 2
    y_mid = y + height / 2
    arrow = _flowchart_arrow(raw_label)
    if arrow == "←":
        return [{"x": x + width, "y": y_mid}, {"x": x, "y": y_mid}]
    if arrow == "↑":
        return [{"x": x_mid, "y": y + height}, {"x": x_mid, "y": y}]
    if arrow == "↕":
        return [{"x": x_mid, "y": y + height}, {"x": x_mid, "y": y}]
    if arrow == "↓":
        return [{"x": x_mid, "y": y}, {"x": x_mid, "y": y + height}]
    return [{"x": x, "y": y_mid}, {"x": x + width, "y": y_mid}]


def _flowchart_arrow(text: str) -> str:
    match = re.search(r"[→←↑↓↕]", text)
    return match.group(0) if match is not None else "→"


def _flowchart_edge_label(text: str) -> str:
    label = re.sub(r"^[→←↑↓↕\s]+", "", text).strip()
    if label.startswith("(") and label.endswith(")"):
        label = label[1:-1].strip()
    return label or text


def _infer_table_grid_edges(
    nodes: list[dict[str, object]],
    connectors: list[dict[str, object]],
) -> list[dict[str, object]]:
    bbox_nodes = [
        (str(node.get("id", "")), bbox)
        for node in nodes
        if not _is_table_flowchart_title_node(node)
        and (bbox := _diagram_node_bbox(node)) is not None
    ]
    edges: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for connector in connectors:
        points = connector.get("points")
        if not isinstance(points, list) or len(points) < 2:
            continue
        start = _diagram_point(points[0])
        end = _diagram_point(points[1])
        if start is None or end is None:
            continue
        from_id = _nearest_node_id(start, bbox_nodes)
        to_id = _nearest_node_id(end, bbox_nodes)
        if from_id is not None and from_id == to_id:
            from_id, to_id = _directional_table_grid_edge_node_ids(
                connector,
                bbox_nodes,
            )
        if from_id is None or to_id is None or from_id == to_id:
            continue
        connector_id = str(connector.get("id", ""))
        key = (from_id, to_id, connector_id)
        if key in seen:
            continue
        seen.add(key)
        metadata = connector.get("metadata")
        label = (
            str(metadata.get("label", "")).strip()
            if isinstance(metadata, Mapping)
            else ""
        )
        edges.append(
            {
                "from": from_id,
                "to": to_id,
                "type": "arrow",
                "label": label,
                "confidence": "inferred_table_grid",
                "connector_id": connector_id,
            }
        )
    return edges


def _directional_table_grid_edge_node_ids(
    connector: dict[str, object],
    bbox_nodes: list[tuple[str, dict[str, int]]],
) -> tuple[str | None, str | None]:
    bbox = connector.get("bbox")
    if not isinstance(bbox, Mapping):
        return None, None
    center = {
        "x": _int_value(bbox.get("x"), 0)
        + _int_value(bbox.get("width"), 1) / 2,
        "y": _int_value(bbox.get("y"), 0)
        + _int_value(bbox.get("height"), 1) / 2,
    }
    metadata = connector.get("metadata")
    raw_label = (
        str(metadata.get("raw_label", ""))
        if isinstance(metadata, Mapping)
        else ""
    )
    arrow = _flowchart_arrow(raw_label)
    if arrow == "←":
        return (
            _nearest_directional_node_id(center, bbox_nodes, "right"),
            _nearest_directional_node_id(center, bbox_nodes, "left"),
        )
    if arrow in {"↑", "↕"}:
        return (
            _nearest_directional_node_id(center, bbox_nodes, "below"),
            _nearest_directional_node_id(center, bbox_nodes, "above"),
        )
    if arrow == "↓":
        return (
            _nearest_directional_node_id(center, bbox_nodes, "above"),
            _nearest_directional_node_id(center, bbox_nodes, "below"),
        )
    return (
        _nearest_directional_node_id(center, bbox_nodes, "left"),
        _nearest_directional_node_id(center, bbox_nodes, "right"),
    )


def _nearest_directional_node_id(
    point: dict[str, float],
    bbox_nodes: list[tuple[str, dict[str, int]]],
    direction: str,
) -> str | None:
    candidates = [
        (node_id, bbox)
        for node_id, bbox in bbox_nodes
        if _bbox_is_in_direction(point, bbox, direction)
    ]
    return _nearest_node_id(point, candidates)


def _bbox_is_in_direction(
    point: dict[str, float],
    bbox: dict[str, int],
    direction: str,
) -> bool:
    if direction == "left":
        return bbox["x"] + bbox["width"] <= point["x"]
    if direction == "right":
        return bbox["x"] >= point["x"]
    if direction == "above":
        return bbox["y"] + bbox["height"] <= point["y"]
    if direction == "below":
        return bbox["y"] >= point["y"]
    return False


def _is_table_flowchart_title_node(node: dict[str, object]) -> bool:
    metadata = node.get("metadata")
    return isinstance(metadata, Mapping) and metadata.get("role") == "title"


def _table_without_flowchart_rows(
    table: dict[str, object],
) -> dict[str, object]:
    rows = table.get("rows")
    if not isinstance(rows, list):
        return table
    flowchart_rows = _flowchart_rows(rows)
    if not flowchart_rows:
        return table

    flowchart_row_ids = {id(row) for row in flowchart_rows}
    return {
        **table,
        "rows": [
            row
            for row in rows
            if id(row) not in flowchart_row_ids
        ],
    }


def _flowchart_rows(rows: list[object]) -> list[object]:
    start: int | None = None
    for index, row in enumerate(rows):
        texts = _row_cell_texts(row)
        if any(_is_flowchart_title(text) for text in texts):
            start = index
            break
    if start is None:
        return []

    result: list[object] = []
    for row in rows[start:]:
        texts = _row_cell_texts(row)
        if result and any(_is_paper_size_note(text) for text in texts):
            break
        result.append(row)
    return result


def _row_cell_texts(row: object) -> list[str]:
    if not isinstance(row, Mapping):
        return []
    cells = row.get("cells")
    if not isinstance(cells, list):
        return []
    return [
        text
        for cell in cells
        if isinstance(cell, Mapping)
        and (text := _flowchart_cell_text(cell))
    ]


def _flowchart_cell_text(cell: dict[str, object]) -> str:
    return _clean_text(str(cell.get("text", ""))).strip("<> ")


def _is_flowchart_title(text: str) -> bool:
    return "등록절차" in text or "처리절차" in text or "업무처리" in text


def _is_flowchart_edge_label(text: str) -> bool:
    return bool(re.search(r"[→←↑↓↕]", text))


def _is_paper_size_note(text: str) -> bool:
    return "mm×" in text or "일반용지" in text


def _diagram_edge_source_lines(edges: object) -> list[str]:
    if not isinstance(edges, list):
        return []
    lines: list[str] = []
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


def _bbox_int(bbox: dict[str, int | str], key: str) -> int:
    try:
        return int(bbox[key])
    except (KeyError, TypeError, ValueError):
        return 0


def _warn_unsupported_drawing(
    warnings: list[dict[str, Any]],
    element: str,
) -> None:
    warning = {
        "type": "hwpx_drawing_structure_unsupported",
        "severity": "medium",
        "element": element,
        "message": f"Unsupported HWPX drawing structure was skipped: {element}",
    }
    if warning not in warnings:
        warnings.append(warning)
