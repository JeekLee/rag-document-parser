from __future__ import annotations

import re
from html import escape
from typing import Any

_DIAGRAM_STEP_RE = re.compile(r"^(?:[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]|\d+[.)])")


def render_evidence_units_html(
    units: list[dict[str, Any]],
    *,
    title: str = "Evidence units",
    assets: list[dict[str, Any]] | None = None,
) -> str:
    assets_by_id = _assets_by_id(assets)
    parts = [
        "<!doctype html>",
        '<html lang="ko">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{escape(title)}</title>",
        "<style>",
        _CSS,
        "</style>",
        "</head>",
        "<body>",
        "<main>",
        f"<h1>{escape(title)}</h1>",
    ]
    for unit in units:
        if isinstance(unit, dict):
            parts.append(_render_evidence_unit(unit, assets_by_id))
    parts.extend(["</main>", "</body>", "</html>"])
    return "\n".join(parts)


def render_evidence_html(
    evidence: dict[str, Any],
    assets_by_id: dict[str, dict[str, Any]] | None = None,
) -> str:
    if assets_by_id is None:
        assets_by_id = {}
    kind = evidence.get("kind")
    fmt = evidence.get("format")
    content = evidence.get("content")
    if kind == "table" and fmt == "structured_table" and isinstance(content, dict):
        return _render_structured_table(content, assets_by_id)
    if kind == "diagram" and fmt == "structured_diagram" and isinstance(content, dict):
        return _render_structured_diagram(content)
    if fmt == "asset_ref" and isinstance(content, dict):
        return _render_asset_ref(kind, content, assets_by_id)
    if isinstance(content, str):
        return f"<p>{escape(content)}</p>"
    return f"<pre>{escape(str(content))}</pre>"


def _render_evidence_unit(
    unit: dict[str, Any],
    assets_by_id: dict[str, dict[str, Any]],
) -> str:
    evidence = unit.get("evidence", {})
    source = unit.get("source", {})
    source_text = source.get("text", "") if isinstance(source, dict) else ""
    return (
        '<section class="chunk">'
        '<header class="chunk-header">'
        f"<code>{escape(str(unit.get('id', '')))}</code>"
        f"<span>{escape(str(unit.get('type', '')))}</span>"
        "</header>"
        f'<pre class="source-text">{escape(str(source_text))}</pre>'
        f"{render_evidence_html(evidence, assets_by_id) if isinstance(evidence, dict) else ''}"
        "</section>"
    )


def _render_structured_table(
    table: dict[str, Any],
    assets_by_id: dict[str, dict[str, Any]],
) -> str:
    columns = [
        column
        for column in table.get("columns", [])
        if isinstance(column, dict)
    ]
    rows = [row for row in table.get("rows", []) if isinstance(row, dict)]
    html = ['<table class="evidence-table">']
    header_rows = [
        row
        for row in table.get("header_rows", [])
        if isinstance(row, dict)
    ]
    if header_rows:
        html.append("<thead>")
        for row in header_rows:
            html.append("<tr>")
            html.append(
                _render_table_row_cells(
                    row.get("cells", []),
                    "th",
                    assets_by_id,
                    column_count=len(columns),
                )
            )
            html.append("</tr>")
        html.append("</thead>")
    elif columns:
        html.append("<thead><tr>")
        for column in columns:
            text = escape(str(column.get("text", ""))) or "&nbsp;"
            html.append(f"<th>{text}</th>")
        html.append("</tr></thead>")
    html.append("<tbody>")
    for row in rows:
        html.append("<tr>")
        html.append(
            _render_table_row_cells(
                row.get("cells", []),
                "td",
                assets_by_id,
                column_count=len(columns),
            )
        )
        if not row.get("cells") and columns:
            html.append(f'<td colspan="{len(columns)}">&nbsp;</td>')
        html.append("</tr>")
    if not rows and columns:
        html.append(f'<tr><td colspan="{len(columns)}">&nbsp;</td></tr>')
    html.append("</tbody></table>")
    if not columns and not rows:
        return "<p class=\"empty-table\">빈 표</p>"
    return "".join(html)


def _render_structured_diagram(diagram: dict[str, Any]) -> str:
    nodes = [node for node in diagram.get("nodes", []) if isinstance(node, dict)]
    edges = [edge for edge in diagram.get("edges", []) if isinstance(edge, dict)]
    connectors = [
        connector
        for connector in diagram.get("connectors", [])
        if isinstance(connector, dict)
    ]
    mermaid = diagram.get("mermaid")
    if _is_positionable_label_diagram(nodes, edges, mermaid, connectors):
        return _render_positioned_label_diagram(nodes, connectors=connectors)
    if _is_label_only_diagram(nodes, edges, mermaid):
        return _render_label_flowchart_diagram(nodes)

    html = ['<section class="diagram-evidence">']
    caption = str(diagram.get("caption") or "").strip()
    if caption:
        html.append(f"<h2>{escape(caption)}</h2>")
    if nodes:
        html.append('<ol class="diagram-nodes">')
        for node in nodes:
            node_id = escape(str(node.get("id", "")))
            shape_type = str(node.get("shape_type", node.get("type", ""))).strip()
            shape_badge = (
                f' <span class="diagram-shape">{escape(shape_type)}</span>'
                if shape_type and shape_type != "label"
                else ""
            )
            text = escape(str(node.get("text", ""))) or "&nbsp;"
            html.append(
                "<li>"
                f"<code>{node_id}</code>"
                f"{shape_badge} "
                f"<span>{text}</span>"
                "</li>"
            )
        html.append("</ol>")
    if edges:
        html.append('<ul class="diagram-edges">')
        for edge in edges:
            from_id = escape(str(edge.get("from", "")))
            to_id = escape(str(edge.get("to", "")))
            label = str(edge.get("label", "")).strip()
            confidence = str(edge.get("confidence", "")).strip()
            details = " · ".join(
                escape(value)
                for value in [label, confidence]
                if value
            )
            suffix = f" <span>{details}</span>" if details else ""
            html.append(f"<li><code>{from_id} → {to_id}</code>{suffix}</li>")
        html.append("</ul>")
    if isinstance(mermaid, str) and mermaid.strip():
        html.append(f'<pre class="mermaid">{escape(mermaid.strip())}</pre>')
    if not nodes and not edges and not (isinstance(mermaid, str) and mermaid.strip()):
        html.append('<p class="empty-diagram">빈 다이어그램</p>')
    html.append("</section>")
    return "".join(html)


def _is_label_only_diagram(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    mermaid: Any,
) -> bool:
    return (
        bool(nodes)
        and not edges
        and not (isinstance(mermaid, str) and mermaid.strip())
        and all(
            str(node.get("shape_type", node.get("type", "label"))).strip()
            in ("", "label")
            for node in nodes
        )
    )


def _is_positionable_label_diagram(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    mermaid: Any,
    connectors: list[dict[str, Any]],
) -> bool:
    return (
        bool(nodes)
        and not (isinstance(mermaid, str) and mermaid.strip())
        and all(_is_label_shape_node(node) for node in nodes)
        and (
            sum(1 for node in nodes if _node_bbox(node) is not None) >= 2
            or any(_connector_line(connector) is not None for connector in connectors)
        )
    )


def _is_label_shape_node(node: dict[str, Any]) -> bool:
    return (
        str(node.get("shape_type", node.get("type", "label"))).strip()
        in ("", "label")
    )


def _render_positioned_label_diagram(
    nodes: list[dict[str, Any]],
    *,
    connectors: list[dict[str, Any]],
) -> str:
    positioned = [
        (node, bbox)
        for node in nodes
        if (bbox := _node_bbox(node)) is not None
    ]
    connector_lines = [
        line
        for connector in connectors
        if (line := _connector_line(connector)) is not None
    ]
    unpositioned = [
        str(node.get("text", "")).strip()
        for node in nodes
        if _node_bbox(node) is None and str(node.get("text", "")).strip()
    ]
    strip_labels, connector_labels = _split_unpositioned_diagram_labels(unpositioned)
    paired_connector_labels = connector_labels[: len(connector_lines)]
    strip_labels.extend(connector_labels[len(connector_lines) :])
    x_values = [
        value
        for _, bbox in positioned
        for value in (bbox["x"], bbox["x"] + bbox["width"])
    ]
    y_values = [
        value
        for _, bbox in positioned
        for value in (bbox["y"], bbox["y"] + bbox["height"])
    ]
    x_values.extend(point["x"] for line in connector_lines for point in line["points"])
    y_values.extend(point["y"] for line in connector_lines for point in line["points"])
    min_x = min(x_values)
    min_y = min(y_values)
    max_x = max(x_values)
    max_y = max(y_values)
    canvas_width = max(max_x - min_x, 1.0)
    canvas_height = max(max_y - min_y, 1.0)

    html = ['<section class="diagram-positioned">']
    if strip_labels:
        html.append('<div class="diagram-positioned-labels">')
        for text in strip_labels:
            html.append(
                '<span class="diagram-positioned-label">'
                f"{_escape_multiline(text)}"
                "</span>"
            )
        html.append("</div>")
    html.append(
        '<div class="diagram-canvas" '
        f'style="aspect-ratio:{canvas_width:.3f}/{canvas_height:.3f}">'
    )
    if connector_lines:
        html.append('<svg class="diagram-connectors" aria-hidden="true">')
        if any(line["arrow"] for line in connector_lines):
            html.append(
                '<defs><marker id="diagram-arrow" markerWidth="8" '
                'markerHeight="8" refX="7" refY="4" orient="auto">'
                '<path d="M0,0 L8,4 L0,8 Z"></path>'
                "</marker></defs>"
            )
        for line in connector_lines:
            start, end = line["points"]
            x1 = (start["x"] - min_x) / canvas_width * 100
            y1 = (start["y"] - min_y) / canvas_height * 100
            x2 = (end["x"] - min_x) / canvas_width * 100
            y2 = (end["y"] - min_y) / canvas_height * 100
            marker = ' marker-end="url(#diagram-arrow)"' if line["arrow"] else ""
            html.append(
                f'<line x1="{x1:.3f}%" y1="{y1:.3f}%" '
                f'x2="{x2:.3f}%" y2="{y2:.3f}%"{marker}></line>'
            )
        html.append("</svg>")
    for text, line in zip(paired_connector_labels, connector_lines, strict=False):
        start, end = line["points"]
        label_x = ((start["x"] + end["x"]) / 2 - min_x) / canvas_width * 100
        label_y = ((start["y"] + end["y"]) / 2 - min_y) / canvas_height * 100
        html.append(
            '<div class="diagram-connector-label" '
            f'style="left:{label_x:.3f}%;top:{label_y:.3f}%">'
            f"{_escape_multiline(text)}"
            "</div>"
        )
    for node, bbox in positioned:
        left = (bbox["x"] - min_x) / canvas_width * 100
        top = (bbox["y"] - min_y) / canvas_height * 100
        width = bbox["width"] / canvas_width * 100
        height = bbox["height"] / canvas_height * 100
        text = str(node.get("text", "")).strip()
        html.append(
            '<div class="diagram-positioned-node" '
            f'style="left:{left:.3f}%;top:{top:.3f}%;'
            f'width:{width:.3f}%;height:{height:.3f}%">'
            f"{_escape_multiline(text)}"
            "</div>"
        )
    html.append("</div></section>")
    return "".join(html)


def _split_unpositioned_diagram_labels(
    texts: list[str],
) -> tuple[list[str], list[str]]:
    strip_labels: list[str] = []
    connector_labels: list[str] = []
    for text in texts:
        if _is_diagram_step(text):
            connector_labels.append(text)
            continue
        if (
            connector_labels
            and not _is_diagram_section_heading(text)
            and not _is_diagram_note(text)
        ):
            connector_labels[-1] = f"{connector_labels[-1]}\n{text}"
            continue
        strip_labels.append(text)
    return strip_labels, connector_labels


def _node_bbox(node: dict[str, Any]) -> dict[str, float] | None:
    bbox = node.get("bbox")
    if not isinstance(bbox, dict):
        return None
    try:
        x = float(bbox["x"])
        y = float(bbox["y"])
        width = float(bbox["width"])
        height = float(bbox["height"])
    except (KeyError, TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return {"x": x, "y": y, "width": width, "height": height}


def _connector_points(
    connector: dict[str, Any],
) -> list[dict[str, float]] | None:
    points = connector.get("points")
    if isinstance(points, list) and len(points) >= 2:
        start = _point_xy(points[0])
        end = _point_xy(points[1])
        if start is not None and end is not None:
            return [start, end]
    bbox = _connector_bbox(connector)
    if bbox is None:
        return None
    x = bbox["x"]
    y = bbox["y"]
    width = bbox["width"]
    height = bbox["height"]
    if width >= max(height, 1.0) * 3:
        y_mid = y + max(height, 1.0) / 2
        return [{"x": x, "y": y_mid}, {"x": x + width, "y": y_mid}]
    if height >= max(width, 1.0) * 3:
        x_mid = x + max(width, 1.0) / 2
        return [{"x": x_mid, "y": y}, {"x": x_mid, "y": y + height}]
    return [{"x": x, "y": y}, {"x": x + width, "y": y + height}]


def _connector_line(connector: dict[str, Any]) -> dict[str, Any] | None:
    points = _connector_points(connector)
    if points is None:
        return None
    return {"points": points, "arrow": bool(connector.get("arrow"))}


def _connector_bbox(connector: dict[str, Any]) -> dict[str, float] | None:
    bbox = connector.get("bbox")
    if not isinstance(bbox, dict):
        return None
    try:
        x = float(bbox["x"])
        y = float(bbox["y"])
        width = float(bbox["width"])
        height = float(bbox["height"])
    except (KeyError, TypeError, ValueError):
        return None
    if width <= 0 and height <= 0:
        return None
    return {"x": x, "y": y, "width": max(width, 0.0), "height": max(height, 0.0)}


def _point_xy(point: Any) -> dict[str, float] | None:
    if not isinstance(point, dict):
        return None
    try:
        return {"x": float(point["x"]), "y": float(point["y"])}
    except (KeyError, TypeError, ValueError):
        return None


def _render_label_flowchart_diagram(nodes: list[dict[str, Any]]) -> str:
    texts = [
        text
        for text in (
            str(node.get("text", "")).strip()
            for node in nodes
        )
        if text
    ]
    title, sections = _label_diagram_outline(texts)
    html = ['<section class="diagram-evidence diagram-flowchart">']
    html.append('<div class="diagram-flowchart-page">')
    if title:
        html.append(
            '<div class="diagram-flowchart-title">'
            f"{_escape_multiline(' '.join(title))}"
            "</div>"
        )
    for section in sections:
        heading = str(section.get("title", "")).strip()
        items = section.get("items", [])
        html.append('<section class="diagram-flowchart-section">')
        if heading:
            html.append(
                '<div class="diagram-flowchart-section-title">'
                f"{escape(heading)}"
                "</div>"
            )
        for group in _group_diagram_section_items(
            [item for item in items if isinstance(item, str)]
        ):
            if group["kind"] == "note":
                html.append(
                    '<div class="diagram-flowchart-note">'
                    f"{_escape_multiline(str(group['text']))}"
                    "</div>"
                )
            else:
                html.append(
                    _render_diagram_flowchart_route(
                        [str(step) for step in group.get("steps", [])],
                        [str(actor) for actor in group.get("actors", [])],
                    )
                )
        html.append("</section>")
    if not title and not sections:
        html.append('<p class="empty-diagram">빈 다이어그램</p>')
    html.append("</div></section>")
    return "".join(html)


def _label_diagram_outline(texts: list[str]) -> tuple[list[str], list[dict[str, Any]]]:
    title: list[str] = []
    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for text in texts:
        if _is_diagram_section_heading(text):
            current = {"title": text, "items": []}
            sections.append(current)
            continue
        if current is None:
            title.append(text)
        else:
            current["items"].append(text)
    if not sections and title:
        sections.append({"title": "", "items": title})
        title = []
    return title, sections


def _group_diagram_section_items(items: list[str]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    pending_steps: list[str] = []
    actors: list[str] = []

    def flush_route() -> None:
        nonlocal pending_steps, actors
        if pending_steps or actors:
            groups.append(
                {
                    "kind": "route",
                    "steps": pending_steps,
                    "actors": actors,
                }
            )
        pending_steps = []
        actors = []

    for item in items:
        text = item.strip()
        if not text:
            continue
        if _is_diagram_note(text):
            flush_route()
            groups.append({"kind": "note", "text": text})
            continue
        if _is_diagram_step(text):
            if actors:
                flush_route()
            pending_steps.append(text)
            continue
        if _looks_like_diagram_actor(text):
            actors.append(text)
            continue
        if pending_steps and not actors:
            pending_steps[-1] = f"{pending_steps[-1]}\n{text}"
        else:
            actors.append(text)
    flush_route()
    return groups


def _render_diagram_flowchart_route(steps: list[str], actors: list[str]) -> str:
    html = ['<div class="diagram-flowchart-route">']
    if steps:
        html.append('<div class="diagram-flowchart-steps">')
        for step in steps:
            html.append(
                '<div class="diagram-flowchart-step">'
                f"{_escape_multiline(step)}"
                "</div>"
            )
        html.append("</div>")
    if actors:
        html.append('<div class="diagram-flowchart-box-row">')
        for index, actor in enumerate(actors):
            if index:
                html.append('<span class="diagram-flowchart-arrow">→</span>')
            html.append(
                '<div class="diagram-flowchart-box">'
                f"{_escape_multiline(actor)}"
                "</div>"
            )
        html.append("</div>")
    html.append("</div>")
    return "".join(html)


def _is_diagram_section_heading(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith("<") and stripped.endswith(">")


def _is_diagram_note(text: str) -> bool:
    stripped = text.strip()
    return (
        (stripped.startswith("(") and stripped.endswith(")"))
        or (stripped.startswith("[") and stripped.endswith("]"))
    )


def _is_diagram_step(text: str) -> bool:
    return bool(_DIAGRAM_STEP_RE.match(text.strip()))


def _looks_like_diagram_actor(text: str) -> bool:
    stripped = text.strip()
    if _is_diagram_step(stripped) or _is_diagram_note(stripped):
        return False
    return any(
        token in stripped
        for token in [
            "기관",
            "공단",
            "권자",
            "보장기관",
            "심사평가원",
        ]
    )


def _escape_multiline(text: str) -> str:
    return "<br>".join(escape(part) or "&nbsp;" for part in text.splitlines())


def _render_table_row_cells(
    cells: Any,
    tag: str,
    assets_by_id: dict[str, dict[str, Any]],
    *,
    column_count: int,
) -> str:
    if not isinstance(cells, list):
        return ""
    html = []
    current_column = 1
    sorted_cells = sorted(
        (cell for cell in cells if isinstance(cell, dict)),
        key=lambda cell: _column_id_number(str(cell.get("column_id", "c1"))),
    )
    for cell in sorted_cells:
        column_number = _column_id_number(str(cell.get("column_id", "c1")))
        while current_column < column_number <= column_count:
            html.append(f"<{tag}>&nbsp;</{tag}>")
            current_column += 1
        html.append(_render_table_cell(cell, tag, assets_by_id))
        current_column = max(
            current_column,
            column_number + _positive_int(cell.get("colspan")),
        )
    return "".join(html)


def _render_table_cell(
    cell: dict[str, Any],
    tag: str,
    assets_by_id: dict[str, dict[str, Any]],
) -> str:
    rowspan = _positive_int(cell.get("rowspan"))
    colspan = _positive_int(cell.get("colspan"))
    attrs = []
    if rowspan > 1:
        attrs.append(f'rowspan="{rowspan}"')
    if colspan > 1:
        attrs.append(f'colspan="{colspan}"')
    attrs_text = (" " + " ".join(attrs)) if attrs else ""
    children = _render_children(cell.get("children", []), assets_by_id)
    text = escape(str(cell.get("text", "")))
    if not text and not children:
        text = "&nbsp;"
    return f"<{tag}{attrs_text}>{text}{children}</{tag}>"


def _render_children(
    children: Any,
    assets_by_id: dict[str, dict[str, Any]],
) -> str:
    if not isinstance(children, list) or not children:
        return ""
    rendered = []
    for child in children:
        if isinstance(child, dict):
            rendered.append('<div class="nested-evidence">')
            rendered.append(render_evidence_html(child, assets_by_id))
            rendered.append("</div>")
    return "".join(rendered)


def _render_asset_ref(
    kind: Any,
    content: dict[str, Any],
    assets_by_id: dict[str, dict[str, Any]],
) -> str:
    content = _merge_asset_content(content, assets_by_id)
    uri = str(content.get("uri", ""))
    render_url = str(content.get("public_url") or content.get("preview_url") or uri)
    caption = str(content.get("caption") or content.get("asset_id") or "asset")
    mime = str(content.get("mime", ""))
    meta = " · ".join(
        item
        for item in [
            mime,
            f"{content.get('bytes')} bytes" if content.get("bytes") is not None else "",
            str(content.get("sha256", "")),
        ]
        if item
    )
    if kind == "image":
        if not render_url:
            return (
                "<figure>"
                f"<figcaption>{escape(caption)}</figcaption>"
                f'<p class="asset-meta">{escape(meta)}</p>'
                "</figure>"
            )
        return (
            "<figure>"
            f'<img src="{escape(render_url)}" alt="{escape(caption)}" />'
            f"<figcaption>{escape(caption)}</figcaption>"
            f'<a href="{escape(render_url)}">{escape(uri or render_url)}</a>'
            f'<p class="asset-meta">{escape(meta)}</p>'
            "</figure>"
        )
    return (
        '<div class="asset-ref">'
        f"<strong>{escape(caption)}</strong>"
        f'<a href="{escape(render_url)}">{escape(uri or render_url)}</a>'
        f'<p class="asset-meta">{escape(meta)}</p>'
        "</div>"
    )


def _assets_by_id(assets: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    if not assets:
        return {}
    return {
        str(asset["id"]): asset
        for asset in assets
        if isinstance(asset, dict) and isinstance(asset.get("id"), str)
    }


def _merge_asset_content(
    content: dict[str, Any],
    assets_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    asset_id = content.get("asset_id")
    if not isinstance(asset_id, str):
        return content
    asset = assets_by_id.get(asset_id)
    if asset is None:
        return content
    return {
        **content,
        "uri": content.get("uri") or asset.get("uri"),
        "public_url": content.get("public_url") or asset.get("public_url"),
        "preview_url": content.get("preview_url") or asset.get("preview_url"),
        "mime": content.get("mime") or asset.get("mime"),
        "ext": content.get("ext") or asset.get("ext"),
        "sha256": content.get("sha256") or asset.get("sha256"),
        "bytes": content.get("bytes") if content.get("bytes") is not None else asset.get("bytes"),
    }


def _positive_int(value: Any) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


def _column_id_number(column_id: str) -> int:
    try:
        return max(1, int(column_id.removeprefix("c")))
    except ValueError:
        return 1


_CSS = """
body {
  margin: 0;
  font: 14px/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: #1d1d1f;
  background: #f7f7f5;
}
main {
  max-width: 1100px;
  margin: 0 auto;
  padding: 32px 24px;
}
h1 {
  margin: 0 0 20px;
  font-size: 24px;
}
.document-meta,
.chunk {
  background: #fff;
  border: 1px solid #d8d8d2;
  border-radius: 6px;
  margin: 0 0 16px;
  padding: 16px;
}
.document-meta {
  display: flex;
  gap: 12px;
  align-items: center;
  flex-wrap: wrap;
}
.chunk-header {
  display: flex;
  gap: 8px;
  align-items: center;
  margin-bottom: 8px;
}
.summary {
  margin: 0 0 12px;
  color: #4d4d4d;
}
.source-text {
  margin: 0 0 12px;
  padding: 8px;
  overflow-x: auto;
  background: #f7f7f5;
  border: 1px solid #e0e0da;
  border-radius: 4px;
  color: #4d4d4d;
  white-space: pre-wrap;
}
table {
  border-collapse: collapse;
  width: 100%;
  margin: 8px 0;
  background: #fff;
}
th,
td {
  border: 1px solid #c9c9c2;
  padding: 8px;
  text-align: left;
  vertical-align: top;
}
th {
  background: #ededdf;
}
.diagram-evidence {
  border: 1px solid #d8d8d2;
  border-radius: 4px;
  padding: 12px;
}
.diagram-positioned {
  border: 1px solid #d8d8d2;
  border-radius: 4px;
  padding: 12px;
  background: #fff;
  overflow-x: auto;
}
.diagram-positioned-labels {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  margin: 0 0 10px;
}
.diagram-positioned-label {
  border: 1px solid #d8d8d2;
  background: #f7f7f5;
  padding: 3px 8px;
  font-size: 12px;
  line-height: 1.25;
}
.diagram-canvas {
  position: relative;
  min-width: 680px;
  max-width: 100%;
  min-height: 180px;
  border: 1px solid #c9c9c2;
  background: #fff;
}
.diagram-connectors {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  z-index: 0;
}
.diagram-connectors line {
  stroke: #3f4a54;
  stroke-width: 1.5;
  vector-effect: non-scaling-stroke;
}
.diagram-connectors marker path {
  fill: #3f4a54;
}
.diagram-connector-label {
  position: absolute;
  z-index: 1;
  max-width: 180px;
  transform: translate(-50%, -50%);
  border: 1px solid #c9c9c2;
  background: rgba(255, 255, 255, 0.94);
  padding: 2px 6px;
  text-align: center;
  font-size: 12px;
  line-height: 1.25;
  word-break: keep-all;
}
.diagram-positioned-node {
  position: absolute;
  z-index: 2;
  box-sizing: border-box;
  border: 1.5px solid #1d1d1f;
  background: #fff;
  padding: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  text-align: center;
  line-height: 1.2;
  word-break: keep-all;
}
.diagram-flowchart {
  background: #f7f7f5;
}
.diagram-flowchart-page {
  background: #fff;
  border: 1px solid #c9c9c2;
  padding: 22px;
}
.diagram-flowchart-title {
  margin: 0 0 18px;
  text-align: center;
  font-size: 18px;
  font-weight: 700;
}
.diagram-flowchart-section {
  margin-top: 18px;
}
.diagram-flowchart-section-title {
  margin: 0 0 12px;
  text-align: center;
  font-weight: 700;
}
.diagram-flowchart-route {
  margin: 12px 0;
}
.diagram-flowchart-steps {
  display: flex;
  gap: 6px;
  justify-content: center;
  flex-wrap: wrap;
  margin-bottom: 6px;
}
.diagram-flowchart-step {
  border: 1px solid #b9b9b2;
  border-radius: 999px;
  background: #f7f7f5;
  padding: 3px 10px;
  font-size: 12px;
  text-align: center;
}
.diagram-flowchart-box-row {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  flex-wrap: wrap;
}
.diagram-flowchart-box {
  min-width: 132px;
  min-height: 36px;
  border: 1.5px solid #1d1d1f;
  background: #fff;
  padding: 7px 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  line-height: 1.25;
}
.diagram-flowchart-arrow {
  color: #1d1d1f;
  font-size: 18px;
  line-height: 1;
}
.diagram-flowchart-note {
  max-width: 760px;
  margin: 10px auto 14px;
  color: #4d4d4d;
  font-size: 12px;
  text-align: center;
}
.diagram-nodes,
.diagram-edges {
  margin: 0;
  padding-left: 24px;
}
.diagram-nodes li,
.diagram-edges li {
  margin: 4px 0;
}
.diagram-shape {
  color: #666;
  font-size: 12px;
}
.mermaid {
  margin: 12px 0 0;
  padding: 8px;
  overflow-x: auto;
  background: #f7f7f5;
  border: 1px solid #e0e0da;
  border-radius: 4px;
}
.nested-evidence {
  margin-top: 8px;
}
figure {
  margin: 0;
}
img {
  display: block;
  max-width: 100%;
  min-height: 40px;
  border: 1px solid #d8d8d2;
  background: #fafafa;
}
.asset-meta {
  margin: 4px 0 0;
  color: #666;
  font-size: 12px;
}
code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
@media (max-width: 720px) {
  .diagram-flowchart-page {
    padding: 14px;
  }
  .diagram-flowchart-box {
    min-width: 108px;
    max-width: 100%;
  }
}
"""
