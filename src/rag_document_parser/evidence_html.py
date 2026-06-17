from __future__ import annotations

from html import escape
from typing import Any


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
    kind, fmt, content = _evidence_shape(evidence)
    if kind == "composite" and isinstance(content, list):
        return "".join(
            render_evidence_html(item, assets_by_id=assets_by_id)
            for item in content
            if isinstance(item, dict)
        )
    if kind == "table" and fmt == "structured_table" and isinstance(content, dict):
        return _render_structured_table(content, assets_by_id)
    if fmt == "asset_ref" and isinstance(content, dict):
        return _render_asset_ref(kind, content, assets_by_id)
    if isinstance(content, str):
        return f"<p>{escape(content)}</p>"
    return f"<pre>{escape(str(content))}</pre>"


def _evidence_shape(evidence: dict[str, Any]) -> tuple[str, str | None, object]:
    if "items" in evidence and isinstance(evidence["items"], list):
        return "composite", None, evidence["items"]
    kind = evidence.get("type", evidence.get("kind", "text"))
    fmt = evidence.get("format")
    content = evidence.get("content")
    return str(kind), str(fmt) if isinstance(fmt, str) else None, content


def _render_evidence_unit(
    unit: dict[str, Any],
    assets_by_id: dict[str, dict[str, Any]],
) -> str:
    evidence = {
        "type": unit.get("type", "text"),
        "format": unit.get("format"),
        "content": unit.get("content"),
    }
    legacy_evidence = unit.get("evidence")
    has_direct_shape = "content" in unit
    if not has_direct_shape and isinstance(legacy_evidence, dict):
        evidence = legacy_evidence
    source = unit.get("source", {})
    source_text = source.get("text", "") if isinstance(source, dict) else ""
    return (
        '<section class="chunk">'
        '<header class="chunk-header">'
        f"<code>{escape(str(unit.get('id', '')))}</code>"
        f"<span>{escape(str(unit.get('type', '')))}</span>"
        "</header>"
        f'<pre class="source-text">{escape(str(source_text))}</pre>'
        f"{render_evidence_html(evidence, assets_by_id)}"
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
            html.append(_render_table_cells(row.get("cells", []), "th", assets_by_id))
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
        html.append(_render_table_cells(row.get("cells", []), "td", assets_by_id))
        if not row.get("cells") and columns:
            html.append(f'<td colspan="{len(columns)}">&nbsp;</td>')
        html.append("</tr>")
    if not rows and columns:
        html.append(f'<tr><td colspan="{len(columns)}">&nbsp;</td></tr>')
    html.append("</tbody></table>")
    if not columns and not rows:
        return "<p class=\"empty-table\">빈 표</p>"
    return "".join(html)


def _render_table_cells(
    cells: Any,
    tag: str,
    assets_by_id: dict[str, dict[str, Any]],
) -> str:
    html = []
    if not isinstance(cells, list):
        return ""
    for cell in cells:
        if not isinstance(cell, dict):
            continue
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
        html.append(f"<{tag}{attrs_text}>{text}{children}</{tag}>")
    return "".join(html)


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
"""
