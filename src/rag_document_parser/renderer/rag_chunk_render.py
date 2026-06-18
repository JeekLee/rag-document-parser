from __future__ import annotations

import json
from collections.abc import Mapping
from html import escape
from typing import Any


def render_rag_chunks_html(
    chunks: list[Any],
    *,
    title: str = "RAG chunks",
    assets: list[dict[str, Any]] | None = None,
) -> str:
    """Render final RagChunk objects for manual chunk quality inspection."""
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
    for chunk in chunks:
        chunk_dict = _as_dict(chunk)
        if chunk_dict is not None:
            parts.append(_render_rag_chunk(chunk_dict, assets_by_id))
    parts.extend(["</main>", "</body>", "</html>"])
    return "\n".join(parts)


def _render_rag_chunk(
    chunk: dict[str, Any],
    assets_by_id: dict[str, dict[str, Any]],
) -> str:
    source = chunk.get("source", {})
    source_text = source.get("text", "") if isinstance(source, Mapping) else ""
    metadata = chunk.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}

    evidence = chunk.get("evidence", {})

    parts = [
        '<section class="chunk rag-chunk">',
        '<header class="chunk-header">',
        f"<code>{escape(str(chunk.get('id', '')))}</code>",
        "</header>",
    ]
    parts.append(_render_title(metadata))

    summary = chunk.get("summary")
    if isinstance(summary, str) and summary:
        parts.append(f'<p class="summary">{escape(summary)}</p>')

    parts.append(_render_chunk_lists(chunk))
    parts.append(_render_chunk_metadata(metadata, evidence))
    parts.append(_render_chunk_review(metadata))
    parts.append(_render_chunk_diagnostics(metadata))
    parts.append('<h2 class="chunk-section-title">source text</h2>')
    parts.append(f'<pre class="source-text">{escape(str(source_text))}</pre>')
    parts.append('<h2 class="chunk-section-title">final evidence</h2>')
    parts.append(_render_final_evidence(evidence, assets_by_id))
    parts.append("</section>")
    return "".join(parts)


def _render_title(metadata: dict[str, Any]) -> str:
    title = metadata.get("title")
    if not isinstance(title, str) or not title:
        return ""
    return f'<h2 class="rag-chunk-title">{escape(title)}</h2>'


def _render_chunk_lists(chunk: dict[str, Any]) -> str:
    keywords = _string_list(chunk.get("keywords"))
    questions = _string_list(chunk.get("questions"))
    if not keywords and not questions:
        return ""

    parts = ['<div class="chunk-fields">']
    if keywords:
        parts.append("<div><strong>keywords</strong>")
        parts.append(_render_tag_list(keywords))
        parts.append("</div>")
    if questions:
        parts.append("<div><strong>questions</strong>")
        parts.append("<ul>")
        for question in questions:
            parts.append(f"<li>{escape(question)}</li>")
        parts.append("</ul></div>")
    parts.append("</div>")
    return "".join(parts)


def _render_chunk_metadata(metadata: dict[str, Any], evidence: Any) -> str:
    labels = []
    source_unit_ids = _string_list(metadata.get("source_unit_ids"))
    context_unit_ids = _string_list(metadata.get("context_unit_ids"))
    evidence_item_count = _evidence_item_count(evidence)
    common = metadata.get("common")
    unit_types = (
        _string_list(common.get("unit_types"))
        if isinstance(common, Mapping)
        else _evidence_item_types(evidence)
    )
    display_format = common.get("display_format") if isinstance(common, Mapping) else None

    if source_unit_ids:
        labels.append(f"source units: {', '.join(source_unit_ids)}")
    if context_unit_ids:
        labels.append(f"context units: {', '.join(context_unit_ids)}")
    if evidence_item_count is not None:
        labels.append(f"evidence items: {evidence_item_count}")
    if unit_types:
        labels.append(f"unit types: {', '.join(unit_types)}")
    if isinstance(display_format, str) and display_format:
        labels.append(f"display: {display_format}")
    if not labels:
        return ""

    return (
        '<div class="chunk-meta">'
        + "".join(f"<span>{escape(label)}</span>" for label in labels)
        + "</div>"
    )


def _render_chunk_review(metadata: dict[str, Any]) -> str:
    parts = []
    source_units = _dicts(metadata.get("source_units"))
    if source_units:
        parts.append('<section class="chunk-review-block">')
        parts.append('<h2 class="chunk-section-title">source units</h2>')
        parts.append('<ol class="source-unit-list">')
        for unit in source_units:
            unit_id = str(unit.get("id", ""))
            unit_type = str(unit.get("type", ""))
            unit_format = str(unit.get("format", ""))
            meta = " / ".join(
                escape(value)
                for value in [unit_type, unit_format]
                if value
            )
            parts.append("<li>")
            parts.append(f"<code>{escape(unit_id)}</code>")
            if meta:
                parts.append(f"<span>{meta}</span>")
            parts.append("</li>")
        parts.append("</ol>")
        parts.append("</section>")

    operations = metadata.get("operations")
    if operations is not None:
        parts.append(_render_debug_details("operations", operations, open_details=True))

    boundary_merges = metadata.get("_boundary_merges")
    if boundary_merges is not None:
        parts.append(_render_debug_details("boundary merges", boundary_merges, open_details=True))

    other_metadata = _other_metadata(metadata)
    if other_metadata:
        parts.append(_render_debug_details("other metadata", other_metadata))

    return "".join(parts)


def _render_chunk_diagnostics(metadata: dict[str, Any]) -> str:
    parts = []
    fallback_reason = metadata.get("_fallback_reason")
    if isinstance(fallback_reason, str) and fallback_reason:
        parts.append(
            '<div class="diagnostic diagnostic-error">'
            f"<strong>fallback</strong>: {escape(fallback_reason)}"
            "</div>"
        )

    warnings = metadata.get("_warnings")
    if warnings:
        parts.append(_render_debug_details("warnings", warnings, open_details=True))

    rejected_plan = metadata.get("_rejected_plan")
    if rejected_plan is not None:
        parts.append(_render_debug_details("rejected plan", rejected_plan, open_details=True))

    return "".join(parts)


def _render_debug_details(label: str, value: Any, *, open_details: bool = False) -> str:
    open_attr = " open" if open_details else ""
    return (
        f'<details class="diagnostic"{open_attr}>'
        f"<summary>{escape(label)}</summary>"
        f"<pre>{_json_debug(value)}</pre>"
        "</details>"
    )


def _render_final_evidence(
    evidence: Any,
    assets_by_id: dict[str, dict[str, Any]],
) -> str:
    if not isinstance(evidence, Mapping):
        return f"<pre>{escape(str(evidence))}</pre>"

    items = evidence.get("items")
    if not isinstance(items, list):
        return _render_evidence_item(evidence, assets_by_id)

    item_dicts = [item for item in items if isinstance(item, Mapping)]
    parts = ['<div class="final-evidence">']
    for item in item_dicts:
        parts.append('<div class="final-evidence-part">')
        parts.append(_render_evidence_item(item, assets_by_id))
        parts.append("</div>")
    parts.append("</div>")

    if item_dicts:
        parts.append('<div class="evidence-item-details">')
        for index, item in enumerate(item_dicts, start=1):
            parts.append(_render_evidence_item_detail(index, item, assets_by_id))
        parts.append("</div>")
    return "".join(parts)


def _render_evidence_item_detail(
    index: int,
    item: dict[str, Any],
    assets_by_id: dict[str, dict[str, Any]],
) -> str:
    labels = [f"evidence item {index}"]
    item_type = item.get("type", item.get("kind"))
    item_format = item.get("format")
    if isinstance(item_type, str) and item_type:
        labels.append(item_type)
    if isinstance(item_format, str) and item_format:
        labels.append(item_format)

    source_unit_ids = _string_list(item.get("source_unit_ids"))
    parts = [
        '<details class="evidence-item-detail">',
        "<summary>",
        " / ".join(escape(label) for label in labels),
        "</summary>",
    ]
    if source_unit_ids:
        parts.append(
            '<div class="chunk-meta">'
            f"<span>item source units: {escape(', '.join(source_unit_ids))}</span>"
            "</div>"
        )
    parts.append(_render_evidence_item(item, assets_by_id))
    parts.append("</details>")
    return "".join(parts)


def _render_evidence_item(
    item: Any,
    assets_by_id: dict[str, dict[str, Any]],
) -> str:
    if not isinstance(item, Mapping):
        return f"<pre>{escape(str(item))}</pre>"

    item_type = item.get("type", item.get("kind"))
    item_format = item.get("format")
    content = item.get("content")

    if item_type == "text" and isinstance(content, str):
        return _render_text_content(content)
    if item_type == "table" and item_format == "structured_table" and isinstance(content, Mapping):
        return _render_structured_table(content, assets_by_id)
    if item_type in {"image", "asset"} and isinstance(content, Mapping):
        return _render_asset_ref(content, assets_by_id)
    if item_type == "diagram" and isinstance(content, Mapping):
        return _render_diagram(content)
    if isinstance(content, (dict, list)):
        return f'<pre class="evidence-json">{_json_debug(content)}</pre>'
    if content is not None:
        return _render_text_content(str(content))
    return f'<pre class="evidence-json">{_json_debug(item)}</pre>'


def _render_text_content(text: str) -> str:
    blocks = [block for block in text.split("\n\n") if block.strip()]
    if not blocks:
        return '<p class="empty-text">&nbsp;</p>'
    return "".join(f"<p>{_escape_multiline(block.strip())}</p>" for block in blocks)


def _render_structured_table(
    table: dict[str, Any],
    assets_by_id: dict[str, dict[str, Any]],
) -> str:
    columns = _dicts(table.get("columns"))
    header_rows = _dicts(table.get("header_rows"))
    rows = _dicts(table.get("rows"))
    if not columns and not rows:
        return '<p class="empty-table">empty table</p>'

    header_rows, rows = _clip_overlapping_table_rowspans(header_rows, rows)
    parts = ['<div class="rag-table-wrap">', '<table class="evidence-table">']
    visible_header_rows = [row for row in header_rows if _table_row_has_content(row)]
    if visible_header_rows:
        parts.append("<thead>")
        parts.append(
            _render_table_rows(
                visible_header_rows,
                "th",
                assets_by_id,
                column_count=len(columns),
            )
        )
        parts.append("</thead>")
    elif columns:
        parts.append("<thead><tr>")
        for column in columns:
            parts.append(f"<th>{escape(str(column.get('text', ''))) or '&nbsp;'}</th>")
        parts.append("</tr></thead>")

    parts.append("<tbody>")
    parts.append(
        _render_table_rows(
            rows,
            "td",
            assets_by_id,
            column_count=len(columns),
        )
    )
    if not rows and columns:
        parts.append(f'<tr><td colspan="{len(columns)}">&nbsp;</td></tr>')
    parts.append("</tbody></table></div>")
    return "".join(parts)


def _clip_overlapping_table_rowspans(
    header_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return (
        _clip_overlapping_rowspans(header_rows),
        _clip_overlapping_rowspans(rows),
    )


def _clip_overlapping_rowspans(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_rows: list[dict[str, Any]] = []
    active_spans: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        cells = [dict(cell) for cell in _dicts(row.get("cells"))]
        normalized_rows.append({**row, "cells": cells})

        for cell in cells:
            cell_start = _column_id_number(str(cell.get("column_id", "c1")))
            cell_end = cell_start + _positive_int(cell.get("colspan"))
            for span in active_spans:
                if span["end"] <= cell_start or cell_end <= span["start"]:
                    continue
                max_rowspan = row_index - span["row_index"]
                if max_rowspan <= 0:
                    continue
                span_cell = span["cell"]
                if max_rowspan < _positive_int(span_cell.get("rowspan")):
                    span_cell["rowspan"] = max_rowspan

        active_spans = [
            span
            for span in active_spans
            if span["row_index"] + _positive_int(span["cell"].get("rowspan"))
            > row_index + 1
        ]
        for cell in cells:
            rowspan = _positive_int(cell.get("rowspan"))
            if rowspan <= 1:
                continue
            cell_start = _column_id_number(str(cell.get("column_id", "c1")))
            active_spans.append(
                {
                    "row_index": row_index,
                    "start": cell_start,
                    "end": cell_start + _positive_int(cell.get("colspan")),
                    "cell": cell,
                }
            )
    return normalized_rows


def _render_table_rows(
    rows: list[dict[str, Any]],
    tag: str,
    assets_by_id: dict[str, dict[str, Any]],
    *,
    column_count: int,
) -> str:
    html: list[str] = []
    active_rowspans: dict[int, int] = {}
    for row in rows:
        cells = row.get("cells", [])
        occupied = {
            column_number
            for column_number, remaining in active_rowspans.items()
            if remaining > 0
        }
        positioned = _position_table_row_cells(
            cells,
            column_count=column_count,
            occupied_columns=occupied,
        )
        html.append("<tr>")
        html.extend(
            f"<{tag}>&nbsp;</{tag}>"
            if cell is None
            else _render_table_cell(cell, tag, assets_by_id)
            for cell, _column_number in positioned
        )
        if not positioned and column_count:
            html.append(
                _render_empty_table_row_cells(
                    tag,
                    column_count=column_count,
                    active_rowspans=active_rowspans,
                )
            )
        html.append("</tr>")

        for column_number in list(active_rowspans):
            active_rowspans[column_number] -= 1
            if active_rowspans[column_number] <= 0:
                del active_rowspans[column_number]

        for cell, column_number in positioned:
            if cell is None:
                continue
            rowspan = _positive_int(cell.get("rowspan"))
            colspan = _positive_int(cell.get("colspan"))
            if rowspan <= 1:
                continue
            for offset in range(colspan):
                active_rowspans[column_number + offset] = max(
                    active_rowspans.get(column_number + offset, 0),
                    rowspan - 1,
                )
    return "".join(html)


def _position_table_row_cells(
    cells: Any,
    *,
    column_count: int,
    occupied_columns: set[int],
) -> list[tuple[dict[str, Any] | None, int]]:
    if not isinstance(cells, list):
        return []
    positioned: list[tuple[dict[str, Any] | None, int]] = []
    current_column = 1
    sorted_cells = sorted(
        (cell for cell in cells if isinstance(cell, Mapping)),
        key=lambda cell: _column_id_number(str(cell.get("column_id", "c1"))),
    )
    for cell in sorted_cells:
        column_number = _column_id_number(str(cell.get("column_id", "c1")))
        while current_column in occupied_columns:
            current_column += 1
        while current_column < column_number <= column_count:
            if current_column not in occupied_columns:
                positioned.append((None, current_column))
            current_column += 1
            while current_column in occupied_columns:
                current_column += 1
        if current_column > column_number:
            column_number = current_column
        positioned.append((cell, column_number))
        current_column = max(
            current_column,
            column_number + _positive_int(cell.get("colspan")),
        )
    return positioned


def _render_empty_table_row_cells(
    tag: str,
    *,
    column_count: int,
    active_rowspans: dict[int, int],
) -> str:
    html = []
    column = 1
    while column <= column_count:
        if column in active_rowspans:
            column += 1
            continue
        start = column
        while column <= column_count and column not in active_rowspans:
            column += 1
        width = column - start
        if width == 1:
            html.append(f"<{tag}>&nbsp;</{tag}>")
        elif width > 1:
            html.append(f'<{tag} colspan="{width}">&nbsp;</{tag}>')
    return "".join(html)


def _render_table_cell(
    cell: dict[str, Any],
    tag: str,
    assets_by_id: dict[str, dict[str, Any]],
) -> str:
    attrs = []
    rowspan = _positive_int(cell.get("rowspan"))
    colspan = _positive_int(cell.get("colspan"))
    if rowspan > 1:
        attrs.append(f'rowspan="{rowspan}"')
    if colspan > 1:
        attrs.append(f'colspan="{colspan}"')
    attrs_text = (" " + " ".join(attrs)) if attrs else ""
    text = _escape_multiline(_cell_text(cell))
    children = _render_children(cell.get("children"), assets_by_id)
    if not text and not children:
        text = "&nbsp;"
    return f"<{tag}{attrs_text}>{text}{children}</{tag}>"


def _render_children(children: Any, assets_by_id: dict[str, dict[str, Any]]) -> str:
    if not isinstance(children, list) or not children:
        return ""
    rendered = []
    for child in children:
        if isinstance(child, Mapping):
            rendered.append('<div class="nested-evidence">')
            rendered.append(_render_evidence_item(child, assets_by_id))
            rendered.append("</div>")
    return "".join(rendered)


def _render_asset_ref(
    content: dict[str, Any],
    assets_by_id: dict[str, dict[str, Any]],
) -> str:
    merged = dict(content)
    asset_id = merged.get("asset_id")
    if isinstance(asset_id, str) and asset_id in assets_by_id:
        merged = {**assets_by_id[asset_id], **merged}
    url = str(merged.get("public_url") or merged.get("preview_url") or merged.get("uri") or "")
    caption = str(merged.get("caption") or merged.get("asset_id") or "asset")
    if url:
        return (
            "<figure>"
            f'<img src="{escape(url, quote=True)}" alt="{escape(caption, quote=True)}">'
            f"<figcaption>{escape(caption)}</figcaption>"
            "</figure>"
        )
    return (
        "<figure>"
        f"<figcaption>{escape(caption)}</figcaption>"
        f'<pre class="evidence-json">{_json_debug(merged)}</pre>'
        "</figure>"
    )


def _render_diagram(content: dict[str, Any]) -> str:
    nodes = _dicts(content.get("nodes"))
    edges = _dicts(content.get("edges"))
    parts = ['<section class="rag-diagram">']
    caption = content.get("caption")
    if isinstance(caption, str) and caption:
        parts.append(f"<h3>{escape(caption)}</h3>")
    if nodes:
        parts.append("<h4>nodes</h4><ul>")
        for node in nodes:
            label = str(node.get("label") or node.get("text") or node.get("id") or "")
            parts.append(f"<li>{escape(label)}</li>")
        parts.append("</ul>")
    if edges:
        parts.append("<h4>edges</h4><ul>")
        for edge in edges:
            label = str(edge.get("label") or edge.get("text") or "")
            source = str(edge.get("source") or edge.get("from") or "")
            target = str(edge.get("target") or edge.get("to") or "")
            parts.append(f"<li>{escape(source)} -> {escape(target)} {escape(label)}</li>")
        parts.append("</ul>")
    if not nodes and not edges:
        parts.append(f'<pre class="evidence-json">{_json_debug(content)}</pre>')
    parts.append("</section>")
    return "".join(parts)


def _table_row_has_content(row: dict[str, Any]) -> bool:
    return any(
        _cell_text(cell).strip() or _dicts(cell.get("children"))
        for cell in _dicts(row.get("cells"))
    )


def _cell_text(cell: dict[str, Any]) -> str:
    return str(cell.get("text", ""))


def _column_id_number(column_id: str) -> int:
    if column_id.startswith("c"):
        suffix = column_id[1:]
        if suffix.isdigit():
            return max(1, int(suffix))
    return 1


def _positive_int(value: Any) -> int:
    return value if type(value) is int and value > 0 else 1


def _escape_multiline(text: str) -> str:
    return "<br>".join(escape(line) for line in text.splitlines())


def _render_tag_list(values: list[str]) -> str:
    return (
        '<ul class="tag-list">'
        + "".join(f"<li>{escape(value)}</li>" for value in values)
        + "</ul>"
    )


def _as_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return value

    to_dict = getattr(value, "to_dict", None)
    if not callable(to_dict):
        return None

    payload = to_dict()
    if isinstance(payload, Mapping):
        return payload
    return None


def _assets_by_id(assets: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    if not assets:
        return {}
    return {
        str(asset["id"]): asset
        for asset in assets
        if isinstance(asset, Mapping) and isinstance(asset.get("id"), str)
    }


def _evidence_item_count(evidence: Any) -> int | None:
    if not isinstance(evidence, Mapping):
        return None
    items = evidence.get("items")
    if not isinstance(items, list):
        return None
    return len(items)


def _evidence_item_types(evidence: Any) -> list[str]:
    if not isinstance(evidence, Mapping):
        return []
    items = evidence.get("items")
    if not isinstance(items, list):
        return []
    result: list[str] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        item_type = item.get("type")
        if isinstance(item_type, str) and item_type and item_type not in result:
            result.append(item_type)
    return result


def _other_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    hidden_keys = {
        "source_unit_ids",
        "source_units",
        "context_unit_ids",
        "operations",
        "title",
        "common",
        "_boundary_merges",
        "_fallback_reason",
        "_rejected_plan",
        "_warnings",
    }
    return {
        key: value
        for key, value in metadata.items()
        if key not in hidden_keys
    }


def _json_debug(value: Any) -> str:
    return escape(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


_CSS = """
* {
  box-sizing: border-box;
}
body {
  margin: 0;
  background: #f4f4f1;
  color: #1f2428;
  font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
main {
  max-width: 1200px;
  margin: 0 auto;
  padding: 24px;
}
h1 {
  margin: 0 0 18px;
  font-size: 24px;
}
.chunk {
  margin: 0 0 18px;
  border: 1px solid #d8d8d2;
  border-radius: 6px;
  background: #fff;
  padding: 16px;
}
.chunk-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 0 0 10px;
}
code {
  border: 1px solid #d8d8d2;
  border-radius: 4px;
  background: #f7f7f5;
  padding: 2px 6px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
}
p {
  margin: 0 0 8px;
}
pre {
  margin: 0 0 12px;
  white-space: pre-wrap;
  overflow-x: auto;
  border: 1px solid #e0e0da;
  border-radius: 4px;
  background: #fbfbf9;
  padding: 10px;
}
figure {
  margin: 0 0 12px;
}
figure img {
  max-width: 100%;
  height: auto;
  border: 1px solid #d8d8d2;
  border-radius: 4px;
}
figcaption {
  margin-top: 4px;
  color: #666;
  font-size: 12px;
}
table {
  border-collapse: collapse;
  background: #fff;
}
th,
td {
  border: 1px solid #c9c9c2;
  padding: 6px 8px;
  text-align: left;
  vertical-align: top;
}
th {
  background: #ededdf;
  font-weight: 600;
}
.rag-table-wrap {
  max-width: 100%;
  overflow-x: auto;
  margin: 0 0 12px;
}
.evidence-table {
  width: max-content;
  min-width: min(100%, 720px);
}
.evidence-table th,
.evidence-table td {
  min-width: 80px;
  max-width: 280px;
}
.nested-evidence {
  margin: 6px 0 0;
  padding: 8px;
  border: 1px solid #e0e0da;
  background: #fafafa;
}
.empty-table,
.empty-text {
  color: #777;
}
.evidence-json {
  font-size: 12px;
}
.rag-chunk-title {
  margin: 0 0 10px;
  font-size: 18px;
}
.chunk-section-title {
  margin: 14px 0 6px;
  font-size: 13px;
  text-transform: uppercase;
  letter-spacing: 0;
  color: #555;
}
.chunk-review-block {
  margin: 0 0 12px;
}
.source-unit-list {
  display: grid;
  gap: 4px;
  margin: 0 0 12px;
  padding-left: 24px;
}
.source-unit-list li {
  padding: 3px 0;
}
.source-unit-list code {
  margin-right: 6px;
}
.source-unit-list span {
  color: #555;
}
.final-evidence {
  display: grid;
  gap: 10px;
  margin: 0 0 12px;
}
.final-evidence-part {
  min-width: 0;
}
.final-evidence-part > :first-child {
  margin-top: 0;
}
.final-evidence-part > :last-child {
  margin-bottom: 0;
}
.evidence-item-details {
  display: grid;
  gap: 8px;
  margin: 0 0 12px;
}
.evidence-item-detail {
  border: 1px solid #e0e0da;
  border-radius: 4px;
  padding: 8px;
  background: #fafafa;
}
.evidence-item-detail summary {
  cursor: pointer;
  font-weight: 600;
}
.evidence-item-detail > :last-child {
  margin-bottom: 0;
}
.chunk-fields {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 12px;
  margin: 0 0 12px;
}
.tag-list {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin: 6px 0 0;
  padding: 0;
  list-style: none;
}
.tag-list li,
.chunk-meta span {
  display: inline-flex;
  align-items: center;
  border: 1px solid #d8d8d2;
  border-radius: 4px;
  background: #f7f7f5;
  padding: 2px 6px;
  font-size: 12px;
}
.chunk-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin: 0 0 12px;
}
.summary {
  margin: 0 0 12px;
  font-weight: 500;
}
.source-text {
  max-height: 320px;
}
.diagnostic {
  margin: 0 0 12px;
  border: 1px solid #e0e0da;
  border-radius: 4px;
  background: #fbfbf9;
  padding: 8px;
}
.diagnostic summary {
  cursor: pointer;
  font-weight: 600;
}
.diagnostic-error {
  border-color: #d38a8a;
  background: #fff5f5;
}
.rag-diagram h3,
.rag-diagram h4 {
  margin: 10px 0 6px;
  font-size: 13px;
}
.rag-diagram {
  border: 1px solid #d8d8d2;
  border-radius: 4px;
  padding: 10px;
  background: #fcfcfa;
}
"""
