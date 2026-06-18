from __future__ import annotations

import json
from html import escape
from typing import Any

from .evidence_unit_render import _CSS as _EVIDENCE_CSS
from .evidence_unit_render import render_evidence_html


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
    source_text = source.get("text", "") if isinstance(source, dict) else ""
    metadata = chunk.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    evidence = chunk.get("evidence", {})
    if isinstance(evidence, dict):
        evidence_html = render_evidence_html(evidence, assets_by_id)
    else:
        evidence_html = f"<pre>{escape(str(evidence))}</pre>"

    parts = [
        '<section class="chunk rag-chunk">',
        '<header class="chunk-header">',
        f"<code>{escape(str(chunk.get('id', '')))}</code>",
        f"<span>{escape(str(chunk.get('type', '')))}</span>",
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
    parts.append(evidence_html)
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
    unit_types = _string_list(common.get("unit_types")) if isinstance(common, dict) else []
    display_format = common.get("display_format") if isinstance(common, dict) else None

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


def _render_tag_list(values: list[str]) -> str:
    return (
        '<ul class="tag-list">'
        + "".join(f"<li>{escape(value)}</li>" for value in values)
        + "</ul>"
    )


def _as_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value

    to_dict = getattr(value, "to_dict", None)
    if not callable(to_dict):
        return None

    payload = to_dict()
    if isinstance(payload, dict):
        return payload
    return None


def _assets_by_id(assets: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    if not assets:
        return {}
    return {
        str(asset["id"]): asset
        for asset in assets
        if isinstance(asset, dict) and isinstance(asset.get("id"), str)
    }


def _evidence_item_count(evidence: Any) -> int | None:
    if not isinstance(evidence, dict):
        return None
    items = evidence.get("items")
    if not isinstance(items, list):
        return None
    return len(items)


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
    return [item for item in value if isinstance(item, dict)]


_CSS = (
    _EVIDENCE_CSS
    + """
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
"""
)
