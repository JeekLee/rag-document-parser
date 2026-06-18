# EvidenceUnit Agentic Chunking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a format-independent agentic chunking stage that converts extracted `EvidenceUnit` objects into enriched `RagChunk` objects with composite final evidence.

**Architecture:** First clean up the evidence model so extraction units directly carry `type`, `format`, and `content`, while `RagChunk.evidence` carries ordered `EvidenceItem` records. Then add a local `EvidenceUnitAgenticChunker` that asks an LLM for a chunk plan, validates that plan, and materializes chunk evidence only from original unit content. `RagDocumentParser.parse()` remains extraction-only.

**Tech Stack:** Python 3.11+ dataclasses, stdlib `urllib` LLM client already in `rag_document_parser.enrichment.llm`, pytest, uv.

---

## File Structure

- Modify `src/rag_document_parser/models.py`
  - Add `EvidenceItem`.
  - Change `Evidence` to a composite container with `items`.
  - Change `EvidenceUnit` to direct `type`, `format`, `source`, `content`, `metadata`.
  - Keep `RagChunk` but point `evidence` at the new composite `Evidence`.

- Modify `src/rag_document_parser/extract/assets.py`
  - Resolve `asset_ref` payloads through `EvidenceUnit.content` instead of `EvidenceUnit.evidence`.
  - Keep nested evidence resolution for structured table cell children by detecting `{type|kind, format, content}` dictionaries.

- Modify extraction backends:
  - `src/rag_document_parser/extract/formats/markdown/backend.py`
  - `src/rag_document_parser/extract/formats/hwpx/backend.py`
  - `src/rag_document_parser/extract/formats/hwp5/backend.py`
  - `src/rag_document_parser/extract/formats/pdf/backend.py`
  - Replace nested `Evidence(kind="text", format="plain", content=text)` construction with direct `EvidenceUnit(format="plain", content=text)`.

- Modify compatibility exports:
  - `src/rag_document_parser/__init__.py`
  - `src/rag_document_parser/backends.py`
  - Export `EvidenceItem`.
  - Keep `Evidence` exported as the composite chunk evidence model.

- Modify `src/rag_document_parser/evidence_html.py`
  - Render extraction units shaped as `{type, format, content}`.
  - Render chunk evidence shaped as `{"items": [{"type": "text", "format": "plain", "content": "example"}]}`.

- Create `src/rag_document_parser/chunk/agentic.py`
  - Public `EvidenceUnitAgenticChunker`.
  - Internal prompt payload, windowing, plan validation, fallback, and materialization helpers.

- Modify `src/rag_document_parser/chunk/__init__.py`
  - Export `EvidenceUnitAgenticChunker`.

- Modify tests:
  - `tests/test_models.py`
  - `tests/test_parser.py`
  - `tests/test_hwpx_backend.py`
  - `tests/test_hwp5_backend.py`
  - `tests/test_pdf_backend.py`
  - `tests/test_evidence_html.py`
  - `tests/test_pipeline_layout.py`
  - `tests/test_agentic_chunker.py`

- Modify `README.md`
  - Document extraction units vs final chunk evidence.
  - Document explicit chunking usage.

## Task 1: Evidence Model Cleanup

**Files:**
- Create: `tests/test_models.py`
- Modify: `src/rag_document_parser/models.py`
- Modify: `src/rag_document_parser/__init__.py`

- [ ] **Step 1: Write failing model serialization tests**

Create `tests/test_models.py` with:

```python
from __future__ import annotations


def test_evidence_unit_carries_direct_format_and_content():
    from rag_document_parser import EvidenceUnit, SourceEvidence

    unit = EvidenceUnit(
        id="b1",
        type="text",
        format="plain",
        source=SourceEvidence(kind="text", text="source text"),
        content="display text",
        metadata={"common": {"chunk_kind": "text"}},
    )

    assert unit.to_dict() == {
        "id": "b1",
        "type": "text",
        "format": "plain",
        "source": {"kind": "text", "text": "source text"},
        "content": "display text",
        "metadata": {"common": {"chunk_kind": "text"}},
    }


def test_chunk_evidence_is_composite_items_only():
    from rag_document_parser import Evidence, EvidenceItem

    evidence = Evidence(
        items=[
            EvidenceItem(
                type="text",
                format="plain",
                content="display text",
                source_unit_ids=["b1"],
                metadata={"page": 1},
            )
        ]
    )

    assert evidence.to_dict() == {
        "items": [
            {
                "type": "text",
                "format": "plain",
                "content": "display text",
                "source_unit_ids": ["b1"],
                "metadata": {"page": 1},
            }
        ]
    }
```

- [ ] **Step 2: Run model tests to verify RED**

Run:

```bash
uv run --extra dev pytest tests/test_models.py -q
```

Expected: FAIL because `EvidenceItem` does not exist and `EvidenceUnit` still expects nested `evidence`.

- [ ] **Step 3: Implement the new dataclasses**

In `src/rag_document_parser/models.py`, replace the old `Evidence`, `EvidenceUnit`, and `RagChunk` definitions with this shape while leaving existing `SourceInfo`, `PendingAsset`, `DocumentAsset`, and `SourceEvidence` intact:

```python
@dataclass(frozen=True)
class EvidenceItem:
    type: str
    content: Any
    format: str | None = None
    source_unit_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": self.type,
            "content": self.content,
            "source_unit_ids": list(self.source_unit_ids),
            "metadata": dict(self.metadata),
        }
        if self.format is not None:
            payload["format"] = self.format
        return payload


@dataclass(frozen=True)
class Evidence:
    items: list[EvidenceItem] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"items": [item.to_dict() for item in self.items]}


@dataclass(frozen=True)
class EvidenceUnit:
    id: str
    type: str
    format: str
    source: SourceEvidence
    content: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "format": self.format,
            "source": self.source.to_dict(),
            "content": self.content,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RagChunk:
    id: str
    type: str
    source: SourceEvidence
    evidence: Evidence
    summary: str
    keywords: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "source": self.source.to_dict(),
            "evidence": self.evidence.to_dict(),
            "summary": self.summary,
            "keywords": list(self.keywords),
            "questions": list(self.questions),
            "metadata": dict(self.metadata),
        }
```

In `src/rag_document_parser/__init__.py`, add `EvidenceItem` to the model import list and `__all__`.

- [ ] **Step 4: Run model tests to verify GREEN**

Run:

```bash
uv run --extra dev pytest tests/test_models.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/rag_document_parser/models.py src/rag_document_parser/__init__.py tests/test_models.py
git commit -m "refactor: split extraction units from chunk evidence"
```

## Task 2: Migrate Extraction Units and Asset Resolution

**Files:**
- Modify: `src/rag_document_parser/extract/assets.py`
- Modify: `src/rag_document_parser/extract/formats/markdown/backend.py`
- Modify: `src/rag_document_parser/extract/formats/hwpx/backend.py`
- Modify: `src/rag_document_parser/extract/formats/hwp5/backend.py`
- Modify: `src/rag_document_parser/extract/formats/pdf/backend.py`
- Modify: `src/rag_document_parser/backends.py`
- Modify: `tests/test_parser.py`
- Modify: `tests/test_hwpx_backend.py`
- Modify: `tests/test_hwp5_backend.py`
- Modify: `tests/test_pdf_backend.py`
- Modify: `tests/test_regression_corpus.py`
- Modify: `tests/test_validate_hwpx_clic_minio.py`

- [ ] **Step 1: Update parser tests to expect direct unit content**

In parser/backend tests, replace assertions like:

```python
assert text_unit.evidence.kind == "text"
assert text_unit.evidence.format == "plain"
assert text_unit.evidence.content == "Plain paragraph."
```

with:

```python
assert text_unit.type == "text"
assert text_unit.format == "plain"
assert text_unit.content == "Plain paragraph."
```

For `ParseResult.to_dict()` assertions, replace:

```python
assert payload["units"][0]["evidence"] == {
    "kind": "text",
    "format": "plain",
    "content": "plain paragraph",
}
```

with:

```python
assert payload["units"][0]["format"] == "plain"
assert payload["units"][0]["content"] == "plain paragraph"
assert "evidence" not in payload["units"][0]
```

For custom test backends, construct units as:

```python
EvidenceUnit(
    id="c1",
    type="text",
    format="plain",
    source=SourceEvidence(kind="text", text="custom text"),
    content="custom text",
)
```

- [ ] **Step 2: Run migrated extraction tests to verify RED**

Run:

```bash
uv run --extra dev pytest tests/test_parser.py tests/test_hwpx_backend.py tests/test_hwp5_backend.py tests/test_pdf_backend.py tests/test_regression_corpus.py -q
```

Expected: FAIL because extraction backends still pass calls such as `evidence=Evidence(kind="text", format="plain", content=text)`.

- [ ] **Step 3: Update asset resolution**

In `src/rag_document_parser/extract/assets.py`, make `resolve_units()` rebuild units with direct `format` and `content`:

```python
def resolve_units(
    units: list[EvidenceUnit],
    assets: list[DocumentAsset],
) -> list[EvidenceUnit]:
    assets_by_id = {asset.id: asset for asset in assets}
    resolved: list[EvidenceUnit] = []
    for unit in units:
        resolved.append(
            EvidenceUnit(
                id=unit.id,
                type=unit.type,
                format=unit.format,
                source=unit.source,
                content=resolve_asset_content(unit.format, unit.content, assets_by_id),
                metadata=dict(unit.metadata),
            )
        )
    return resolved
```

Replace `resolve_asset_evidence()` with direct content resolution:

```python
def resolve_asset_content(
    fmt: str,
    content: Any,
    assets_by_id: dict[str, DocumentAsset],
) -> Any:
    if fmt != "asset_ref":
        return resolve_asset_refs_in_value(content, assets_by_id)
    if not isinstance(content, dict):
        raise ValueError("asset_ref content must be an object")
    asset_id = content.get("asset_id")
    if not isinstance(asset_id, str):
        raise ValueError("asset_ref content requires asset_id")
    try:
        asset = assets_by_id[asset_id]
    except KeyError as exc:
        raise ValueError(f"asset_ref content points to unknown asset: {asset_id}") from exc
    return {
        **content,
        "uri": asset.uri,
        "mime": asset.mime,
        "ext": asset.ext,
        "sha256": asset.sha256,
        "bytes": asset.bytes,
    }
```

Update nested evidence helpers to accept both old nested dictionaries such as `{"kind": "image", "format": "asset_ref", "content": {"asset_id": "img-0001"}}` and new dictionaries such as `{"type": "image", "format": "asset_ref", "content": {"asset_id": "img-0001"}}` while resolving assets:

```python
def nested_evidence(value: dict[str, Any]) -> tuple[str, str, Any] | None:
    evidence_type = value.get("type", value.get("kind"))
    fmt = value.get("format")
    if not isinstance(evidence_type, str) or not isinstance(fmt, str):
        return None
    if "content" not in value:
        return None
    return evidence_type, fmt, value["content"]
```

And in `resolve_asset_refs_in_value()`:

```python
nested = nested_evidence(value)
if nested is not None:
    evidence_type, fmt, nested_content = nested
    return {
        "type": evidence_type,
        "format": fmt,
        "content": resolve_asset_content(fmt, nested_content, assets_by_id),
    }
```

- [ ] **Step 4: Migrate Markdown backend**

In `src/rag_document_parser/extract/formats/markdown/backend.py`, remove the `Evidence` import and construct units like:

```python
EvidenceUnit(
    id=chunk_id,
    type="text",
    format="plain",
    source=SourceEvidence(
        kind="text",
        text=_with_section(section_path, text),
    ),
    content=text,
    metadata={
        "common": {
            "chunk_kind": "text",
            "section_path": list(section_path),
            "display_format": "plain",
        }
    },
)
```

For tables:

```python
EvidenceUnit(
    id=block_id,
    type="table",
    format="structured_table",
    source=SourceEvidence(kind="table", text=source_text),
    content=structured_table(headers, rows),
    metadata={
        "common": {
            "chunk_kind": "table",
            "section_path": list(section_path),
            "display_format": "structured_table",
        },
        "table": {
            "table_id": table_id,
            "headers": headers,
            "row_count": len(rows),
        },
    },
)
```

- [ ] **Step 5: Migrate HWPX/HWP5/PDF backends**

In each backend, replace:

```python
evidence=Evidence(kind="table", format="structured_table", content=structured)
```

with:

```python
format="structured_table",
content=structured,
```

Replace text evidence construction with:

```python
format="plain",
content=text,
```

Replace image evidence construction with:

```python
format="asset_ref",
content={"asset_id": asset_id, "caption": None},
```

For nested table/image children inside structured table cells, emit new nested dictionaries:

```python
{
    "type": "table",
    "format": "structured_table",
    "content": _structured_table(nested, z, bin_data_map, assets),
}
```

and:

```python
{
    "type": "image",
    "format": "asset_ref",
    "content": {"asset_id": asset_id, "caption": None},
}
```

- [ ] **Step 6: Update compatibility exports**

In `src/rag_document_parser/backends.py`, keep `Evidence` exported for compatibility because callers can import it from the legacy module. Add `EvidenceItem` to imports and `__all__`. Keep `EvidenceUnit`, `PendingAsset`, `SourceEvidence`, and backend classes exported.

- [ ] **Step 7: Run extraction tests to verify GREEN**

Run:

```bash
uv run --extra dev pytest tests/test_parser.py tests/test_hwpx_backend.py tests/test_hwp5_backend.py tests/test_pdf_backend.py tests/test_regression_corpus.py tests/test_validate_hwpx_clic_minio.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

Run:

```bash
git add src/rag_document_parser tests
git commit -m "refactor: make evidence units direct payloads"
```

## Task 3: Evidence HTML Rendering for Units and Chunks

**Files:**
- Modify: `src/rag_document_parser/evidence_html.py`
- Modify: `tests/test_evidence_html.py`

- [ ] **Step 1: Write failing HTML tests for direct units and composite evidence**

Update the first fixture in `tests/test_evidence_html.py` so units use direct shape:

```python
{
    "id": "b1",
    "type": "table",
    "format": "structured_table",
    "source": {"kind": "table", "text": "columns: 구분 | 세부"},
    "content": {
        "caption": None,
        "columns": [
            {"id": "c1", "text": "구분"},
            {"id": "c2", "text": "세부"},
        ],
        "rows": [
            {
                "index": 1,
                "cells": [
                    {
                        "column_id": "c1",
                        "text": "본인부담",
                        "rowspan": 1,
                        "colspan": 1,
                        "children": [],
                    },
                    {
                        "column_id": "c2",
                        "text": "",
                        "rowspan": 1,
                        "colspan": 1,
                        "children": [
                            {
                                "type": "table",
                                "format": "structured_table",
                                "content": {
                                    "caption": None,
                                    "columns": [
                                        {"id": "c1", "text": "항목"},
                                        {"id": "c2", "text": "금액"},
                                    ],
                                    "rows": [
                                        {
                                            "index": 1,
                                            "cells": [
                                                {
                                                    "column_id": "c1",
                                                    "text": "외래",
                                                    "rowspan": 1,
                                                    "colspan": 1,
                                                    "children": [],
                                                },
                                                {
                                                    "column_id": "c2",
                                                    "text": "1000",
                                                    "rowspan": 1,
                                                    "colspan": 1,
                                                    "children": [],
                                                },
                                            ],
                                        }
                                    ],
                                },
                            }
                        ],
                    },
                ],
            }
        ],
    },
    "metadata": {},
}
```

Add a new test:

```python
def test_render_composite_chunk_evidence_items():
    from rag_document_parser.evidence_html import render_evidence_html

    html = render_evidence_html(
        {
            "items": [
                {
                    "type": "text",
                    "format": "plain",
                    "content": "청크 설명",
                    "source_unit_ids": ["b1"],
                    "metadata": {},
                },
                {
                    "type": "table",
                    "format": "structured_table",
                    "content": {
                        "caption": None,
                        "columns": [{"id": "c1", "text": "항목"}],
                        "rows": [
                            {
                                "index": 1,
                                "cells": [
                                    {
                                        "column_id": "c1",
                                        "text": "급여",
                                        "rowspan": 1,
                                        "colspan": 1,
                                        "children": [],
                                    }
                                ],
                            }
                        ],
                    },
                    "source_unit_ids": ["b2"],
                    "metadata": {},
                },
            ]
        }
    )

    assert "청크 설명" in html
    assert "급여" in html
    assert html.count("<table") == 1
```

- [ ] **Step 2: Run HTML tests to verify RED**

Run:

```bash
uv run --extra dev pytest tests/test_evidence_html.py -q
```

Expected: FAIL because `render_evidence_units_html()` still reads `unit["evidence"]`.

- [ ] **Step 3: Normalize evidence rendering input**

In `src/rag_document_parser/evidence_html.py`, add a normalizer near `render_evidence_html()`:

```python
def _evidence_shape(evidence: dict) -> tuple[str, str | None, object]:
    if "items" in evidence and isinstance(evidence["items"], list):
        return "composite", None, evidence["items"]
    kind = evidence.get("type", evidence.get("kind", "text"))
    fmt = evidence.get("format")
    content = evidence.get("content")
    return str(kind), str(fmt) if isinstance(fmt, str) else None, content
```

Change `render_evidence_html()` so it handles composite evidence first:

```python
kind, fmt, content = _evidence_shape(evidence)
if kind == "composite" and isinstance(content, list):
    return "".join(
        render_evidence_html(item, assets_by_id=assets_by_id)
        for item in content
        if isinstance(item, dict)
    )
```

Keep existing structured table and asset rendering branches, but use normalized `kind`, `fmt`, and `content`.

- [ ] **Step 4: Render direct extraction units**

In `render_evidence_units_html()`, replace nested evidence access with:

```python
evidence = {
    "type": unit.get("type", "text"),
    "format": unit.get("format"),
    "content": unit.get("content"),
}
```

When rendering legacy unit dictionaries in tests or old MinIO outputs, keep this fallback:

```python
legacy_evidence = unit.get("evidence")
if isinstance(legacy_evidence, dict):
    evidence = legacy_evidence
```

- [ ] **Step 5: Run HTML tests to verify GREEN**

Run:

```bash
uv run --extra dev pytest tests/test_evidence_html.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/rag_document_parser/evidence_html.py tests/test_evidence_html.py
git commit -m "refactor: render direct and composite evidence"
```

## Task 4: Chunk Evidence Materialization Helpers

**Files:**
- Create: `tests/test_agentic_chunker.py`
- Create: `src/rag_document_parser/chunk/agentic.py`
- Modify: `src/rag_document_parser/chunk/__init__.py`
- Modify: `tests/test_pipeline_layout.py`

- [ ] **Step 1: Write failing tests for mixed chunk and table row materialization**

Create `tests/test_agentic_chunker.py` with:

```python
from __future__ import annotations


def _text_unit(id: str, text: str):
    from rag_document_parser import EvidenceUnit, SourceEvidence

    return EvidenceUnit(
        id=id,
        type="text",
        format="plain",
        source=SourceEvidence(kind="text", text=text),
        content=text,
        metadata={"common": {"chunk_kind": "text", "section_path": [], "display_format": "plain"}},
    )


def _table_unit(id: str):
    from rag_document_parser import EvidenceUnit, SourceEvidence

    table = {
        "caption": None,
        "columns": [
            {"id": "c1", "text": "항목"},
            {"id": "c2", "text": "내용"},
        ],
        "rows": [
            {
                "index": 1,
                "cells": [
                    {"column_id": "c1", "text": "A", "rowspan": 1, "colspan": 1, "children": []},
                    {"column_id": "c2", "text": "Alpha", "rowspan": 1, "colspan": 1, "children": []},
                ],
            },
            {
                "index": 2,
                "cells": [
                    {"column_id": "c1", "text": "B", "rowspan": 1, "colspan": 1, "children": []},
                    {"column_id": "c2", "text": "Beta", "rowspan": 1, "colspan": 1, "children": []},
                ],
            },
        ],
    }
    return EvidenceUnit(
        id=id,
        type="table",
        format="structured_table",
        source=SourceEvidence(
            kind="table",
            text="table: 2 columns\nrow 1: 항목=A; 내용=Alpha\nrow 2: 항목=B; 내용=Beta",
        ),
        content=table,
        metadata={
            "common": {"chunk_kind": "table", "section_path": [], "display_format": "structured_table"},
            "table": {"table_id": "t1", "headers": ["항목", "내용"], "row_count": 2},
        },
    )


def test_agentic_chunker_materializes_cross_kind_chunk_from_plan():
    from rag_document_parser.chunk import EvidenceUnitAgenticChunker

    units = [_text_unit("b1", "기준 설명"), _table_unit("b2")]

    def plan_fn(window, cfg, max_units):
        return [
            {
                "unit_ids": ["b1", "b2"],
                "operations": [
                    {"unit_id": "b1", "action": "include"},
                    {"unit_id": "b2", "action": "include"},
                ],
                "context_unit_ids": [],
                "title": "기준 설명과 표",
                "summary": "기준 설명과 표를 함께 제공한다.",
                "keywords": ["기준", "표"],
                "questions": ["기준 설명과 표에는 무엇이 있나요?"],
            }
        ]

    chunks = EvidenceUnitAgenticChunker(llm=None, plan_fn=plan_fn).chunk(units)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.type == "mixed"
    assert chunk.summary == "기준 설명과 표를 함께 제공한다."
    assert chunk.keywords == ["기준", "표"]
    assert chunk.questions == ["기준 설명과 표에는 무엇이 있나요?"]
    assert chunk.metadata["source_unit_ids"] == ["b1", "b2"]
    assert chunk.metadata["context_unit_ids"] == []
    assert [item.type for item in chunk.evidence.items] == ["text", "table"]
    assert chunk.evidence.items[0].content == "기준 설명"
    assert chunk.evidence.items[1].content["rows"][1]["index"] == 2


def test_agentic_chunker_materializes_table_row_subset():
    from rag_document_parser.chunk import EvidenceUnitAgenticChunker

    def plan_fn(window, cfg, max_units):
        return [
            {
                "unit_ids": ["b2"],
                "operations": [
                    {"unit_id": "b2", "action": "include_rows", "row_ranges": [[2, 2]]}
                ],
                "title": "B 항목",
                "summary": "B 항목만 제공한다.",
                "keywords": ["B", "Beta"],
                "questions": ["B 항목의 내용은 무엇인가요?"],
            }
        ]

    chunks = EvidenceUnitAgenticChunker(llm=None, plan_fn=plan_fn).chunk([_table_unit("b2")])

    table_item = chunks[0].evidence.items[0]
    assert table_item.type == "table"
    assert table_item.format == "structured_table"
    assert [row["index"] for row in table_item.content["rows"]] == [2]
    assert "row 2" in chunks[0].source.text
    assert "row 1" not in chunks[0].source.text
```

- [ ] **Step 2: Run agentic tests to verify RED**

Run:

```bash
uv run --extra dev pytest tests/test_agentic_chunker.py -q
```

Expected: FAIL because `EvidenceUnitAgenticChunker` does not exist.

- [ ] **Step 3: Create public chunker skeleton**

Create `src/rag_document_parser/chunk/agentic.py` with imports and class signature:

```python
from __future__ import annotations

import json
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from ..enrichment.llm import LlmConfig, chat_json
from ..models import Evidence, EvidenceItem, EvidenceUnit, RagChunk, SourceEvidence

PlanFn = Callable[[list[EvidenceUnit], LlmConfig | None, int], Any]


@dataclass(frozen=True)
class _WindowResult:
    chunks: list[RagChunk]
    fallback_reason: str | None = None


class EvidenceUnitAgenticChunker:
    def __init__(
        self,
        *,
        llm: LlmConfig | None,
        max_units_per_chunk: int = 10,
        window_size: int = 40,
        max_concurrency: int = 4,
        plan_fn: PlanFn | None = None,
    ) -> None:
        self._llm = llm
        self._max_units = max(1, max_units_per_chunk)
        self._window_size = max(1, window_size)
        self._concurrency = max(1, max_concurrency)
        self._plan_fn = plan_fn or self._default_plan

    def chunk(self, units: list[EvidenceUnit]) -> list[RagChunk]:
        if not units:
            return []
        windows = _windows(units, self._window_size)
        with ThreadPoolExecutor(max_workers=self._concurrency) as executor:
            results = list(executor.map(self._chunk_window, windows))
        chunks: list[RagChunk] = []
        for result in results:
            chunks.extend(result.chunks)
        return [
            _reindex_chunk(index, chunk)
            for index, chunk in enumerate(chunks, start=1)
        ]

    def _chunk_window(self, window: list[EvidenceUnit]) -> _WindowResult:
        raw_plan = self._plan_fn(window, self._llm, self._max_units)
        try:
            chunks = _materialize_window(window, raw_plan)
        except ValueError as exc:
            return _WindowResult(_fallback_chunks(window, str(exc)), str(exc))
        return _WindowResult(chunks)

    def _default_plan(self, window: list[EvidenceUnit], cfg: LlmConfig | None, max_units: int) -> Any:
        if cfg is None:
            return None
        return chat_json(_plan_prompt(window, max_units), cfg)
```

Export it in `src/rag_document_parser/chunk/__init__.py`:

```python
from .agentic import EvidenceUnitAgenticChunker
from .backend import Chunker

__all__ = ["Chunker", "EvidenceUnitAgenticChunker"]
```

Update `tests/test_pipeline_layout.py` to import and assert `EvidenceUnitAgenticChunker.__name__ == "EvidenceUnitAgenticChunker"`.

- [ ] **Step 4: Implement materialization helpers**

Add these helpers to `src/rag_document_parser/chunk/agentic.py`:

```python
def _windows(units: list[EvidenceUnit], size: int) -> list[list[EvidenceUnit]]:
    return [units[index : index + size] for index in range(0, len(units), size)]


def _materialize_window(units: list[EvidenceUnit], raw_plan: Any) -> list[RagChunk]:
    if not isinstance(raw_plan, list):
        raise ValueError("chunk plan must be a list")
    by_id = {unit.id: unit for unit in units}
    assigned: set[str] = set()
    chunks: list[RagChunk] = []
    for item in raw_plan:
        if not isinstance(item, dict):
            raise ValueError("chunk plan item must be an object")
        operations = item.get("operations")
        if not isinstance(operations, list) or not operations:
            raise ValueError("chunk plan item requires operations")
        chunk_units: list[EvidenceUnit] = []
        evidence_items: list[EvidenceItem] = []
        source_parts: list[str] = []
        normalized_ops: list[dict[str, Any]] = []
        for operation in operations:
            if not isinstance(operation, dict):
                raise ValueError("operation must be an object")
            unit_id = operation.get("unit_id")
            action = operation.get("action", "include")
            if not isinstance(unit_id, str) or unit_id not in by_id:
                raise ValueError(f"unknown unit id: {unit_id!r}")
            if unit_id in assigned:
                raise ValueError(f"duplicate unit id: {unit_id}")
            unit = by_id[unit_id]
            assigned.add(unit_id)
            chunk_units.append(unit)
            evidence_item, source_text, normalized = _materialize_operation(unit, operation)
            evidence_items.append(evidence_item)
            if source_text:
                source_parts.append(source_text)
            normalized_ops.append(normalized)
        context_unit_ids = _context_unit_ids(item.get("context_unit_ids"), by_id, assigned)
        chunks.append(_chunk_from_items(len(chunks) + 1, chunk_units, evidence_items, source_parts, item, normalized_ops, context_unit_ids))
    missing = [unit.id for unit in units if unit.id not in assigned]
    if missing:
        raise ValueError(f"chunk plan omitted units: {', '.join(missing)}")
    return chunks
```

Then add:

```python
def _materialize_operation(unit: EvidenceUnit, operation: dict[str, Any]) -> tuple[EvidenceItem, str, dict[str, Any]]:
    action = operation.get("action", "include")
    if action == "include":
        return (
            EvidenceItem(
                type=unit.type,
                format=unit.format,
                content=unit.content,
                source_unit_ids=[unit.id],
                metadata=dict(unit.metadata),
            ),
            unit.source.text,
            {"unit_id": unit.id, "action": "include"},
        )
    if action == "include_rows":
        ranges = operation.get("row_ranges")
        if unit.format != "structured_table" or not isinstance(unit.content, dict):
            raise ValueError(f"include_rows requires structured_table unit: {unit.id}")
        if not isinstance(ranges, list):
            raise ValueError("include_rows requires row_ranges")
        subset = _table_subset(unit.content, ranges)
        return (
            EvidenceItem(
                type=unit.type,
                format=unit.format,
                content=subset,
                source_unit_ids=[unit.id],
                metadata={**dict(unit.metadata), "row_ranges": ranges},
            ),
            _table_source_text(subset),
            {"unit_id": unit.id, "action": "include_rows", "row_ranges": ranges},
        )
    raise ValueError(f"unsupported action: {action!r}")
```

Implement table subset/source helpers:

```python
def _table_subset(table: dict[str, Any], ranges: list[Any]) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    for row in table.get("rows", []):
        if not isinstance(row, dict):
            continue
        index = row.get("index")
        if isinstance(index, int) and _row_selected(index, ranges):
            selected.append(row)
    if not selected:
        raise ValueError("row_ranges selected no rows")
    subset = dict(table)
    subset["rows"] = selected
    return subset


def _row_selected(index: int, ranges: list[Any]) -> bool:
    for item in ranges:
        if (
            isinstance(item, list)
            and len(item) == 2
            and isinstance(item[0], int)
            and isinstance(item[1], int)
            and item[0] <= index <= item[1]
        ):
            return True
    return False


def _table_source_text(table: dict[str, Any]) -> str:
    columns = [str(column.get("text", "")) for column in table.get("columns", []) if isinstance(column, dict)]
    lines = [f"table: {len(columns)} columns"]
    if columns:
        lines.append("columns: " + " | ".join(columns))
    for row in table.get("rows", []):
        if not isinstance(row, dict):
            continue
        values: list[str] = []
        cells = row.get("cells", [])
        if isinstance(cells, list):
            for cell in cells:
                if not isinstance(cell, dict):
                    continue
                column_id = cell.get("column_id")
                label = _column_label(columns, column_id)
                text = str(cell.get("text", "")).strip()
                if text:
                    values.append(f"{label}={text}")
        lines.append(f"row {row.get('index', '?')}: " + "; ".join(values))
    return "\n".join(lines)


def _column_label(columns: list[str], column_id: Any) -> str:
    if isinstance(column_id, str) and column_id.startswith("c") and column_id[1:].isdigit():
        idx = int(column_id[1:]) - 1
        if 0 <= idx < len(columns) and columns[idx]:
            return columns[idx]
    return str(column_id or "col")
```

Implement chunk assembly:

```python
def _chunk_from_items(
    index: int,
    units: list[EvidenceUnit],
    evidence_items: list[EvidenceItem],
    source_parts: list[str],
    plan: dict[str, Any],
    operations: list[dict[str, Any]],
    context_unit_ids: list[str],
) -> RagChunk:
    source_unit_ids = [unit.id for unit in units]
    title = plan.get("title") if isinstance(plan.get("title"), str) else ""
    summary = plan.get("summary") if isinstance(plan.get("summary"), str) else _fallback_summary(units)
    keywords = _strings(plan.get("keywords")) or _fallback_keywords(units)
    questions = _strings(plan.get("questions")) or _fallback_questions(title or summary)
    chunk_type = _chunk_type(evidence_items)
    return RagChunk(
        id=f"chunk-{index}",
        type=chunk_type,
        source=SourceEvidence(kind=chunk_type, text="\n\n".join(source_parts)),
        evidence=Evidence(items=evidence_items),
        summary=summary,
        keywords=keywords,
        questions=questions,
        metadata={
            "source_unit_ids": source_unit_ids,
            "context_unit_ids": context_unit_ids,
            "operations": operations,
            "title": title,
            "common": {
                "unit_types": _unique([unit.type for unit in units]),
                "display_format": "composite",
            },
        },
    )
```

Add `_strings`, `_context_unit_ids`, `_unique`, `_chunk_type`, `_fallback_summary`, `_fallback_keywords`, `_fallback_questions`, `_fallback_chunks`, and `_reindex_chunk`:

```python
def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _context_unit_ids(value: Any, by_id: dict[str, EvidenceUnit], assigned: set[str]) -> list[str]:
    result: list[str] = []
    for unit_id in _strings(value):
        if unit_id not in by_id:
            raise ValueError(f"unknown context unit id: {unit_id!r}")
        if unit_id not in assigned:
            raise ValueError(f"context unit id must refer to an assigned unit: {unit_id}")
        if unit_id not in result:
            result.append(unit_id)
    return result


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _chunk_type(items: list[EvidenceItem]) -> str:
    types = _unique([item.type for item in items])
    return types[0] if len(types) == 1 else "mixed"


def _fallback_summary(units: list[EvidenceUnit]) -> str:
    return " / ".join(unit.source.text.strip().replace("\n", " ")[:160] for unit in units if unit.source.text.strip())[:500]


def _fallback_keywords(units: list[EvidenceUnit]) -> list[str]:
    words: list[str] = []
    for unit in units:
        for token in re.findall(r"[0-9A-Za-z가-힣]{2,}", unit.source.text):
            if token not in words:
                words.append(token)
            if len(words) >= 8:
                return words
    return words


def _fallback_questions(topic: str) -> list[str]:
    base = topic.strip() or "이 청크"
    return [f"{base}에 대해 무엇을 알 수 있나요?"]


def _fallback_chunks(units: list[EvidenceUnit], reason: str) -> list[RagChunk]:
    chunks: list[RagChunk] = []
    for index, unit in enumerate(units, start=1):
        item = EvidenceItem(
            type=unit.type,
            format=unit.format,
            content=unit.content,
            source_unit_ids=[unit.id],
            metadata=dict(unit.metadata),
        )
        chunks.append(
            RagChunk(
                id=f"chunk-{index}",
                type=unit.type,
                source=SourceEvidence(kind=unit.type, text=unit.source.text),
                evidence=Evidence(items=[item]),
                summary=_fallback_summary([unit]),
                keywords=_fallback_keywords([unit]),
                questions=_fallback_questions(unit.source.text[:80]),
                metadata={
                    "source_unit_ids": [unit.id],
                    "context_unit_ids": [],
                    "_fallback_reason": reason,
                    "common": {"unit_types": [unit.type], "display_format": "composite"},
                },
            )
        )
    return chunks


def _reindex_chunk(index: int, chunk: RagChunk) -> RagChunk:
    return RagChunk(
        id=f"chunk-{index}",
        type=chunk.type,
        source=chunk.source,
        evidence=chunk.evidence,
        summary=chunk.summary,
        keywords=list(chunk.keywords),
        questions=list(chunk.questions),
        metadata=dict(chunk.metadata),
    )
```

- [ ] **Step 5: Run materialization tests to verify GREEN**

Run:

```bash
uv run --extra dev pytest tests/test_agentic_chunker.py tests/test_pipeline_layout.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/rag_document_parser/chunk src/rag_document_parser/models.py src/rag_document_parser/__init__.py tests/test_agentic_chunker.py tests/test_pipeline_layout.py
git commit -m "feat: materialize agentic chunk evidence"
```

## Task 5: LLM Planning, Validation, and Fallback Behavior

**Files:**
- Modify: `src/rag_document_parser/chunk/agentic.py`
- Modify: `tests/test_agentic_chunker.py`

- [ ] **Step 1: Add failing tests for fallback and prompt planning**

Append to `tests/test_agentic_chunker.py`:

```python
def test_agentic_chunker_falls_back_without_dropping_units_on_invalid_plan():
    from rag_document_parser.chunk import EvidenceUnitAgenticChunker

    units = [_text_unit("b1", "첫 문장"), _text_unit("b2", "둘째 문장")]

    def bad_plan(window, cfg, max_units):
        return [{"operations": [{"unit_id": "b1", "action": "include"}]}]

    chunks = EvidenceUnitAgenticChunker(llm=None, plan_fn=bad_plan).chunk(units)

    assert [chunk.metadata["source_unit_ids"] for chunk in chunks] == [["b1"], ["b2"]]
    assert all(chunk.metadata["_fallback_reason"].startswith("chunk plan omitted units") for chunk in chunks)


def test_agentic_chunker_uses_llm_prompt_when_no_plan_fn(monkeypatch):
    from rag_document_parser import LlmConfig
    from rag_document_parser.chunk import EvidenceUnitAgenticChunker

    calls = []

    def fake_chat_json(prompt, cfg):
        calls.append((prompt, cfg))
        return [
            {
                "unit_ids": ["b1"],
                "operations": [{"unit_id": "b1", "action": "include"}],
                "title": "첫 문장",
                "summary": "첫 문장 요약",
                "keywords": ["첫"],
                "questions": ["첫 문장은 무엇인가요?"],
            }
        ]

    monkeypatch.setattr("rag_document_parser.chunk.agentic.chat_json", fake_chat_json)
    cfg = LlmConfig(url="http://llm.test/v1", api_key="key", model="model")

    chunks = EvidenceUnitAgenticChunker(llm=cfg).chunk([_text_unit("b1", "첫 문장")])

    assert len(calls) == 1
    assert '"id": "b1"' in calls[0][0]
    assert calls[0][1] is cfg
    assert chunks[0].summary == "첫 문장 요약"


def test_agentic_chunker_records_context_units_without_duplicate_evidence():
    from rag_document_parser.chunk import EvidenceUnitAgenticChunker

    units = [_text_unit("b1", "앞 문맥"), _text_unit("b2", "대상 문장")]

    def plan_fn(window, cfg, max_units):
        return [
            {
                "unit_ids": ["b1"],
                "operations": [{"unit_id": "b1", "action": "include"}],
                "context_unit_ids": [],
                "summary": "앞 문맥",
                "keywords": ["앞"],
                "questions": ["앞 문맥은 무엇인가요?"],
            },
            {
                "unit_ids": ["b2"],
                "operations": [{"unit_id": "b2", "action": "include"}],
                "context_unit_ids": ["b1"],
                "summary": "대상 문장",
                "keywords": ["대상"],
                "questions": ["대상 문장은 무엇인가요?"],
            },
        ]

    chunks = EvidenceUnitAgenticChunker(llm=None, plan_fn=plan_fn).chunk(units)

    assert chunks[1].metadata["source_unit_ids"] == ["b2"]
    assert chunks[1].metadata["context_unit_ids"] == ["b1"]
    assert chunks[1].evidence.items[0].source_unit_ids == ["b2"]
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
uv run --extra dev pytest tests/test_agentic_chunker.py -q
```

Expected: FAIL because `_plan_prompt()` is referenced by `_default_plan()` but is not implemented yet.

- [ ] **Step 3: Implement prompt payload**

In `src/rag_document_parser/chunk/agentic.py`, add:

```python
_PROMPT = """\
당신은 RAG 인덱싱용 EvidenceUnit chunk planner입니다.
아래 unit 목록을 의미적으로 일관된 chunk plan으로 묶어 주세요.

규칙:
- evidence content는 작성하지 않습니다. unit_id와 operation만 작성합니다.
- 모든 unit은 정확히 한 번 evidence로 포함되어야 합니다.
- text, table, image를 같은 chunk에 묶을 수 있습니다.
- 큰 structured_table은 include_rows로 row range를 선택할 수 있습니다.
- 원문에 없는 사실을 summary, keywords, questions에 추가하지 않습니다.
- 한 chunk는 가능하면 unit {max_units}개 이하로 유지합니다.

Unit 목록:
{units}

JSON 배열만 출력하세요:
[
  {{
    "unit_ids": ["b1"],
    "operations": [
      {{"unit_id": "b1", "action": "include"}}
    ],
    "title": "제목",
    "summary": "요약",
    "keywords": ["키워드"],
    "questions": ["이 chunk로 답할 수 있는 질문"]
  }}
]
"""


def _plan_prompt(window: list[EvidenceUnit], max_units: int) -> str:
    payload = [_unit_payload(index, unit) for index, unit in enumerate(window)]
    return _PROMPT.replace("{max_units}", str(max_units)).replace(
        "{units}",
        json.dumps(payload, ensure_ascii=False, indent=2),
    )


def _unit_payload(index: int, unit: EvidenceUnit) -> dict[str, Any]:
    common = unit.metadata.get("common", {})
    table = unit.metadata.get("table", {})
    asset = unit.metadata.get("asset", {})
    return {
        "id": unit.id,
        "index": index,
        "type": unit.type,
        "format": unit.format,
        "section_path": common.get("section_path", []),
        "source_preview": _truncate(unit.source.text, 900),
        "table": _compact_table(table),
        "asset": dict(asset) if isinstance(asset, dict) else {},
    }


def _compact_table(table: Any) -> dict[str, Any]:
    if not isinstance(table, dict):
        return {}
    result: dict[str, Any] = {}
    if "table_id" in table:
        result["table_id"] = table["table_id"]
    if "headers" in table and isinstance(table["headers"], list):
        result["headers"] = [str(header)[:80] for header in table["headers"][:12]]
    if "row_count" in table:
        result["row_count"] = table["row_count"]
    return result


def _truncate(value: str, limit: int) -> str:
    text = value.strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"
```

- [ ] **Step 4: Harden operation validation**

In `_materialize_window()`, before materializing operations, validate `unit_ids` when present:

```python
unit_ids = item.get("unit_ids", [])
if unit_ids and not isinstance(unit_ids, list):
    raise ValueError("unit_ids must be a list")
if unit_ids:
    operation_ids = [op.get("unit_id") for op in operations if isinstance(op, dict)]
    if unit_ids != operation_ids:
        raise ValueError("unit_ids must match operation unit_ids")
```

Keep duplicate/missing-unit checks from Task 4.

- [ ] **Step 5: Run agentic chunker tests**

Run:

```bash
uv run --extra dev pytest tests/test_agentic_chunker.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/rag_document_parser/chunk/agentic.py tests/test_agentic_chunker.py
git commit -m "feat: plan evidence chunks with llm"
```

## Task 6: Documentation and Full Verification

**Files:**
- Modify: `README.md`
- Modify: `tests/test_pipeline_layout.py`

- [ ] **Step 1: Update README usage**

In `README.md`, replace the old example loop:

```python
for unit in result.units:
    send_to_chunker(unit.source, unit.evidence, unit.metadata)
    store_evidence(unit.evidence)
```

with:

```python
for unit in result.units:
    store_extracted_unit(unit.source, unit.type, unit.format, unit.content, unit.metadata)
```

Add explicit chunking usage:

```python
from rag_document_parser import EvidenceUnitAgenticChunker, LlmConfig

chunker = EvidenceUnitAgenticChunker(
    llm=LlmConfig(
        url=os.environ["LLM_URL"],
        api_key=os.environ["LLM_API_KEY"],
        model=os.environ["LLM_MODEL"],
    ),
)

chunks = chunker.chunk(result.units)

for chunk in chunks:
    index_chunk(
        source=chunk.source.text,
        evidence=chunk.evidence.to_dict(),
        summary=chunk.summary,
        keywords=chunk.keywords,
        questions=chunk.questions,
    )
```

Update the model bullet list to include `EvidenceItem` and describe `Evidence` as composite chunk evidence. Update the pipeline text to say:

```text
input -> extract EvidenceUnit -> agentic chunk -> RagChunk
```

- [ ] **Step 2: Update pipeline layout export test**

In `tests/test_pipeline_layout.py`, assert public export availability:

```python
from rag_document_parser import EvidenceItem, EvidenceUnitAgenticChunker

assert EvidenceItem.__name__ == "EvidenceItem"
assert EvidenceUnitAgenticChunker.__name__ == "EvidenceUnitAgenticChunker"
```

Also assert:

```python
from rag_document_parser.chunk import EvidenceUnitAgenticChunker as StageAgenticChunker

assert StageAgenticChunker is EvidenceUnitAgenticChunker
```

- [ ] **Step 3: Run full test suite**

Run:

```bash
uv run --extra dev pytest -q
```

Expected: all tests pass.

- [ ] **Step 4: Run compile, build, and whitespace checks**

Run:

```bash
uv run python -m compileall -q src tests
uv build
git diff --check
```

Expected: every command exits 0.

- [ ] **Step 5: Commit docs and export cleanup**

Run:

```bash
git add README.md tests/test_pipeline_layout.py src/rag_document_parser
git commit -m "docs: document evidence unit agentic chunking"
```

## Final Verification

After all tasks are complete, run:

```bash
uv run --extra dev pytest -q
uv run python -m compileall -q src tests
uv build
git diff --check
git status --short --branch
```

Expected:

- pytest exits 0
- compileall exits 0
- build exits 0
- diff check exits 0
- worktree is clean on `feature/chunking`
