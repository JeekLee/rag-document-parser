# Evidence Unit Extraction Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the extraction package around EvidenceUnit output, introduce shared evidence payload schema helpers, and move HTML rendering under a renderer package while preserving compatibility imports.

**Architecture:** Keep `EvidenceUnit` as the public parse output envelope. Move extraction internals from `extract` to `evidence_unit_extraction`, add schema helper modules under that package, and move `evidence_html.py` implementation to `renderer/evidence_unit_render.py` with deprecated re-export shims.

**Tech Stack:** Python >=3.11, dataclasses/TypedDict-style dict helpers, pytest, existing backend registry and parser pipeline.

---

### Task 1: Add New Import Paths With Compatibility Shims

**Files:**
- Create: `src/rag_document_parser/evidence_unit_extraction/`
- Create: `src/rag_document_parser/renderer/`
- Modify: `tests/test_pipeline_layout.py`
- Modify: `tests/test_evidence_html.py`

- [x] **Step 1: Write failing tests for new package paths**

Add assertions that `rag_document_parser.evidence_unit_extraction.registry.default_backends`,
`rag_document_parser.evidence_unit_extraction.formats.pdf.PdfBackend`, and
`rag_document_parser.renderer.evidence_unit_render.render_evidence_units_html` import successfully.

- [x] **Step 2: Verify red**

Run: `uv run pytest tests/test_pipeline_layout.py tests/test_evidence_html.py -q`

Expected: fail with `ModuleNotFoundError` for the new packages.

- [x] **Step 3: Move implementation files and leave shims**

Move `src/rag_document_parser/extract/*` to `src/rag_document_parser/evidence_unit_extraction/*`.
Move `src/rag_document_parser/evidence_html.py` to
`src/rag_document_parser/renderer/evidence_unit_render.py`.
Add `extract` and `evidence_html.py` compatibility modules that re-export from the new locations.

- [x] **Step 4: Verify green**

Run: `uv run pytest tests/test_pipeline_layout.py tests/test_evidence_html.py -q`

Expected: pass.

### Task 2: Update Internal Imports And Documentation

**Files:**
- Modify: `src/rag_document_parser/__init__.py`
- Modify: `src/rag_document_parser/backends.py`
- Modify: `src/rag_document_parser/hwpx.py`
- Modify: `src/rag_document_parser/pipeline/parser.py`
- Modify: `scripts/*.py`
- Modify: `tests/*.py`
- Modify: `README.md`

- [x] **Step 1: Update first-party imports to new package names**

Replace internal imports from `rag_document_parser.extract` with
`rag_document_parser.evidence_unit_extraction`, and from
`rag_document_parser.evidence_html` with
`rag_document_parser.renderer.evidence_unit_render`.

- [x] **Step 2: Keep legacy import tests**

Add or keep focused tests proving old imports still work:
`rag_document_parser.extract.assets.resolve_units` and
`rag_document_parser.evidence_html.render_evidence_units_html`.

- [x] **Step 3: Verify all import-facing tests**

Run: `uv run pytest tests/test_pipeline_layout.py tests/test_parser.py tests/test_evidence_html.py -q`

Expected: pass.

### Task 3: Introduce Evidence Unit Payload Schema Helpers

**Files:**
- Create: `src/rag_document_parser/evidence_unit_extraction/schema/__init__.py`
- Create: `src/rag_document_parser/evidence_unit_extraction/schema/asset_ref.py`
- Create: `src/rag_document_parser/evidence_unit_extraction/schema/structured_table.py`
- Create: `src/rag_document_parser/evidence_unit_extraction/schema/structured_diagram.py`
- Create: `src/rag_document_parser/evidence_unit_extraction/schema/evidence_unit.py`
- Create: `tests/test_evidence_unit_schema.py`

- [x] **Step 1: Write failing schema tests**

Add tests for `asset_ref_content`, `table_cell`, `structured_table`,
`diagram_node`, `diagram_connector`, `diagram_edge`, `structured_diagram`,
and `common_metadata`.

- [x] **Step 2: Verify red**

Run: `uv run pytest tests/test_evidence_unit_schema.py -q`

Expected: fail because schema modules do not exist yet.

- [x] **Step 3: Implement minimal schema helpers**

Implement dict-returning helpers that preserve the existing payload shapes exactly.
Do not introduce runtime-heavy validation or new dependencies.

- [x] **Step 4: Verify green**

Run: `uv run pytest tests/test_evidence_unit_schema.py -q`

Expected: pass.

### Task 4: Adopt Schema Helpers In Render-Critical Paths

**Files:**
- Modify: `src/rag_document_parser/evidence_unit_extraction/formats/markdown/tables.py`
- Modify: `src/rag_document_parser/evidence_unit_extraction/formats/hwp5/backend.py`
- Modify: `src/rag_document_parser/evidence_unit_extraction/formats/hwpx/backend.py`
- Modify: `src/rag_document_parser/evidence_unit_extraction/formats/pdf/backend.py`

- [x] **Step 1: Replace repeated payload literals gradually**

Use schema helpers for newly centralized table cells, asset refs, diagram nodes,
diagram connectors, diagram edges, and structured diagram/table root payloads where
the replacement is mechanical and low risk.

- [x] **Step 2: Verify backend behavior**

Run: `uv run pytest tests/test_hwpx_backend.py tests/test_hwp5_backend.py tests/test_pdf_backend.py tests/test_regression_corpus.py -q`

Expected: pass.

### Task 5: Final Verification

**Files:**
- All touched files

- [x] **Step 1: Run full suite**

Run: `uv run pytest -q`

Expected: pass.

- [x] **Step 2: Inspect compatibility status**

Run: `git status -sb`

Expected: only intentional new paths, shims, tests, and docs are changed.
