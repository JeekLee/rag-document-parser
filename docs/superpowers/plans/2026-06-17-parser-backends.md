# Parser Backends Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split raw document parsing from RAG chunk enrichment so HWPX/HWP/PDF parsers can be added behind a stable interface.

**Architecture:** `RagDocumentParser` selects a backend by suffix. Backends return unenriched `RagChunk` objects from normalized document content, then the existing LLM enrichment step adds `summary`, `keywords`, and `questions`.

**Tech Stack:** Python 3.11 dataclasses, pytest, stdlib only.

---

### Task 1: Backend Selection Contract

**Files:**
- Modify: `tests/test_parser.py`
- Modify: `src/rag_document_parser/parser.py`
- Create: `src/rag_document_parser/backends.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing tests**

Add tests that show `.txt` and `.md` are handled by the Markdown backend, custom registered suffixes can provide chunks, and unsupported suffixes fail before LLM calls.

- [ ] **Step 2: Verify RED**

Run: `uv run --extra dev pytest -q`

Expected: tests fail because backend registration and unsupported suffix behavior do not exist.

- [ ] **Step 3: Implement minimal backend layer**

Create `DocumentBackend`, `MarkdownBackend`, `ParsedDocument`, and suffix registration inside `RagDocumentParser`.

- [ ] **Step 4: Verify GREEN**

Run: `uv run --extra dev pytest -q`

Expected: all tests pass.

### Task 2: Documentation And Full Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update docs**

Document the backend boundary and current supported suffixes.

- [ ] **Step 2: Run full verification**

Run:

```bash
uv run --extra dev pytest -q
uv run python -m compileall -q src tests
uv build
git diff --check
```

Expected: every command exits 0.
