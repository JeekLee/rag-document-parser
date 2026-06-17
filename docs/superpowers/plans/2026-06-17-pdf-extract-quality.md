# PDF Extract Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Improve PDF evidence extraction quality against the paired HWPX/HWP corpus baseline, preserving grouped table headers, reducing page-fragmented tables, and converting title-like table artifacts into text units.

**Architecture:** Keep the PDF backend's current pdfplumber/PyMuPDF/OCR pipeline, but move md-converter's Markdown-stage cleanup ideas into structured evidence generation. Header inference happens while converting pdfplumber tables to structured table payloads; continuation merging happens on `_Segment` table payloads before `EvidenceUnit` creation; title-table conversion remains a page segmentation concern.

**Tech Stack:** Python 3.12, pytest, pdfplumber table objects, existing `EvidenceUnit` dataclasses.

---

### Task 1: Regression Tests For PDF Quality

**Files:**
- Modify: `tests/test_pdf_backend.py`
- Modify: `tests/test_regression_corpus.py`

- [x] **Step 1: Write failing grouped-header unit tests**

Add tests showing that a two-row PDF header with blank merged-cell placeholders produces combined column labels such as `현행 / 항목`, and that title-only multi-cell tables become text units.

- [x] **Step 2: Write failing corpus quality tests**

Add corpus assertions for `pdf-benefit-criteria-2024-278` and `pdf-cesarean-copay-qa`: no blank column labels, no zero-row title tables, and lower table fragmentation after continuation merge.

- [x] **Step 3: Verify RED**

Run:

```bash
uv run pytest tests/test_pdf_backend.py tests/test_regression_corpus.py -q
```

Expected: tests fail because PDF columns still use only the first row, repeated-page tables are separate units, and title tables are emitted as table evidence.

### Task 2: Grouped Header Inference

**Files:**
- Modify: `src/rag_document_parser/extract/formats/pdf/backend.py`
- Test: `tests/test_pdf_backend.py`
- Test: `tests/test_regression_corpus.py`

- [x] **Step 1: Implement header-depth inference**

Add helpers that detect a two-row grouped header when the first row contains blank merged placeholders and the second row contains concise leaf labels.

- [x] **Step 2: Build semantic column labels**

Forward-fill group labels from the first header row and combine them with leaf labels from the second row. Example: `["현행", "", "", "개정"]` plus `["항목", "제목", "세부인정사항", "항목"]` becomes `현행 / 항목`, `현행 / 제목`, `현행 / 세부인정사항`, `개정 / 항목`.

- [x] **Step 3: Verify GREEN**

Run:

```bash
uv run pytest tests/test_pdf_backend.py tests/test_regression_corpus.py -q
```

Expected: grouped-header tests pass; continuation-related corpus assertions may still fail until Task 3.

### Task 3: Structured Table Continuation Merge

**Files:**
- Modify: `src/rag_document_parser/extract/formats/pdf/backend.py`
- Test: `tests/test_pdf_backend.py`
- Test: `tests/test_regression_corpus.py`

- [x] **Step 1: Merge adjacent continuation tables**

Before converting segments to units, merge a table segment into the previous table segment when both table payloads have the same semantic columns, appear on the same or next page in document order, and there is no intervening text segment before the later table on that page.

- [x] **Step 2: Renumber merged rows**

Append row payloads from continuation tables and rewrite row indexes sequentially. Keep the first table's header rows and first page metadata.

- [x] **Step 3: Verify GREEN**

Run:

```bash
uv run pytest tests/test_pdf_backend.py tests/test_regression_corpus.py -q
```

Expected: benefit and cesarean table counts move closer to HWPX baseline without losing source text.

### Task 4: Title Table Artifact Conversion

**Files:**
- Modify: `src/rag_document_parser/extract/formats/pdf/backend.py`
- Test: `tests/test_pdf_backend.py`
- Test: `tests/test_regression_corpus.py`

- [x] **Step 1: Convert title-only tables to text**

Extend the existing single-cell table conversion so a table with header cells and no body rows becomes a text segment when all cells are short and the joined text is title-like.

- [x] **Step 2: Preserve CJK split words**

Join title cells using the existing CJK line-join behavior so split cells such as `질병군 적용 대` + `상` become `질병군 적용 대상`.

- [x] **Step 3: Verify GREEN**

Run:

```bash
uv run pytest tests/test_pdf_backend.py tests/test_regression_corpus.py -q
```

Expected: title artifacts no longer appear as zero-row table evidence.

### Task 5: Quality Check And Full Verification

**Files:**
- Modify if needed: `tests/test_regression_corpus.py`

- [x] **Step 1: Run PDF/HWPX quality summary**

Run a local summary script over the regression corpus and compare PDF table counts, blank columns, generic labels, and source character counts against the HWPX/HWP paired documents.

- [x] **Step 2: Run full verification**

Run:

```bash
uv run pytest
uv run python -m compileall -q src tests
git diff --check
```

Expected: all commands exit 0.

### Task 6: Scanned PDF Vision OCR

**Files:**
- Modify: `src/rag_document_parser/extract/formats/pdf/backend.py`
- Modify: `src/rag_document_parser/extract/formats/pdf/__init__.py`
- Modify: `src/rag_document_parser/__init__.py`
- Modify: `tests/test_pdf_backend.py`
- Modify: `tests/test_pipeline_layout.py`
- Modify: `README.md`

- [x] **Step 1: Add OpenAI-compatible OCR config**

Add a small PDF OCR configuration object with base URL, API key, model, timeout,
and temperature fields. Keep it optional so the existing local OCR fallback
remains the default behavior.

- [x] **Step 2: Route scanned page PNGs through vision OCR first**

When `ocr_llm` is configured, render the scanned page to PNG and call
`/chat/completions` with an image data URL. If the request fails or returns
empty content, fall back to the current pytesseract/pdf2image path.

- [x] **Step 3: Verify against spark-gateway**

Run the scanned regression PDF with the local gateway model
`qwen3-vl-30b-a3b`. Expected: scanned pages produce meaningful Korean OCR
text instead of placeholder page labels.

### Task 7: HWPX Parity Tightening

**Files:**
- Modify: `src/rag_document_parser/extract/formats/pdf/backend.py`
- Modify: `tests/test_pdf_backend.py`
- Modify: `tests/test_regression_corpus.py`
- Modify: `tests/fixtures/corpus/manifest.json`

- [x] **Step 1: Merge wrapped PDF rows conservatively**

Merge rows that are PDF line-wrap artifacts into the previous logical row while
preserving bullet subrows and table-of-contents rows as separate evidence rows.

- [x] **Step 2: Promote structured text blocks**

Convert revision-history text blocks that pdfplumber misses as tables into
structured table evidence. Split official notice, related-basis, and sectioned
heading text blocks into HWPX-like paragraph units.

- [x] **Step 3: Tighten corpus parity assertions**

Assert HWPX-aligned row shapes and unit counts for the benefit and cesarean
paired PDFs, and require the ultrasound PDF to preserve the revision-history
table and higher unit/table minimums.
