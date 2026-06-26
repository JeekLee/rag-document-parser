# HTML Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add built-in `.html` and `.htm` parsing with text, structured table, and embedded image asset preservation.

**Architecture:** Add a new `HtmlBackend` beside the existing format backends and register it in the default backend registry. The backend parses HTML with BeautifulSoup, walks top-level document nodes in source order, emits canonical `EvidenceUnit` objects, and reuses the existing `PendingAsset` upload and `asset_ref` resolution pipeline.

**Tech Stack:** Python 3.11+, BeautifulSoup (`beautifulsoup4`), Pydantic model objects already in `rag_document_parser.models`, pytest, uv.

---

## File Structure

- Create `src/rag_document_parser/evidence_unit_extraction/formats/html/__init__.py`: package export for `HtmlBackend`.
- Create `src/rag_document_parser/evidence_unit_extraction/formats/html/backend.py`: HTML extraction implementation.
- Modify `src/rag_document_parser/evidence_unit_extraction/registry.py`: register `.html` and `.htm`.
- Modify `src/rag_document_parser/__init__.py`: public export.
- Modify `pyproject.toml`: add `beautifulsoup4>=4.12` to runtime dependencies.
- Modify `uv.lock`: refresh lockfile with `uv lock`.
- Modify `tests/test_pipeline_layout.py`: export/registry assertions.
- Create `tests/test_html_backend.py`: backend behavior tests.
- Modify `tests/test_parser.py`: parser-level upload/resolution coverage for HTML.
- Modify `README.md`: supported inputs list and format behavior table.

## Task 1: Register HTML Backend Surface

**Files:**
- Create: `src/rag_document_parser/evidence_unit_extraction/formats/html/__init__.py`
- Create: `src/rag_document_parser/evidence_unit_extraction/formats/html/backend.py`
- Modify: `src/rag_document_parser/evidence_unit_extraction/registry.py`
- Modify: `src/rag_document_parser/__init__.py`
- Modify: `pyproject.toml`
- Modify: `tests/test_pipeline_layout.py`

- [ ] **Step 1: Write failing layout tests**

Update `tests/test_pipeline_layout.py` imports and assertions:

```python
from rag_document_parser import HtmlBackend
from rag_document_parser.evidence_unit_extraction.formats.html.backend import (
    HtmlBackend as StageHtmlBackend,
)

assert StageHtmlBackend is HtmlBackend
assert HtmlBackend.supported_suffixes == (".html", ".htm")
assert isinstance(backends[".html"], HtmlBackend)
assert isinstance(backends[".htm"], HtmlBackend)
```

- [ ] **Step 2: Run layout test and verify it fails**

Run:

```bash
uv run pytest tests/test_pipeline_layout.py::test_pipeline_layout_exports_stage_and_format_modules -q
```

Expected: FAIL because `HtmlBackend` and the `formats.html` package do not exist.

- [ ] **Step 3: Add minimal backend/export/registry code**

Create `src/rag_document_parser/evidence_unit_extraction/formats/html/backend.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

from ...backend import ParsedDocument


@dataclass
class HtmlBackend:
    supported_suffixes = (".html", ".htm")

    def parse(self, data: bytes, suffix: str) -> ParsedDocument:
        return ParsedDocument(units=[])
```

Create `src/rag_document_parser/evidence_unit_extraction/formats/html/__init__.py`:

```python
from .backend import HtmlBackend

__all__ = ["HtmlBackend"]
```

Modify `registry.py` to instantiate `HtmlBackend` and map `.html`/`.htm`.
Modify package root `__init__.py` to import and include `HtmlBackend` in
`__all__`.
Add `beautifulsoup4>=4.12` to `pyproject.toml` dependencies.

- [ ] **Step 4: Run layout test and verify it passes**

Run:

```bash
uv run pytest tests/test_pipeline_layout.py::test_pipeline_layout_exports_stage_and_format_modules -q
```

Expected: PASS.

## Task 2: Extract HTML Text, Sections, Links, And Lists

**Files:**
- Modify: `tests/test_html_backend.py`
- Modify: `src/rag_document_parser/evidence_unit_extraction/formats/html/backend.py`

- [ ] **Step 1: Write failing text extraction test**

Create `tests/test_html_backend.py` with:

```python
from __future__ import annotations


def test_html_backend_extracts_text_sections_links_and_lists():
    from rag_document_parser import HtmlBackend

    raw = b"""
    <html><body>
      <h1>Coverage Rules</h1>
      <p>Apply the <a href="https://example.test/rule">rule</a> today.</p>
      <ul><li>First item</li><li>Second item</li></ul>
      <blockquote>Quoted guidance</blockquote>
      <pre>code line 1
code line 2</pre>
    </body></html>
    """

    parsed = HtmlBackend().parse(raw, ".html")

    assert [unit.type for unit in parsed.units] == ["text", "text", "text", "text", "text"]
    assert [unit.content for unit in parsed.units] == [
        "Apply the rule (https://example.test/rule) today.",
        "First item",
        "Second item",
        "Quoted guidance",
        "code line 1\ncode line 2",
    ]
    assert all(unit.format == "plain" for unit in parsed.units)
    assert all(unit.metadata["common"]["section_path"] == ["Coverage Rules"] for unit in parsed.units)
    assert parsed.units[0].source.text == (
        "section: Coverage Rules\n"
        "Apply the rule (https://example.test/rule) today."
    )
```

- [ ] **Step 2: Run text test and verify it fails**

Run:

```bash
uv run pytest tests/test_html_backend.py::test_html_backend_extracts_text_sections_links_and_lists -q
```

Expected: FAIL because `HtmlBackend.parse()` returns no units.

- [ ] **Step 3: Implement text traversal**

In `backend.py`, import BeautifulSoup, `EvidenceUnit`, `SourceEvidence`, and
`common_metadata`. Parse bytes as UTF-8 with replacement, walk `body` direct
children in source order, track `section_path` from `h1`-`h6`, and emit text
units for `p`, `li`, `blockquote`, and `pre`.

Use helper behavior:

```python
def _text_with_links(node: Tag, *, preserve_pre: bool = False) -> str:
    # Replace <a href> text with "label (href)" and normalize whitespace.
```

Unit IDs should increment as `b1`, `b2`, etc. Text units should use
`metadata=common_metadata("text", "plain", section_path=section_path)`.

- [ ] **Step 4: Run text test and verify it passes**

Run:

```bash
uv run pytest tests/test_html_backend.py::test_html_backend_extracts_text_sections_links_and_lists -q
```

Expected: PASS.

## Task 3: Extract Structured HTML Tables

**Files:**
- Modify: `tests/test_html_backend.py`
- Modify: `src/rag_document_parser/evidence_unit_extraction/formats/html/backend.py`

- [ ] **Step 1: Write failing table test**

Append:

```python
def test_html_backend_extracts_structured_table_with_caption_and_spans():
    from rag_document_parser import HtmlBackend

    raw = b"""
    <h1>Fee Criteria</h1>
    <table>
      <caption>Copay Table</caption>
      <thead><tr><th>Type</th><th>Amount</th></tr></thead>
      <tbody>
        <tr><td rowspan="2">Clinic</td><td>1000</td></tr>
        <tr><td colspan="1">2000</td></tr>
      </tbody>
    </table>
    """

    parsed = HtmlBackend().parse(raw, ".html")

    assert [unit.type for unit in parsed.units] == ["table"]
    table = parsed.units[0]
    assert table.format == "structured_table"
    assert table.content["caption"] == "Copay Table"
    assert table.content["columns"] == [
        {"id": "c1", "text": "Type"},
        {"id": "c2", "text": "Amount"},
    ]
    assert table.content["rows"] == [
        {
            "index": 1,
            "cells": [
                {"column_id": "c1", "text": "Clinic", "rowspan": 2, "colspan": 1, "children": []},
                {"column_id": "c2", "text": "1000", "rowspan": 1, "colspan": 1, "children": []},
            ],
        },
        {
            "index": 2,
            "cells": [
                {"column_id": "c2", "text": "2000", "rowspan": 1, "colspan": 1, "children": []},
            ],
        },
    ]
    assert table.metadata["common"]["section_path"] == ["Fee Criteria"]
    assert table.metadata["table"] == {
        "table_id": "t1",
        "headers": ["Type", "Amount"],
        "row_count": 2,
    }
    assert table.source.text == (
        "section: Fee Criteria\n"
        "caption: Copay Table\n"
        "columns: Type | Amount\n"
        "row 1: Type=Clinic; Amount=1000\n"
        "row 2: Amount=2000"
    )
```

- [ ] **Step 2: Run table test and verify it fails**

Run:

```bash
uv run pytest tests/test_html_backend.py::test_html_backend_extracts_structured_table_with_caption_and_spans -q
```

Expected: FAIL because tables are not yet converted.

- [ ] **Step 3: Implement table extraction**

Add helpers in `backend.py`:

```python
def _parse_table(table: Tag, section_path: list[str], table_index: int) -> EvidenceUnit:
    # Use caption text, header row from th or first row, row cells from tbody/body.
    # Preserve rowspan and colspan as integers, defaulting to 1.
```

Use `structured_table()` from the schema helper or construct the canonical
dict shape matching existing tests. Source text must include section, caption,
columns, and row values.

- [ ] **Step 4: Run table test and verify it passes**

Run:

```bash
uv run pytest tests/test_html_backend.py::test_html_backend_extracts_structured_table_with_caption_and_spans -q
```

Expected: PASS.

## Task 4: Preserve Embedded Images And Quality Warnings

**Files:**
- Modify: `tests/test_html_backend.py`
- Modify: `src/rag_document_parser/evidence_unit_extraction/formats/html/backend.py`

- [ ] **Step 1: Write failing standalone image and warning tests**

Append:

```python
import base64

PNG_BYTES = b"png bytes"


def _data_uri(data: bytes = PNG_BYTES, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def test_html_backend_preserves_figure_data_uri_image_asset():
    from rag_document_parser import HtmlBackend

    raw = f"""
    <h1>Images</h1>
    <figure>
      <img src="{_data_uri()}" alt="chart alt">
      <figcaption>Chart caption</figcaption>
    </figure>
    """.encode()

    parsed = HtmlBackend().parse(raw, ".html")

    assert [unit.type for unit in parsed.units] == ["image"]
    image = parsed.units[0]
    assert image.source.kind == "image"
    assert image.source.text == "section: Images\nimage: img-0001\ncaption: Chart caption\nalt: chart alt"
    assert image.format == "asset_ref"
    assert image.content == {"asset_id": "img-0001", "caption": "Chart caption"}
    assert parsed.assets[0].id == "img-0001"
    assert parsed.assets[0].data == PNG_BYTES
    assert parsed.assets[0].mime == "image/png"
    assert parsed.assets[0].ext == "png"


def test_html_backend_warns_for_external_and_invalid_images():
    from rag_document_parser import HtmlBackend

    raw = b'''
    <img src="https://example.test/image.png" alt="remote">
    <img src="data:image/png;base64,not-valid" alt="bad">
    <img src="data:image/svg+xml;base64,PHN2Zy8+" alt="svg">
    '''

    parsed = HtmlBackend().parse(raw, ".html")

    assert parsed.units == []
    assert parsed.assets == []
    assert [warning["type"] for warning in parsed.quality_warnings] == [
        "html_image_external_reference",
        "html_image_data_uri_invalid",
        "html_image_mime_unsupported",
    ]
```

- [ ] **Step 2: Run image tests and verify they fail**

Run:

```bash
uv run pytest tests/test_html_backend.py::test_html_backend_preserves_figure_data_uri_image_asset tests/test_html_backend.py::test_html_backend_warns_for_external_and_invalid_images -q
```

Expected: FAIL because image extraction is not implemented.

- [ ] **Step 3: Implement data URI image extraction and warnings**

Add helpers in `backend.py`:

```python
_SUPPORTED_IMAGE_MIME = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/gif": "gif",
    "image/webp": "webp",
}

def _image_from_tag(img: Tag, caption: str | None, section_path: list[str], state: _State) -> EvidenceUnit | None:
    # Decode data URI images into PendingAsset and return image evidence.
    # Add warnings for external refs, invalid base64, and unsupported MIME.
```

Use sequential asset IDs `img-0001`, `img-0002`, etc. Include `caption` and
`alt` in source text when present.

- [ ] **Step 4: Run image tests and verify they pass**

Run:

```bash
uv run pytest tests/test_html_backend.py::test_html_backend_preserves_figure_data_uri_image_asset tests/test_html_backend.py::test_html_backend_warns_for_external_and_invalid_images -q
```

Expected: PASS.

## Task 5: Preserve Nested Table Children, Nested Images, And Parser Upload Resolution

**Files:**
- Modify: `tests/test_html_backend.py`
- Modify: `tests/test_parser.py`
- Modify: `src/rag_document_parser/evidence_unit_extraction/formats/html/backend.py`

- [ ] **Step 1: Write failing nested table and image backend tests**

Append:

```python
def test_html_backend_preserves_nested_table_as_table_child():
    from rag_document_parser import HtmlBackend

    raw = b"""
    <table>
      <tr><th>Item</th><th>Detail</th></tr>
      <tr>
        <td>Criteria</td>
        <td>
          <table>
            <tr><th>Subitem</th><th>Value</th></tr>
            <tr><td>A</td><td>1</td></tr>
          </table>
        </td>
      </tr>
    </table>
    """

    parsed = HtmlBackend().parse(raw, ".html")

    table = parsed.units[0]
    nested_child = table.content["rows"][0]["cells"][1]["children"][0]
    assert nested_child["type"] == "table"
    assert nested_child["format"] == "structured_table"
    assert nested_child["content"]["columns"] == [
        {"id": "c1", "text": "Subitem"},
        {"id": "c2", "text": "Value"},
    ]
    assert nested_child["content"]["rows"][0]["cells"][0]["text"] == "A"
    assert "nested table:" in table.source.text


def test_html_backend_preserves_table_cell_image_as_nested_asset_ref():
    from rag_document_parser import HtmlBackend

    raw = f"""
    <table>
      <tr><th>Item</th><th>Image</th></tr>
      <tr><td>Criteria</td><td><img src="{_data_uri()}" alt="cell chart"></td></tr>
    </table>
    """.encode()

    parsed = HtmlBackend().parse(raw, ".html")

    table = parsed.units[0]
    image_child = table.content["rows"][0]["cells"][1]["children"][0]
    assert image_child == {
        "type": "image",
        "format": "asset_ref",
        "content": {"asset_id": "img-0001", "caption": "cell chart"},
    }
    assert parsed.assets[0].id == "img-0001"
    assert "image: img-0001" in table.source.text
```

- [ ] **Step 2: Write failing parser upload test**

Append to `tests/test_parser.py`:

```python
def test_parser_registers_html_backend_and_uploads_nested_html_images(monkeypatch):
    from rag_document_parser import RagDocumentParser

    uploads = []

    def fake_put_object(cfg, key, data, content_type):
        uploads.append((key, data, content_type))
        return f"s3://{cfg.bucket}/{cfg.prefix}/{key}"

    monkeypatch.setattr("rag_document_parser.evidence_unit_extraction.assets._put_object", fake_put_object)

    import base64

    image_bytes = b"png bytes"
    data_uri = f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}"
    raw = f"""
    <table>
      <tr><th>Item</th><th>Image</th></tr>
      <tr><td>Criteria</td><td><img src="{data_uri}" alt="cell chart"></td></tr>
    </table>
    """.encode()

    result = RagDocumentParser(object_storage=_s3_config()).parse(raw, suffix=".HTML")
    document_hash = hashlib.sha256(raw).hexdigest()

    assert result.source.suffix == ".html"
    assert uploads == [
        (
            f"{document_hash}/assets/img-0001.png",
            image_bytes,
            "image/png",
        )
    ]
    image_child = result.units[0].content["rows"][0]["cells"][1]["children"][0]
    assert image_child["content"]["uri"] == (
        f"s3://rag-assets/documents/{document_hash}/assets/img-0001.png"
    )
    assert image_child["content"]["sha256"] == hashlib.sha256(image_bytes).hexdigest()
```

- [ ] **Step 3: Run nested/parser tests and verify they fail**

Run:

```bash
uv run pytest tests/test_html_backend.py::test_html_backend_preserves_nested_table_as_table_child tests/test_html_backend.py::test_html_backend_preserves_table_cell_image_as_nested_asset_ref tests/test_parser.py::test_parser_registers_html_backend_and_uploads_nested_html_images -q
```

Expected: FAIL because nested table/image children are not implemented and
parser registry may still be incomplete.

- [ ] **Step 4: Implement nested table and image children**

When parsing a table cell, remove nested `table` elements and `img` elements
from cell text extraction, then append children:

```python
{
    "type": "image",
    "format": "asset_ref",
    "content": {"asset_id": asset_id, "caption": caption_or_alt},
}
```

For nested tables, append:

```python
{
    "type": "table",
    "format": "structured_table",
    "content": nested_table_content,
}
```

The table source text for those cells should include `nested table:` and
`image: img-0001` markers.

- [ ] **Step 5: Run nested/parser tests and verify they pass**

Run:

```bash
uv run pytest tests/test_html_backend.py::test_html_backend_preserves_nested_table_as_table_child tests/test_html_backend.py::test_html_backend_preserves_table_cell_image_as_nested_asset_ref tests/test_parser.py::test_parser_registers_html_backend_and_uploads_nested_html_images -q
```

Expected: PASS.

## Task 6: Documentation, Lockfile, And Final Verification

**Files:**
- Modify: `README.md`
- Modify: `uv.lock`

- [ ] **Step 1: Update README supported inputs**

Add `.html`, `.htm` to the built-in parser suffix list and add an HTML row to
the extraction behavior table:

```markdown
- `.html`, `.htm`: HTML backend.

| HTML | yes | yes | yes | embedded data URI images | no | no |
```

- [ ] **Step 2: Refresh lockfile**

Run:

```bash
uv lock
```

Expected: lockfile includes `beautifulsoup4` and any required parser support
dependency.

- [ ] **Step 3: Run focused tests**

Run:

```bash
uv run pytest tests/test_html_backend.py tests/test_parser.py tests/test_pipeline_layout.py -q
```

Expected: PASS.

- [ ] **Step 4: Run full test suite**

Run:

```bash
uv run pytest
```

Expected: PASS.

- [ ] **Step 5: Review changed files**

Run:

```bash
git diff --stat
git diff --check
```

Expected: no whitespace errors and changes limited to HTML backend support,
tests, dependencies, and docs.
