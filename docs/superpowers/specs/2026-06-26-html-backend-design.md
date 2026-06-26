# HTML Backend Design

## Context

`rag-document-parser` has built-in backends for Markdown/text, HWPX, HWP5, and
PDF. The public parser chooses a backend by normalized suffix and then reuses
the shared asset upload and asset reference resolution pipeline.

HTML should be handled as a source-preserving document format, not as plain
text. The target quality bar is close to HWPX/HWP5/PDF extraction for the
structures HTML can represent directly: sections, text blocks, lists, tables,
figures, and images.

## Goals

1. Add built-in `.html` and `.htm` support.
2. Preserve text as `EvidenceUnit(type="text", format="plain")`.
3. Convert HTML tables into `EvidenceUnit(type="table",
   format="structured_table")`.
4. Preserve `img` and `figure` content as image `asset_ref` evidence when image
   bytes are embedded in the HTML.
5. Preserve images inside table cells as nested `asset_ref` children, matching
   the HWPX/PDF structured table shape.
6. Emit quality warnings for image references or structures that cannot be fully
   preserved.

## Non-Goals

- Do not download remote image URLs during parsing.
- Do not execute JavaScript or render CSS.
- Do not infer visual layout from CSS positioning.
- Do not implement OCR for HTML images in this feature.
- Do not turn HTML into Markdown as an intermediate contract.

## Parser Backend

Add `HtmlBackend` under
`src/rag_document_parser/evidence_unit_extraction/formats/html/`.

The backend uses BeautifulSoup as an implementation dependency because real HTML
is often malformed and the parser needs predictable traversal around tables and
figures. Add `beautifulsoup4` to the project dependencies.

`HtmlBackend.supported_suffixes` should be:

```python
(".html", ".htm")
```

The default backend registry should map `.html` and `.htm` to a shared
`HtmlBackend` instance. The package root should export `HtmlBackend` like the
other built-in backend classes.

## Extraction Rules

### Text

Block-level text from `p`, `li`, `blockquote`, `pre`, and standalone text under
common sectioning containers should become plain text evidence. Whitespace
should be normalized for normal blocks and preserved enough for `pre` content to
remain intelligible.

Heading tags `h1` through `h6` update `section_path`. A heading can also produce
a text unit only when it carries meaningful standalone content not otherwise
represented by following blocks.

### Links

Anchor text should remain in the surrounding text. When an `href` is present,
the source text should preserve it in a compact form such as
`label (https://example.test/path)`.

### Tables

HTML `table` elements should become structured table evidence.

- Header labels come from `th` cells, or from the first row when no explicit
  header exists.
- `caption` should populate `StructuredTableContent.caption`.
- `rowspan` and `colspan` should be preserved on cells.
- Nested tables inside cells should become table children.
- Embedded images inside cells should become image `asset_ref` children.

The table source text should include section context, columns, row values, and
markers for nested evidence such as `image: img-0001`.

### Images And Figures

`figure` captions from `figcaption` should become the image caption when the
figure contains an image.

Image bytes are only available when `src` is a data URI. Supported data URI
images should create:

- a `PendingAsset(kind="image", data=..., mime=..., ext=...)`
- an `EvidenceUnit(type="image", format="asset_ref", content={"asset_id": ...,
  "caption": ...})`, when the image is not inside a table cell
- a nested image child with the same asset reference when the image is inside a
  table cell

Remote URLs, relative URLs, unsupported data URI MIME types, and invalid base64
emit quality warnings. URL/path references are preserved in the image source
text and warning payload, but do not create `PendingAsset` entries because the
parser does not fetch external resources.

## Quality Warnings

Use warnings with stable `type` values:

- `html_image_external_reference`: image bytes were not available because `src`
  was a URL/path.
- `html_image_data_uri_invalid`: a data URI could not be decoded.
- `html_image_mime_unsupported`: decoded image MIME was not supported.
- `html_table_structure_degraded`: a table structure could not be represented
  exactly.

Warnings should include enough metadata to diagnose the element, such as `src`,
`mime`, or table index when available.

Supported embedded image MIME types are `image/png`, `image/jpeg`, `image/gif`,
and `image/webp`.

## Testing

Add focused tests for:

1. Parser registry support for `.html` and `.htm`.
2. Public export of `HtmlBackend`.
3. Text, heading section path, links, and list extraction.
4. Structured table extraction with caption, headers, row/column spans.
5. Standalone `figure`/`img` data URI extraction into `PendingAsset` and
   `asset_ref`.
6. Image inside a table cell as nested `asset_ref`.
7. Parser-level asset upload and nested asset resolution for HTML.
8. Quality warnings for external image references and invalid data URIs.

Tests should be written before implementation and should verify the failing
behavior first.
