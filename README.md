# rag-document-parser

RAG-ready document parser for producing canonical source evidence units and
user-facing evidence payloads from document formats such as HWP, HWPX, and PDF.

The parser is not a Markdown converter. Its primary output is source-preserving
evidence units for downstream agentic chunking, embedding, retrieval, LLM
grounding, and user-facing evidence display.

```python
import os

from rag_document_parser import (
    EvidenceUnitAgenticChunker,
    LlmConfig,
    RagDocumentParser,
    S3Config,
)

parser = RagDocumentParser(
    object_storage=S3Config(
        endpoint=os.environ["S3_ENDPOINT"],
        bucket=os.environ["S3_BUCKET"],
        access_key=os.environ["S3_ACCESS_KEY"],
        secret_key=os.environ["S3_SECRET_KEY"],
        prefix="rag-document-parser",
    ),
)

result = parser.parse(raw_bytes, suffix=".md")

for unit in result.units:
    store_extracted_unit(unit.id, unit.source, unit.type, unit.format, unit.content, unit.metadata)

for asset in result.assets:
    register_asset(asset.uri)

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

## Current scope

- Defines the public RAG result model:
  - `ParseResult`: parser output containing extracted units and uploaded assets.
  - `EvidenceUnit`: extraction-stage id, source, content, type, format, and
    metadata.
  - `EvidenceItem`: one evidence payload item inside composite chunk evidence.
  - `Evidence`: composite chunk evidence made from one or more `EvidenceItem`
    values for a `RagChunk`.
  - `RagChunk`: final chunk with source text, composite evidence, summary,
    keywords, questions, and metadata.
  - `PendingAsset`
  - `DocumentAsset`
  - `SourceInfo`
  - `SourceEvidence`
  - `S3Config`
- Requires S3-compatible object storage; binary assets are uploaded and exposed
  as asset references instead of being embedded in source or evidence.
- Supports UTF-8 text/Markdown, HWPX, HWP5 (`.hwp`), and PDF parsing.
- Selects a parser backend by suffix; `.md`, `.markdown`, `.txt`, `.hwpx`,
  `.hwp`, and `.pdf` are backed by built-in backends.
- Parser backends produce `EvidenceUnit` objects; the current default chunker
  has intentionally been removed from parsing. Agentic chunking and
  summary/keyword/question generation happen after parsing.
- The HWPX backend extracts text, coordinate-based structured tables, nested
  tables, multi-row headers, merged cells, and image assets. Images embedded in
  table cells are preserved as nested `asset_ref` evidence and uploaded to
  S3-compatible object storage.
- The HWP5 backend adapts the `md-converter` record parser to produce text,
  structured table, nested table, and image evidence units instead of Markdown.
- The PDF backend adapts the `md-converter` pdfplumber/OCR flow to produce
  page-ordered text, structured tables, nested table evidence, image assets,
  and OCR text units for scanned pages.
- Converts simple Markdown tables into table evidence units with:
  - canonical row-oriented source text for LLM grounding
  - `structured_table` evidence payloads instead of Markdown table strings
  - user-facing evidence payloads
  - `agentic-chunker`-compatible metadata such as `common.chunk_kind`

## Internal pipeline layout

The package is organized around the document pipeline:

```text
source -> evidence_unit_extraction EvidenceUnit -> agentic chunk -> RagChunk
```

- `evidence_unit_extraction/`: EvidenceUnit extraction, asset upload/resolve,
  backend registry, and shared evidence payload schema helpers.
- `evidence_unit_extraction/formats/<format>/backend.py`: format-specific
  extraction entrypoints for Markdown, HWPX, HWP5, and PDF.
- `renderer/`: EvidenceUnit and RagChunk HTML rendering.
- `pipeline/`: orchestration for the public parser API.
- `chunk/`: chunker protocol, `EvidenceUnitAgenticChunker`, and final
  `RagChunk` enrichment.
- `llm.py`: shared OpenAI-compatible LLM config and chat-completions helpers
  for chunking and PDF vision OCR.

## Optional dependencies

Install format dependencies explicitly when using non-HWPX formats:

```bash
uv sync --extra hwp5
uv sync --extra pdf
uv sync --extra pdf-ocr
```

`pdf-ocr` includes Python bindings for OCR fallback. The local Tesseract binary,
Korean language data, and Poppler must still be installed on the host if
pytesseract/pdf2image OCR fallback is used.

For scanned PDFs, `PdfBackend` can also call an OpenAI-compatible vision model
before the local OCR fallback:

```python
import os

from rag_document_parser import LlmConfig, PdfBackend

backend = PdfBackend(
    ocr_llm=LlmConfig(
        url=os.environ.get("RDP_PDF_OCR_BASE_URL", "http://localhost:10080/v1"),
        api_key=os.environ["RDP_PDF_OCR_API_KEY"],
        model=os.environ.get("RDP_PDF_OCR_MODEL", "qwen3-vl-30b-a3b"),
        timeout=240.0,
    ),
)

result = backend.parse(raw_pdf_bytes, suffix=".pdf")
```

When both `ocr_fn` and `ocr_llm` are configured, `ocr_fn` takes precedence. If
the vision OCR request fails or returns empty text, the backend falls back to
the local pytesseract/pdf2image path.

## Next scope

- Tune and evaluate production chunking prompts against broader document
  fixtures and retrieval-quality metrics.
- Improve complex table fidelity beyond the current HWPX/HWP5/PDF baseline,
  especially header inference and PDF table fragmentation.
- Add optional source locators later only if product UX needs page/region jumps.

## Validation

When local `clic-minio` is running, the HWPX backend can be validated against a
real HWPX file and upload the evidence outputs back to MinIO:

```bash
docker exec clic-minio sh -lc \
  '/usr/bin/mc mb --ignore-existing local/rag-document-parser-test'

docker exec clic-minio sh -lc \
  '/usr/bin/mc anonymous set download local/rag-document-parser-test'

uv run python scripts/validate_hwpx_clic_minio.py /path/to/sample.hwpx \
  --source-name "sample.hwpx" \
  --public-asset-endpoint "http://<browser-reachable-server>:10190"
```

`--public-asset-endpoint` must point to the MinIO/S3 API endpoint reachable
from the browser opening `evidence-units.html`; for local checks this can be
`http://localhost:10190`, and for external checks it should be the server IP or
DNS name plus the published S3 API port. Evidence JSON keeps canonical
`s3://bucket/key` asset URIs; the generated HTML uses `public_url` only for
browser rendering.

The script writes and uploads:

- `evidence-units.json`
- `evidence-units.html`
- `metrics.json`
- extracted image assets under `{document_sha256}/assets/`

For HWP5 corpus checks, scan table and diagram extraction outliers from the
same local MinIO corpus:

```bash
uv run python scripts/scan_hwp5_clic_minio.py \
  --max-documents 300 \
  --top 30 \
  --output /tmp/hwp5-scan-300.json
```

The scanner reports per-document table counts, cell counts, blank ratios, span
counts, diagram node/connectors/edge counts, and ranked table and diagram
outliers so parser changes can be compared against a stable corpus slice.
