# rag-document-parser

RAG-ready document parser for producing source-preserving evidence units,
structured evidence payloads, and final retrieval chunks from HWP, HWPX, PDF,
Markdown, and plain text documents.

The parser is not a Markdown converter. Its primary output is a typed evidence
contract that downstream systems can use for indexing, retrieval, LLM grounding,
manual inspection, and user-facing evidence display.

## Pipeline

```text
raw document
  -> EvidenceUnit extraction
  -> optional asset upload and asset_ref resolution
  -> optional agentic chunking
  -> RagChunk enrichment
  -> optional HTML rendering for inspection
```

The public `RagDocumentParser` handles extraction plus S3-compatible asset
upload. Format backends can also be used directly when you want raw
`ParsedDocument` output before asset upload.

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

result = parser.parse(
    raw_bytes,
    suffix=".pdf",
    source_name="notice.pdf",
)

for unit in result.units:
    store_evidence_unit(unit.to_dict())

for asset in result.assets:
    register_asset(asset.uri)

chunker = EvidenceUnitAgenticChunker(
    llm=LlmConfig(
        url=os.environ["LLM_URL"],
        api_key=os.environ["LLM_API_KEY"],
        model=os.environ["LLM_MODEL"],
    ),
    max_concurrency=4,
    enrichment_batch_token_budget=24000,
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

`enrichment_batch_token_budget` enables final chunk enrichment batching by an
estimated prompt/output token budget, not by a fixed chunk count. Leave it unset
to enrich one chunk per LLM call. For Gemini Flash-Lite, `24000` is a practical
fast default; lower it for stricter latency/error isolation and raise it only
after checking provider rate limits and JSON response stability.

## Model Contract

The canonical contract lives in `rag_document_parser.models`. Models are
Pydantic v2 models with explicit fields, runtime validation, JSON schema
introspection, attribute access, mapping-compatible access, and stable
`to_dict()` serialization.

Use model attributes inside Python code:

```python
if unit.format == "structured_table":
    first_column = unit.content.columns[0].text
```

Use `to_dict()` at storage, API, JSON, or renderer boundaries:

```python
payload = result.to_dict()
schema = RagChunk.model_json_schema()
```

Primary output models:

- `ParseResult`: public parser output with `source`, resolved `units`,
  uploaded `assets`, and `quality_warnings`.
- `ParsedDocument`: backend output before S3 upload and asset URI resolution.
- `EvidenceUnit`: extracted unit with `id`, `type`, `format`, `source`,
  structured `content`, and `metadata`.
- `RagChunk`: retrieval chunk with `source`, composite `evidence`, `summary`,
  `keywords`, `questions`, and chunk metadata.
- `Evidence` and `EvidenceItem`: composite chunk evidence. The JSON field is
  `items`; the Python attribute `chunk.evidence.items` is preserved.
- `PendingAsset` and `DocumentAsset`: pre-upload and uploaded asset records.
- `SourceInfo` and `SourceEvidence`: source metadata and source-grounding text.

Structured evidence content models:

- `AssetRefContent`: `{asset_id, caption}` references to uploaded assets.
- `StructuredTableContent`: table `caption`, `columns`, `rows`,
  `header_rows`, and optional compact metadata.
- `TableColumn`, `TableRow`, `TableCell`, `EvidenceChild`: table internals and
  nested evidence inside cells.
- `StructuredDiagramContent`: diagram `caption`, `nodes`, `edges`,
  `connectors`, optional Mermaid text, and optional asset/confidence metadata.
- `DiagramNode`, `DiagramEdge`, `DiagramConnector`, `BoundingBox`,
  `DiagramPoint`: diagram internals.
- `CommonMetadata` and `CommonMetadataPayload`: common metadata envelope used by
  extractors and chunkers.

`TypedDict` schema aliases are not part of the current implementation. Schema
helpers under `evidence_unit_extraction/schema/` construct these Pydantic model
objects directly.

## Supported Inputs

Built-in parser suffixes:

- `.md`, `.markdown`, `.txt`: text and Markdown backend.
- `.hwpx`: HWPX backend.
- `.hwp`: HWP5 backend.
- `.pdf`: PDF backend.

Current extraction behavior by format:

| Format | Text | Structured tables | Nested tables | Images/assets | Structured diagrams | OCR |
| --- | --- | --- | --- | --- | --- | --- |
| Markdown/text | yes | Markdown tables | no | no | no | no |
| HWPX | yes | yes | yes | yes | yes | optional `ocr_fn` fallback |
| HWP5 | yes | yes | yes | yes | yes | optional `ocr_fn` fallback |
| PDF | yes | yes | yes | yes | vector/fallback diagrams | local or vision OCR |

Notes:

- Parser backends produce `EvidenceUnit` objects. Chunking and
  summary/keyword/question generation are intentionally separate from parsing.
- Binary assets are uploaded to S3-compatible object storage by
  `RagDocumentParser` and represented as `asset_ref` evidence instead of being
  embedded into text or content payloads.
- Backend classes such as `HwpxBackend`, `Hwp5Backend`, and `PdfBackend` return
  `ParsedDocument` directly and do not upload assets by themselves.

## Package Layout

```text
src/rag_document_parser/
  models.py
  llm.py
  storage.py
  evidence_unit_extraction/
    backend.py
    registry.py
    assets.py
    schema/
    formats/
      markdown/
      hwpx/
      hwp5/
      pdf/
  chunk/
    backend.py
    agentic.py
    enrichment.py
  pipeline/
    parser.py
  renderer/
    evidence_unit_render.py
    rag_chunk_render.py
```

Key boundaries:

- `models.py`: canonical Pydantic contract.
- `evidence_unit_extraction/`: extraction backends, schema construction
  helpers, table source text helpers, and asset upload/resolve support.
- `pipeline/parser.py`: public parser orchestration.
- `chunk/`: `EvidenceUnitAgenticChunker`, chunker protocol, and final
  `RagChunkEnricher`.
- `renderer/`: HTML rendering for extracted evidence units and final chunks.
- `llm.py`: OpenAI-compatible `LlmConfig` plus provider-specific Qwen,
  Gemini, and Gemma config classes for chunking and PDF vision OCR.

## Rendering

HTML renderers are inspection tools. They accept canonical model objects or
serialized dictionaries and preserve plain-text newlines, structured tables,
asset references, and structured diagrams.

```python
from pathlib import Path

from rag_document_parser.renderer import (
    render_evidence_units_html,
    render_rag_chunks_html,
)

Path("evidence-units.html").write_text(
    render_evidence_units_html(
        result.units,
        title="Extracted evidence",
        assets=result.assets,
    ),
    encoding="utf-8",
)

Path("rag-chunks.html").write_text(
    render_rag_chunks_html(
        chunks,
        title="Final chunks",
        assets=result.assets,
    ),
    encoding="utf-8",
)
```

## Optional Dependencies

Install only the format dependencies you need:

```bash
uv sync --extra hwp5
uv sync --extra pdf
uv sync --extra pdf-ocr
```

For development and the full test suite:

```bash
uv sync --extra dev
uv run pytest -q
```

`pdf-ocr` installs Python bindings for local OCR fallback. The host still needs
the local Tesseract binary, Korean language data, and Poppler when
pytesseract/pdf2image OCR fallback is used.

## OCR

HWPX and HWP5 backends accept an optional `ocr_fn` callback for image fallback
OCR. PDF supports `ocr_fn`, OpenAI-compatible vision OCR, and local OCR
fallback.

PDF OCR configuration is independent from chunk enrichment configuration.
Passing `LlmConfig` to `EvidenceUnitAgenticChunker(llm=...)` does not enable PDF
vision OCR in `RagDocumentParser`. The parser's default backend registry uses
`PdfBackend()` without `ocr_llm`, so vision OCR is off unless you explicitly
install a PDF backend configured with `ocr_llm`.

```python
import os

from rag_document_parser import (
    EvidenceUnitAgenticChunker,
    GeminiLlmConfig,
    PdfBackend,
    RagDocumentParser,
    S3Config,
)

llm = GeminiLlmConfig(
    url=os.environ["LLM_URL"],
    api_key=os.environ["LLM_API_KEY"],
    model=os.environ["LLM_MODEL"],
    thinking="disabled",
)
storage = S3Config(
    endpoint=os.environ["S3_ENDPOINT"],
    bucket=os.environ["S3_BUCKET"],
    access_key=os.environ["S3_ACCESS_KEY"],
    secret_key=os.environ["S3_SECRET_KEY"],
)

parser = RagDocumentParser(
    object_storage=storage,
    backends={
        ".pdf": PdfBackend(ocr_llm=llm),
    },
)

chunker = EvidenceUnitAgenticChunker(
    llm=llm,
    max_concurrency=4,
    enrichment_batch_token_budget=24000,
)
```

These two call sites may share the same `LlmConfig`-compatible object, but they
are not wired together automatically. If both `ocr_fn` and `ocr_llm` are omitted,
`PdfBackend` can still try local OCR for scanned pages when local OCR
dependencies are installed; it will not call a vision model.

`PdfBackend` can call an OpenAI-compatible vision model before local OCR:

```python
import os

from rag_document_parser import PdfBackend, QwenLlmConfig

backend = PdfBackend(
    ocr_llm=QwenLlmConfig(
        url=os.environ.get("RDP_PDF_OCR_BASE_URL", "http://localhost:10080/v1"),
        api_key=os.environ["RDP_PDF_OCR_API_KEY"],
        model=os.environ.get("RDP_PDF_OCR_MODEL", "qwen3-vl-30b-a3b"),
        thinking="disabled",
        timeout=240.0,
    ),
)

parsed = backend.parse(raw_pdf_bytes, ".pdf")
```

For Gemini Flash-Lite OCR through Google's OpenAI-compatible endpoint:

```python
from rag_document_parser import GeminiLlmConfig, PdfBackend

backend = PdfBackend(
    ocr_llm=GeminiLlmConfig(
        url="https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        api_key=os.environ["GEMINI_API_KEY"],
        model="gemini-2.5-flash-lite",
        thinking="disabled",
        timeout=240.0,
    ),
)
```

`QwenLlmConfig(thinking="disabled")` uses Qwen's local OpenAI-compatible hard
switch shape, `chat_template_kwargs.enable_thinking = false`. For DashScope's
OpenAI-compatible endpoint, set `thinking_parameter="enable_thinking"` to send
top-level `enable_thinking`.

When both `ocr_fn` and `ocr_llm` are configured, `ocr_fn` takes precedence. If
vision OCR fails or returns empty text, the backend falls back to the local OCR
path when local OCR dependencies are available.

## Validation Scripts

The repository includes corpus-oriented scripts for local MinIO validation:

- `scripts/validate_hwpx_clic_minio.py`: validate one HWPX file, upload
  evidence JSON/HTML, metrics, and assets.
- `scripts/validate_hwp5_clic_minio.py`: validate one HWP file, upload evidence
  JSON/HTML, metrics, and assets.
- `scripts/scan_hwp5_clic_minio.py`: scan HWP5 corpus extraction outliers.
- `scripts/scan_hwpx_clic_minio_size_samples.py`: scan and report HWPX size
  samples from the CLIC MinIO corpus.
- `scripts/validate_pdf_clic_size_samples.py`: sample PDF corpus by size band,
  validate extraction, upload HTML/JSON reports, and emit summary indexes.

Example local MinIO setup:

```bash
docker exec clic-minio sh -lc \
  '/usr/bin/mc mb --ignore-existing local/rag-document-parser-test'

docker exec clic-minio sh -lc \
  '/usr/bin/mc anonymous set download local/rag-document-parser-test'
```

Validate a single HWPX file:

```bash
uv run python scripts/validate_hwpx_clic_minio.py /path/to/sample.hwpx \
  --source-name "sample.hwpx" \
  --public-asset-endpoint "http://<browser-reachable-server>:10190"
```

`--public-asset-endpoint` must point to the S3 API endpoint reachable from the
browser opening the generated HTML. Evidence JSON keeps canonical
`s3://bucket/key` URIs; generated HTML uses public URLs only for display.

Scan HWP5 corpus outliers:

```bash
uv run python scripts/scan_hwp5_clic_minio.py \
  --max-documents 300 \
  --top 30 \
  --output /tmp/hwp5-scan-300.json
```

## Current Scope

- Source-preserving evidence extraction for text, tables, images, diagrams, and
  scanned PDF OCR text.
- Strong Pydantic object contracts for evidence units, chunk evidence, assets,
  structured tables, structured diagrams, and shared LLM configuration.
- Agentic chunk planning with table row splitting, omitted-row repair,
  boundary merging, and final chunk enrichment.
- HTML inspection output for extracted evidence units and final chunks.
- S3-compatible asset upload and browser-resolvable asset rendering support.

## Known Follow-Up Areas

- Tune production chunking prompts against broader retrieval-quality metrics.
- Continue improving complex table fidelity, especially PDF header inference,
  fragmentation, and continuation handling.
- Add source locators only if product UX needs page/region jump behavior.
