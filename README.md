# rag-document-parser

RAG-ready document parser for producing canonical LLM source text,
user-facing evidence payloads, and LLM-enriched chunk metadata from document
formats such as HWP, HWPX, and PDF.

The parser is not a Markdown converter. Its primary output is source-preserving
evidence units for downstream agentic chunking, embedding, retrieval, LLM
grounding, and user-facing evidence display.

```python
import os

from rag_document_parser import LlmConfig, RagDocumentParser, S3Config

parser = RagDocumentParser(
    llm=LlmConfig(
        url="https://api.openai.com/v1",
        api_key=os.environ["OPENAI_API_KEY"],
        model="gpt-4.1-mini",
    ),
    object_storage=S3Config(
        endpoint=os.environ["S3_ENDPOINT"],
        bucket=os.environ["S3_BUCKET"],
        access_key=os.environ["S3_ACCESS_KEY"],
        secret_key=os.environ["S3_SECRET_KEY"],
        prefix="rag-document-parser",
    ),
)

result = parser.parse(raw_bytes, suffix=".md")

for chunk in result.chunks:
    index_payload = {
        "summary": chunk.summary,
        "keywords": chunk.keywords,
        "questions": chunk.questions,
        "source": chunk.source,
    }
    send_to_llm(chunk.source)
    store_evidence(chunk.evidence)

for asset in result.assets:
    register_asset(asset.uri)
```

## Current scope

- Defines the public RAG result model:
  - `ParseResult`
  - `RagChunk`
  - `EvidenceUnit`
  - `Evidence`
  - `PendingAsset`
  - `DocumentAsset`
  - `SourceInfo`
  - `SourceEvidence`
  - `S3Config`
- Requires S3-compatible object storage; binary assets are uploaded and exposed
  as asset references instead of being embedded in source or evidence.
- Supports UTF-8 text/Markdown parsing and HWPX parsing.
- Selects a parser backend by suffix; `.md`, `.markdown`, `.txt`, and `.hwpx`
  are currently backed by built-in backends.
- Parser backends produce `EvidenceUnit` objects; the current default chunker
  preserves one unit as one `RagChunk` before LLM enrichment.
- The HWPX backend extracts text, structured tables, nested tables, and image
  assets. Images embedded in table cells are preserved as nested `asset_ref`
  evidence and uploaded to S3-compatible object storage.
- Requires an LLM configuration and fails parsing when chunk enrichment is
  missing or invalid.
- Converts simple Markdown tables into table evidence units with:
  - canonical row-oriented source text for LLM grounding
  - `structured_table` evidence payloads instead of Markdown table strings
  - LLM-generated summaries, keywords, and answerable questions
  - user-facing evidence payloads
  - `agentic-chunker`-compatible metadata such as `common.chunk_kind`

## Next scope

- Move HWP/PDF parsing code in from `md-converter`.
- Improve HWPX complex table fidelity beyond the current rowspan, colspan,
  nested table, and nested asset baseline.
- Add optional source locators later only if product UX needs page/region jumps.

## Validation

When local `clic-minio` and `spark-inference-gateway` are running, the HWPX
backend can be validated against a real HWPX file and upload the evidence
outputs back to MinIO:

```bash
docker exec clic-minio sh -lc \
  '/usr/bin/mc mb --ignore-existing local/rag-document-parser-test'

RDP_LLM_API_KEY="$SPARK_GATEWAY_API_KEY" \
uv run python scripts/validate_hwpx_clic_minio.py /path/to/sample.hwpx \
  --source-name "sample.hwpx"
```

The script writes and uploads:

- `evidence-units.json`
- `evidence-units.html`
- `parse-result.llm.json`
- `metrics.json`
- extracted image assets under `{document_sha256}/assets/`
