# rag-document-parser

RAG-ready document parser for producing canonical source evidence units and
user-facing evidence payloads from document formats such as HWP, HWPX, and PDF.

The parser is not a Markdown converter. Its primary output is source-preserving
evidence units for downstream agentic chunking, embedding, retrieval, LLM
grounding, and user-facing evidence display.

```python
import os

from rag_document_parser import RagDocumentParser, S3Config

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
    send_to_chunker(unit.source, unit.evidence, unit.metadata)
    store_evidence(unit.evidence)

for asset in result.assets:
    register_asset(asset.uri)
```

## Current scope

- Defines the public RAG result model:
  - `ParseResult`
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
  has intentionally been removed from parsing. Agentic chunking and
  summary/keyword/question generation happen after parsing.
- The HWPX backend extracts text, structured tables, nested tables, and image
  assets. Images embedded in table cells are preserved as nested `asset_ref`
  evidence and uploaded to S3-compatible object storage.
- Converts simple Markdown tables into table evidence units with:
  - canonical row-oriented source text for LLM grounding
  - `structured_table` evidence payloads instead of Markdown table strings
  - user-facing evidence payloads
  - `agentic-chunker`-compatible metadata such as `common.chunk_kind`

## Next scope

- Move HWP/PDF parsing code in from `md-converter`.
- Add the agentic chunking adapter that consumes `EvidenceUnit` objects and
  performs LLM-based summary, keyword, and question generation on final chunks.
- Improve HWPX complex table fidelity beyond the current rowspan, colspan,
  nested table, and nested asset baseline.
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
