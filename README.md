# rag-document-parser

RAG-ready document parser for producing source evidence, user-facing evidence
payloads, and LLM-enriched chunk metadata from document formats such as HWP,
HWPX, and PDF.

The parser is not a Markdown converter. Its primary output is source-preserving
evidence units for downstream agentic chunking, embedding, retrieval, LLM
grounding, and user-facing evidence display.

```python
import os

from rag_document_parser import LlmConfig, RagDocumentParser

parser = RagDocumentParser(
    llm=LlmConfig(
        url="https://api.openai.com/v1",
        api_key=os.environ["OPENAI_API_KEY"],
        model="gpt-4.1-mini",
    )
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
```

## Current scope

- Defines the public RAG result model:
  - `ParseResult`
  - `RagChunk`
  - `EvidenceUnit`
  - `Evidence`
  - `SourceInfo`
  - `SourceEvidence`
- Supports UTF-8 text/Markdown parsing as the first contract fixture.
- Selects a parser backend by suffix; `.md`, `.markdown`, and `.txt` are
  currently backed by the built-in Markdown backend.
- Parser backends produce `EvidenceUnit` objects; the current default chunker
  preserves one unit as one `RagChunk` before LLM enrichment.
- Requires an LLM configuration and fails parsing when chunk enrichment is
  missing or invalid.
- Converts simple Markdown tables into table evidence units with:
  - structured source evidence for LLM context
  - `structured_table` evidence payloads instead of Markdown table strings
  - LLM-generated summaries, keywords, and answerable questions
  - user-facing evidence payloads
  - `agentic-chunker`-compatible metadata such as `common.chunk_kind`

## Next scope

- Move HWP/HWPX/PDF parsing code in from `md-converter`.
- Preserve table cell structure before evidence rendering.
- Add rowspan, colspan, nested blocks, and nested-table support for complex tables.
- Add optional source locators later only if product UX needs page/region jumps.
