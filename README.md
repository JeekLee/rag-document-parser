# rag-document-parser

RAG-ready document parser for producing embedding text, source evidence, and
user-facing evidence payloads from document formats such as HWP, HWPX, and PDF.

The parser is not a Markdown converter. Its primary output is source-preserving
evidence units for downstream agentic chunking, embedding, retrieval, LLM
grounding, and user-facing evidence display.

```python
from rag_document_parser import RagDocumentParser

result = RagDocumentParser().parse(raw_bytes, suffix=".md")

for chunk in result.chunks:
    embed(chunk.embedding_text)
    send_to_llm(chunk.source)
    store_evidence(chunk.evidence)
```

## Current scope

- Defines the public RAG result model:
  - `ParseResult`
  - `RagChunk`
  - `Evidence`
  - `SourceInfo`
  - `SourceEvidence`
- Supports UTF-8 text/Markdown parsing as the first contract fixture.
- Converts simple Markdown tables into table evidence units with:
  - structured source evidence for LLM context
  - embedding-oriented text
  - user-facing evidence payloads
  - `agentic-chunker`-compatible metadata such as `common.chunk_kind`

## Next scope

- Move HWP/HWPX/PDF parsing code in from `md-converter`.
- Preserve table cell structure before evidence rendering.
- Add HTML/table evidence for complex tables.
- Add optional source locators later only if product UX needs page/region jumps.
