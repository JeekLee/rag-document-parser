# rag-document-parser

RAG-ready document parser for producing chunks, evidence payloads, and source
pointers from document formats such as HWP, HWPX, and PDF.

The parser is not a Markdown converter. Its primary output is source-preserving
evidence units for downstream agentic chunking, embedding, retrieval, and
user-facing source restoration.

```python
from rag_document_parser import RagDocumentParser

result = RagDocumentParser().parse(raw_bytes, suffix=".md")

for chunk in result.chunks:
    embed(chunk.embedding_text)
    store_evidence(chunk.evidence, source_pointer=chunk.source_pointer)
```

## Current scope

- Defines the public RAG result model:
  - `ParseResult`
  - `RagChunk`
  - `Evidence`
  - `SourceInfo`
  - `SourcePointer`
- Supports UTF-8 text/Markdown parsing as the first contract fixture.
- Converts simple Markdown tables into table evidence units with:
  - displayable source evidence
  - embedding-oriented text
  - standardized source pointers
  - `agentic-chunker`-compatible metadata such as `common.chunk_kind`

## Next scope

- Move HWP/HWPX/PDF parsing code in from `md-converter`.
- Preserve table cell structure before evidence rendering.
- Add HTML/table evidence for complex tables.
- Add source-page and source-region pointers where formats expose them.
