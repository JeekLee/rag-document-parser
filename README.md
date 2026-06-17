# rag-document-parser

RAG-ready document parser for producing chunks, evidence payloads, and source
pointers from document formats such as HWP, HWPX, and PDF.

The parser is not a Markdown converter. Markdown is treated as one preview
format, while chunks, evidence, and source pointers are the primary output.

```python
from rag_document_parser import RagDocumentParser

result = RagDocumentParser().parse(raw_bytes, suffix=".md")

for chunk in result.chunks:
    embed(chunk.llm_text)
    store_evidence(chunk.display, source=chunk.source)
```

## Current scope

- Defines the public RAG result model:
  - `ParseResult`
  - `RagChunk`
  - `Evidence`
  - `SourceInfo`
  - `SourcePointer`
- Supports UTF-8 text/Markdown parsing as the first contract fixture.
- Converts simple Markdown tables into table chunks with LLM-readable row text.

## Next scope

- Move HWP/HWPX/PDF parsing code in from `md-converter`.
- Preserve table cell structure before rendering.
- Add HTML/table evidence for complex tables.
- Add source-page and source-region pointers where formats expose them.
