# EvidenceUnit Agentic Chunking Design

## Context

`rag-document-parser` currently stops at source-preserving extraction:

```text
input -> extract EvidenceUnit -> asset upload/resolve -> ParseResult
```

The public `parse()` contract intentionally returns `ParseResult.units` and does
not run LLM enrichment or chunking. Tests assert that parsed units do not have
`summary`, `keywords`, `questions`, or a `chunks` payload.

The repository already has stage placeholders:

- `src/rag_document_parser/chunk/backend.py`: `Chunker` protocol only.
- `src/rag_document_parser/enrichment/backend.py`: `Enricher` protocol only.
- `src/rag_document_parser/enrichment/llm.py`: OpenAI-compatible JSON chat
  helper.

MinIO validation outputs under
`rag-document-parser-test/rag-document-parser-results/20260617-180701-hwp5-pdf-evidence`
show why deterministic chunking is the wrong primary feature here:

- HWPX/PDF extraction already emits structural evidence units.
- HWP5 can emit many very small text units.
- PDF/HWPX table units can be large and table-heavy.
- The stable boundary across formats is `EvidenceUnit`, not HWPX-specific output.

The chunking stage should therefore be:

```text
EvidenceUnit[] -> Agentic Chunking -> RagChunk[]
```

## Goals

1. Add a format-independent agentic chunker that consumes `EvidenceUnit`
   objects from any parser backend.
2. Keep `RagDocumentParser.parse()` extraction-only.
3. Let the LLM decide semantic grouping, splitting, and context use, while
   program code materializes final evidence from original extracted content.
4. Support cross-kind chunks, such as text plus table or text plus image.
5. Preserve provenance from final chunk evidence back to source unit IDs.
6. Fail soft: invalid LLM output must preserve all source evidence instead of
   dropping content.

## Non-Goals

- Do not make chunking HWPX-specific.
- Do not add fixed-size token or character chunking as the main behavior.
- Do not let the LLM rewrite final evidence content.
- Do not make `parse()` implicitly call chunking or enrichment.
- Do not solve source locator/page jump UX beyond preserving existing metadata.

## Data Model

The current `Evidence` model mixes two different roles:

1. extracted single evidence attached to `EvidenceUnit`
2. final composite evidence attached to `RagChunk`

The implementation should separate those roles.

### EvidenceUnit

`EvidenceUnit` is the extract-stage unit. It should directly carry its evidence
shape:

```python
@dataclass(frozen=True)
class EvidenceUnit:
    id: str
    type: str
    format: str
    source: SourceEvidence
    content: Any
    metadata: dict[str, Any] = field(default_factory=dict)
```

Examples:

- `type="text"`, `format="plain"`, `content="..."`
- `type="table"`, `format="structured_table"`, `content={...}`
- `type="image"`, `format="asset_ref"`, `content={...}`

This removes the nested `EvidenceUnit.evidence.kind/format/content` wrapper.
The unit itself is already typed.

### Chunk Evidence

`RagChunk.evidence` is final user-facing evidence. It should not have top-level
`kind` or `format`. It is a composite list of evidence items:

```python
@dataclass(frozen=True)
class EvidenceItem:
    type: str
    content: Any
    format: str | None = None
    source_unit_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Evidence:
    items: list[EvidenceItem]
```

`format` remains only where it helps render or interpret an item:

- text item: usually `format="plain"`
- table item: `format="structured_table"`
- image item: `format="asset_ref"`

The top-level evidence object is just the ordered evidence payload for the
chunk.

### RagChunk

`RagChunk` keeps the current enrichment fields, but its evidence is the new
composite `Evidence`:

```python
@dataclass(frozen=True)
class RagChunk:
    id: str
    source: SourceEvidence
    evidence: Evidence
    summary: str
    keywords: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
```

`RagChunk` does not carry a top-level type. Item-level type information remains
on `RagChunk.evidence.items[].type`; aggregate type summaries for diagnostics or
rendering can be derived from those items or stored in
`metadata["common"]["unit_types"]`.

## Public API

Add a chunker implementation under `src/rag_document_parser/chunk/`:

```python
chunker = EvidenceUnitAgenticChunker(
    llm=LlmConfig(...),
    max_units_per_chunk=10,
    window_size=40,
    max_concurrency=4,
)

chunks: list[RagChunk] = chunker.chunk(parse_result.units)
```

`RagDocumentParser.parse()` remains unchanged in behavior and continues to
return only extraction results.

The existing `Chunker` protocol should remain:

```python
class Chunker(Protocol):
    def chunk(self, units: list[EvidenceUnit]) -> list[RagChunk]:
        ...
```

The implementation can later be moved to or backed by the external
`agentic-chunker` package if that package exposes a public `EvidenceUnit`
contract. The first implementation should stay local to avoid depending on
private modules in another repository.

## Chunking Flow

### 1. Normalize Units for Prompting

Convert `EvidenceUnit` objects into compact prompt records:

```json
{
  "id": "b10",
  "index": 9,
  "type": "table",
  "format": "structured_table",
  "source_preview": "table: 3 columns ...",
  "section_path": [],
  "table": {"headers": ["항목", "제목"], "row_count": 12},
  "asset": {"asset_id": "img-0001"}
}
```

Large text/table payloads are truncated for the prompt. The materializer still
uses the original full content.

### 2. Window LLM Calls

Split unit lists into windows by:

- `window_size`
- prompt character budget
- optional source character budget

Windowing keeps calls bounded for large PDF/HWPX outputs. Windows preserve unit
order and do not alter content.

### 3. Ask the LLM for a Plan

The LLM returns a JSON array of chunk plans:

```json
[
  {
    "unit_ids": ["b7", "b8"],
    "operations": [
      {"unit_id": "b7", "action": "include"},
      {"unit_id": "b8", "action": "include_rows", "row_ranges": [[1, 3]]}
    ],
    "title": "자연분만 본인부담금 면제대상",
    "summary": "...",
    "keywords": ["자연분만", "본인부담금", "면제대상"],
    "questions": ["자연분만 본인부담금 면제대상은 무엇인가요?"]
  }
]
```

Allowed actions:

- `include`: include the whole unit.
- `include_rows`: include row subsets from a `structured_table` unit.

The LLM may group cross-kind units in one chunk. It may split large table units
by row ranges. Every unit must produce evidence in exactly one chunk. A plan may
also cite already-assigned units as `context_unit_ids`; those context references
are metadata only and do not duplicate evidence items.

### 4. Validate Plans

Before materialization, validate:

- every referenced unit ID exists
- every evidence-producing unit is assigned exactly once within its window
- row ranges are valid for `structured_table` content
- text/image units do not use table-only actions
- generated metadata fields have expected JSON types

Invalid plans fall back to source-preserving chunks for the affected window.

### 5. Materialize Evidence

Build `RagChunk.evidence.items` from original `EvidenceUnit` content:

- `include` on text: one text item with original content.
- `include` on table: one table item with original structured table.
- `include_rows` on table: one table item with copied columns/header rows and
  selected rows.
- `include` on image: one image item preserving resolved `asset_ref` content.

The LLM never writes evidence item content.

Each evidence item records:

- `type`
- `format`
- `content`
- `source_unit_ids`
- item metadata copied from unit metadata plus action metadata

Chunk-level metadata records:

- `source_unit_ids`
- `context_unit_ids`
- `operations`
- `common.unit_types`
- source backend metadata such as `pdf.page`, `table.table_id`, and `asset_id`

### 6. Build Chunk Source

`RagChunk.source.text` is the ordered join of source evidence text for included
units and selected table rows. It is used for grounding and embedding, not as
the user-facing evidence payload.

For row-selected table chunks, source text is regenerated from the selected
structured rows. If a table format does not support row-level source
regeneration, the materializer uses the original table source text and records
the row range in metadata.

### 7. Fail Soft

If the LLM call fails, returns invalid JSON, omits units, duplicates units, or
creates unsupported operations:

- preserve all units in order
- create fallback chunks using whole units
- include deterministic fallback summaries/keywords/questions from source text
- mark `metadata["_fallback_reason"]`

Fallback is not the primary chunking strategy; it is only a safety path.

## Error Handling

- LLM transport or JSON errors fall back per window.
- Invalid row ranges fall back for the affected plan.
- Unknown unit IDs are ignored only if all real units remain assigned; otherwise
  the window falls back.
- Empty input returns an empty list.
- Empty LLM metadata fields are filled with deterministic fallback metadata.

## Testing Strategy

Add tests before implementation:

1. Model serialization:
   - `EvidenceUnit.to_dict()` exposes `type`, `format`, `source`, `content`,
     and `metadata` directly.
   - `Evidence.to_dict()` exposes only `items`.
2. Parser compatibility migration:
   - Markdown/HWPX/HWP5/PDF backends return direct `EvidenceUnit.content`
     instead of nested `unit.evidence`.
   - `parse()` still returns no chunks and does not call LLM.
3. Agentic chunking:
   - fake LLM groups text plus table into one mixed `RagChunk`.
   - materialized evidence contains ordered `EvidenceItem` records.
   - fake LLM splits a structured table by row range.
   - invalid LLM output falls back without dropping units.
4. Asset preservation:
   - image `asset_ref` content remains resolved after parser asset upload.
   - chunk evidence item keeps the resolved asset fields.
5. Regression fixtures:
   - run chunking tests against representative HWPX evidence units.
   - keep HWP/PDF parser tests focused on extraction until those parsers are
     stable enough for chunk quality assertions.

Verification commands:

```bash
uv run --extra dev pytest -q
uv run python -m compileall -q src tests
uv build
git diff --check
```

## Migration Notes

This is an intentional breaking cleanup of the evidence model:

- remove `Evidence.kind`
- remove top-level `Evidence.format`
- remove `EvidenceUnit.evidence`
- add `EvidenceUnit.format`
- add `EvidenceUnit.content`
- add `Evidence.items`
- add `EvidenceItem`

Legacy renderers such as `evidence_html.py` should support both extraction
units and final chunk evidence:

- extraction unit rendering uses `unit.type`, `unit.format`, `unit.content`
- chunk evidence rendering iterates `chunk.evidence.items`

The README should be updated to describe:

- parse output as extraction evidence units
- chunk output as composite final evidence chunks
- `parse()` and chunking as separate explicit pipeline stages
