# Current Architecture

This document describes the current package boundaries and dependency rules.
The dated files under `docs/superpowers/plans/` and
`docs/superpowers/specs/` are historical design and implementation records;
their paths and proposed structures are not the current source of truth.

## Processing Flow

```text
document bytes
  -> RagDocumentParser
  -> suffix-selected DocumentBackend
  -> ParsedDocument (EvidenceUnit + PendingAsset + warnings)
  -> optional AssetStore upload and asset reference resolution
  -> ParseResult
  -> optional EvidenceUnitAgenticChunker
  -> optional RagChunkEnricher through LlmGateway
  -> RagChunk
  -> optional HTML renderer
```

Extraction and retrieval chunking are separate application stages.
`RagDocumentParser.parse()` stops at `ParseResult`; it does not invoke an LLM
or create retrieval chunks.

## Package Responsibilities

### Contracts

- `models.py` owns the canonical Pydantic models exchanged across package
  boundaries. Serialized dictionaries are produced only at storage, API, or
  rendering boundaries.
- `evidence_unit_extraction/backend.py` defines the `DocumentBackend` protocol.
- `ports/asset_store.py` and `ports/llm.py` define the application-facing
  `AssetStore` and `LlmGateway` protocols.

Contracts must not depend on a concrete storage provider, LLM provider, or
document-format implementation.

### Application orchestration

- `pipeline/parser.py` normalizes input, selects a backend, uploads pending
  assets through `AssetStore`, resolves asset references, and builds the public
  `ParseResult`.
- `evidence_unit_extraction/registry.py` contains only the built-in suffix to
  backend mapping. Applications can supply additional or replacement mappings
  through `RagDocumentParser(backends=...)`.
- `chunk/` owns format-independent planning, validation, materialization,
  boundary handling, token accounting, and final enrichment.

Application orchestration may depend on contracts and ports, but it must not
contain format parsing rules or provider-specific network code.

### Format domains

Each package under `evidence_unit_extraction/formats/` owns the parsing rules
for one document format:

- `html/`: unsafe-source repair, DOM traversal, text interpretation, embedded
  assets, nested tables, and source projection.
- `hwpx/`: package reading, XML interpretation, table and diagram construction,
  source projection, and document/OCR assembly.
- `hwp5/`: OLE and binary-record reading, intermediate parsed blocks, table and
  diagram construction, source projection, and document/OCR assembly.
- `xlsx/`: package validation, workbook loading, worksheet interpretation,
  region detection, and source projection.
- `pdf/`: page orchestration, embedded images, vector diagrams, OCR parsing,
  segment processing, table extraction and normalization, and source
  projection.
- `markdown/`: an internal backend that is not registered by default.

A format backend is its package's entry point. It may coordinate focused
services and translate dependency or input failures, but reusable parsing rules
belong in the focused modules rather than accumulating in `backend.py`.
Format packages may share canonical models and common extraction helpers, but
must not import another format package's internals.

### Infrastructure and presentation

- `llm.py` implements the OpenAI-compatible `LlmGateway` adapter and
  provider-specific request configuration.
- `storage.py` implements the S3-compatible `AssetStore` adapter.
- `renderer/` renders canonical evidence and chunk models for inspection. It
  does not participate in parsing or mutate extraction results.

`LlmConfig` and `S3Config` remain supported convenience configuration objects.
New integrations should prefer injecting the corresponding port when they need
custom provider behavior. The OCR compatibility surface accepts either an
`LlmConfig` or an `LlmGateway`; internally it adapts configuration to the port.

## Dependency Rules

The intended dependency direction is:

```text
public API
  -> application orchestration
  -> contracts and ports
  <- infrastructure adapters

format backend
  -> its own focused services
  -> shared extraction helpers and canonical models
```

In practical terms:

1. `models.py` and `ports/` stay independent of pipeline, format, storage, and
   provider implementations.
2. `pipeline/` communicates with parsing and storage through `DocumentBackend`
   and `AssetStore`.
3. `chunk/` remains format-independent and communicates with LLMs through
   `LlmGateway`.
4. A format package owns its intermediate types and parsing policy; other
   formats do not reuse those internals.
5. `backend.py` files expose stable entry points and orchestration, not a second
   copy of domain logic.
6. Provider-specific request shapes and network calls remain in infrastructure
   adapters.
7. Renderers consume completed models and never become extraction dependencies.

## Adding or Changing a Backend

When adding a built-in format or materially changing an existing one:

1. Implement `DocumentBackend.parse(data, suffix) -> ParsedDocument` in the
   format package.
2. Separate package reading, interpretation, evidence construction, and source
   projection when they have distinct responsibilities.
3. Keep intermediate models private to the format package unless they are true
   cross-format contracts.
4. Register supported suffixes in `default_backends()` only when the backend is
   ready for default use.
5. Export a public backend from `rag_document_parser.__init__` only when direct
   application use is supported.
6. Cover backend behavior, public registration, malformed input, and relevant
   package-boundary assumptions with tests.
7. Update the README supported-input matrix and package layout.

Legacy import paths are intentionally not part of the current architecture.
Compatibility should be provided through explicit constructor adapters or
documented public exports rather than recreating obsolete package trees.
