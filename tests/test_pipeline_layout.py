from __future__ import annotations

import importlib.util


def test_pipeline_layout_exports_stage_and_format_modules():
    from rag_document_parser import (
        EvidenceItem,
        EvidenceUnitAgenticChunker,
        Hwp5Backend,
        HwpxBackend,
        MarkdownBackend,
        PdfBackend,
        PdfOcrConfig,
        RagChunkEnricher,
        RagDocumentParser,
    )
    from rag_document_parser.chunk import EvidenceUnitAgenticChunker as StageAgenticChunker
    from rag_document_parser.chunk.backend import Chunker
    from rag_document_parser.chunk.enrichment import Enricher
    from rag_document_parser.chunk.enrichment import RagChunkEnricher as StageRagChunkEnricher
    from rag_document_parser.evidence_unit_extraction.backend import (
        DocumentBackend,
        ParsedDocument,
    )
    from rag_document_parser.evidence_unit_extraction.formats.hwp5.backend import (
        Hwp5Backend as StageHwp5Backend,
    )
    from rag_document_parser.evidence_unit_extraction.formats.hwpx.backend import (
        HwpxBackend as StageHwpxBackend,
    )
    from rag_document_parser.evidence_unit_extraction.formats.markdown.backend import (
        MarkdownBackend as StageMarkdownBackend,
    )
    from rag_document_parser.evidence_unit_extraction.formats.pdf.backend import (
        PdfBackend as StagePdfBackend,
        PdfOcrConfig as StagePdfOcrConfig,
    )
    from rag_document_parser.evidence_unit_extraction.registry import default_backends
    from rag_document_parser.llm import LlmConfig
    from rag_document_parser.pipeline.parser import RagDocumentParser as StageParser

    backends = default_backends()

    assert StageParser is RagDocumentParser
    assert StageHwp5Backend is Hwp5Backend
    assert StageHwpxBackend is HwpxBackend
    assert StageMarkdownBackend is MarkdownBackend
    assert StagePdfBackend is PdfBackend
    assert StagePdfOcrConfig is PdfOcrConfig
    assert PdfOcrConfig is LlmConfig
    assert ParsedDocument.__name__ == "ParsedDocument"
    assert DocumentBackend.__name__ == "DocumentBackend"
    assert Chunker.__name__ == "Chunker"
    assert EvidenceItem.__name__ == "EvidenceItem"
    assert EvidenceUnitAgenticChunker.__name__ == "EvidenceUnitAgenticChunker"
    assert StageAgenticChunker is EvidenceUnitAgenticChunker
    assert StageRagChunkEnricher is RagChunkEnricher
    assert Enricher.__name__ == "Enricher"
    assert LlmConfig.__name__ == "LlmConfig"
    assert Hwp5Backend.supported_suffixes == (".hwp",)
    assert PdfBackend.supported_suffixes == (".pdf",)
    assert isinstance(backends[".hwp"], Hwp5Backend)
    assert isinstance(backends[".hwpx"], HwpxBackend)
    assert isinstance(backends[".md"], MarkdownBackend)
    assert isinstance(backends[".pdf"], PdfBackend)


def test_legacy_import_paths_are_removed():
    assert importlib.util.find_spec("rag_document_parser.backends") is None
    assert importlib.util.find_spec("rag_document_parser.evidence_html") is None
    assert importlib.util.find_spec("rag_document_parser.extract") is None
    assert importlib.util.find_spec("rag_document_parser.hwpx") is None
    assert importlib.util.find_spec("rag_document_parser.chunk.llm") is None
    assert importlib.util.find_spec("rag_document_parser.parser") is None
    assert importlib.util.find_spec("rag_document_parser.enrichment") is None
    assert importlib.util.find_spec("rag_document_parser.input") is None
