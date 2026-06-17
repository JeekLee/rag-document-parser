from __future__ import annotations


def test_pipeline_layout_exports_stage_and_format_modules():
    from rag_document_parser import (
        Hwp5Backend,
        HwpxBackend,
        MarkdownBackend,
        PdfBackend,
        PdfOcrConfig,
        RagDocumentParser,
    )
    from rag_document_parser.chunk.backend import Chunker
    from rag_document_parser.enrichment.backend import Enricher
    from rag_document_parser.enrichment.llm import LlmConfig
    from rag_document_parser.extract.backend import DocumentBackend, ParsedDocument
    from rag_document_parser.extract.formats.hwp5.backend import Hwp5Backend as StageHwp5Backend
    from rag_document_parser.extract.formats.hwpx.backend import HwpxBackend as StageHwpxBackend
    from rag_document_parser.extract.formats.markdown.backend import (
        MarkdownBackend as StageMarkdownBackend,
    )
    from rag_document_parser.extract.formats.pdf.backend import (
        PdfBackend as StagePdfBackend,
        PdfOcrConfig as StagePdfOcrConfig,
    )
    from rag_document_parser.extract.registry import default_backends
    from rag_document_parser.pipeline.parser import RagDocumentParser as StageParser

    backends = default_backends()

    assert StageParser is RagDocumentParser
    assert StageHwp5Backend is Hwp5Backend
    assert StageHwpxBackend is HwpxBackend
    assert StageMarkdownBackend is MarkdownBackend
    assert StagePdfBackend is PdfBackend
    assert StagePdfOcrConfig is PdfOcrConfig
    assert ParsedDocument.__name__ == "ParsedDocument"
    assert DocumentBackend.__name__ == "DocumentBackend"
    assert Chunker.__name__ == "Chunker"
    assert Enricher.__name__ == "Enricher"
    assert LlmConfig.__name__ == "LlmConfig"
    assert Hwp5Backend.supported_suffixes == (".hwp",)
    assert PdfBackend.supported_suffixes == (".pdf",)
    assert isinstance(backends[".hwp"], Hwp5Backend)
    assert isinstance(backends[".hwpx"], HwpxBackend)
    assert isinstance(backends[".md"], MarkdownBackend)
    assert isinstance(backends[".pdf"], PdfBackend)


def test_legacy_import_paths_remain_compatible():
    from rag_document_parser import HwpxBackend, MarkdownBackend, RagDocumentParser
    from rag_document_parser.backends import MarkdownBackend as LegacyMarkdownBackend
    from rag_document_parser.backends import ParsedDocument as LegacyParsedDocument
    from rag_document_parser.extract.backend import ParsedDocument
    from rag_document_parser.hwpx import HwpxBackend as LegacyHwpxBackend
    from rag_document_parser.llm import LlmConfig as LegacyLlmConfig
    from rag_document_parser.parser import RagDocumentParser as LegacyParser
    from rag_document_parser.enrichment.llm import LlmConfig

    assert LegacyParser is RagDocumentParser
    assert LegacyHwpxBackend is HwpxBackend
    assert LegacyMarkdownBackend is MarkdownBackend
    assert LegacyParsedDocument is ParsedDocument
    assert LegacyLlmConfig is LlmConfig
