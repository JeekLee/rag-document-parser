from __future__ import annotations

import importlib.util
from typing import get_type_hints


def test_pipeline_layout_exports_stage_and_format_modules():
    import rag_document_parser
    from rag_document_parser.evidence_unit_extraction.formats import pdf as pdf_format
    from rag_document_parser import (
        EvidenceItem,
        EvidenceUnitAgenticChunker,
        GeminiLlmConfig,
        GemmaLlmConfig,
        Hwp5Backend,
        HtmlBackend,
        HwpxBackend,
        LlmConfig,
        MarkdownBackend,
        PdfBackend,
        QwenLlmConfig,
        RagChunkEnricher,
        RagDocumentParser,
    )
    from rag_document_parser.chunk import EvidenceUnitAgenticChunker as StageAgenticChunker
    from rag_document_parser.chunk.backend import Chunker
    from rag_document_parser.chunk.enrichment import Enricher
    from rag_document_parser.chunk.enrichment import RagChunkEnricher as StageRagChunkEnricher
    from rag_document_parser.evidence_unit_extraction.backend import DocumentBackend
    from rag_document_parser.evidence_unit_extraction.formats.hwp5.backend import (
        Hwp5Backend as StageHwp5Backend,
    )
    from rag_document_parser.evidence_unit_extraction.formats.html.backend import (
        HtmlBackend as StageHtmlBackend,
    )
    from rag_document_parser.evidence_unit_extraction.formats.hwpx.backend import (
        HwpxBackend as StageHwpxBackend,
    )
    from rag_document_parser.evidence_unit_extraction.formats.markdown.backend import (
        MarkdownBackend as StageMarkdownBackend,
    )
    from rag_document_parser.evidence_unit_extraction.formats.pdf.backend import (
        PdfBackend as StagePdfBackend,
    )
    from rag_document_parser.evidence_unit_extraction.registry import default_backends
    from rag_document_parser.models import ParsedDocument
    from rag_document_parser.pipeline.parser import RagDocumentParser as StageParser

    backends = default_backends()

    assert StageParser is RagDocumentParser
    assert StageHwp5Backend is Hwp5Backend
    assert StageHtmlBackend is HtmlBackend
    assert StageHwpxBackend is HwpxBackend
    assert StageMarkdownBackend is MarkdownBackend
    assert StagePdfBackend is PdfBackend
    assert not hasattr(rag_document_parser, "PdfOcrConfig")
    assert not hasattr(pdf_format, "PdfOcrConfig")
    assert get_type_hints(PdfBackend)["ocr_llm"] == LlmConfig | None
    assert ParsedDocument.__name__ == "ParsedDocument"
    assert DocumentBackend.__name__ == "DocumentBackend"
    assert Chunker.__name__ == "Chunker"
    assert EvidenceItem.__name__ == "EvidenceItem"
    assert EvidenceUnitAgenticChunker.__name__ == "EvidenceUnitAgenticChunker"
    assert StageAgenticChunker is EvidenceUnitAgenticChunker
    assert StageRagChunkEnricher is RagChunkEnricher
    assert Enricher.__name__ == "Enricher"
    assert LlmConfig.__name__ == "LlmConfig"
    assert GeminiLlmConfig.__name__ == "GeminiLlmConfig"
    assert GemmaLlmConfig.__name__ == "GemmaLlmConfig"
    assert QwenLlmConfig.__name__ == "QwenLlmConfig"
    assert Hwp5Backend.supported_suffixes == (".hwp",)
    assert HtmlBackend.supported_suffixes == (".html", ".htm")
    assert PdfBackend.supported_suffixes == (".pdf",)
    assert isinstance(backends[".hwp"], Hwp5Backend)
    assert isinstance(backends[".html"], HtmlBackend)
    assert isinstance(backends[".htm"], HtmlBackend)
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
