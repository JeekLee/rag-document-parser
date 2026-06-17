from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


CORPUS_DIR = Path(__file__).parent / "fixtures" / "corpus"
MANIFEST_PATH = CORPUS_DIR / "manifest.json"


def _manifest_documents() -> list[dict[str, object]]:
    assert MANIFEST_PATH.is_file()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return manifest["documents"]


def test_regression_corpus_files_are_pinned_by_hash_and_size():
    documents = _manifest_documents()

    assert len(documents) == 8
    assert {str(document["id"]) for document in documents} == {
        "hwpx-ultrasound-qa",
        "pdf-ultrasound-qa",
        "hwpx-benefit-criteria-2024-278",
        "pdf-benefit-criteria-2024-278",
        "hwpx-cesarean-copay-qa",
        "pdf-cesarean-copay-qa",
        "hwp-medical-aid-overpayment-deduction",
        "pdf-medical-aid-overpayment-deduction",
    }
    for document in documents:
        path = CORPUS_DIR / str(document["path"])

        assert path.is_file(), document["id"]
        assert path.suffix in {".hwp", ".hwpx", ".pdf"}, document["id"]
        data = path.read_bytes()
        assert len(data) == document["bytes"], document["id"]
        assert hashlib.sha256(data).hexdigest() == document["sha256"], document["id"]


def test_regression_corpus_is_paired_with_pdf_counterparts():
    documents = _manifest_documents()
    pairs: dict[str, set[str]] = {}
    for document in documents:
        pair_id = str(document["pair_id"])
        pairs.setdefault(pair_id, set()).add(str(document["format"]))

    assert len(pairs) == 4
    assert pairs == {
        "ultrasound-qa": {"hwpx", "pdf"},
        "benefit-criteria-2024-278": {"hwpx", "pdf"},
        "cesarean-copay-qa": {"hwpx", "pdf"},
        "medical-aid-overpayment-deduction": {"hwp", "pdf"},
    }


def test_supported_hwpx_corpus_emits_canonical_table_source():
    from rag_document_parser import HwpxBackend

    documents = [
        document
        for document in _manifest_documents()
        if document["format"] == "hwpx" and document["parser_supported"]
    ]
    assert documents
    for document in documents:
        path = CORPUS_DIR / str(document["path"])
        parsed = HwpxBackend().parse(path.read_bytes(), ".hwpx")

        table_units = [unit for unit in parsed.units if unit.type == "table"]
        expected = document["expected"]
        assert len(parsed.units) >= expected["min_units"], document["id"]
        assert len(table_units) >= expected["min_tables"], document["id"]

        for unit in table_units:
            source = unit.source.text
            assert source.startswith("table: "), document["id"]
            assert "\ncolumns:" not in source, document["id"]
            assert "header row" not in source, document["id"]
            assert not re.search(r"\bcol \d+-col \d+\b", source), document["id"]

        if expected.get("requires_colspan_source"):
            assert any("cols " in unit.source.text for unit in table_units), (
                document["id"]
            )


def test_supported_hwpx_corpus_preserves_grouped_header_context():
    from rag_document_parser import HwpxBackend

    documents = {document["id"]: document for document in _manifest_documents()}
    benefit_path = CORPUS_DIR / str(
        documents["hwpx-benefit-criteria-2024-278"]["path"]
    )
    benefit = HwpxBackend().parse(
        benefit_path.read_bytes(),
        ".hwpx",
    )
    benefit_table = [unit for unit in benefit.units if unit.type == "table"][2]
    assert [column["text"] for column in benefit_table.evidence.content["columns"]] == [
        "현   행 / 항목",
        "현   행 / 제목",
        "현   행 / 세부인정사항",
        "개   정 / 항목",
        "개   정 / 제목",
        "개   정 / 세부인정사항",
        "비고",
    ]
    assert "row 1: 현   행: I. 행위 일반사항; 개   정: I. 행위 일반사항" in (
        benefit_table.source.text
    )

    cesarean_path = CORPUS_DIR / str(documents["hwpx-cesarean-copay-qa"]["path"])
    cesarean = HwpxBackend().parse(
        cesarean_path.read_bytes(),
        ".hwpx",
    )
    cesarean_table = [unit for unit in cesarean.units if unit.type == "table"][0]
    assert [column["text"] for column in cesarean_table.evidence.content["columns"]] == [
        "연번",
        "본인부담률 인하(5%) 관련 / 질의",
        "본인부담률 인하(5%) 관련 / 답변",
        "본인부담률 개정(5%→0%) 관련 / 개정(안)",
    ]


def test_supported_hwp5_and_pdf_corpus_extracts_evidence_units():
    from rag_document_parser import Hwp5Backend, PdfBackend

    documents = [
        document
        for document in _manifest_documents()
        if document["format"] in {"hwp", "pdf"} and document["parser_supported"]
    ]
    assert documents

    for document in documents:
        path = CORPUS_DIR / str(document["path"])
        expected = document["expected"]
        backend = (
            Hwp5Backend()
            if document["format"] == "hwp"
            else PdfBackend(
                max_ocr_workers=2,
                ocr_fn=(
                    (lambda png, page_idx: f"OCR page {page_idx + 1}")
                    if expected.get("requires_ocr")
                    else None
                ),
            )
        )

        parsed = backend.parse(path.read_bytes(), f".{document['format']}")
        table_units = [unit for unit in parsed.units if unit.type == "table"]

        assert len(parsed.units) >= expected["min_units"], document["id"]
        assert len(table_units) >= expected["min_tables"], document["id"]
        assert len(parsed.assets) >= expected.get("min_assets", 0), document["id"]
        assert (
            sum(
                1
                for unit in parsed.units
                if unit.metadata.get("common", {}).get("chunk_kind") == "diagram"
            )
            >= expected.get("min_diagram_units", 0)
        ), document["id"]
        if expected.get("requires_ocr"):
            assert len(parsed.units) == expected["scanned_pages"], document["id"]
            assert all(unit.type == "text" for unit in parsed.units), document["id"]
            assert parsed.quality_warnings == [], document["id"]

        for unit in parsed.units:
            assert unit.source.text, document["id"]
            assert unit.evidence.kind == unit.type, document["id"]
            assert "common" in unit.metadata, document["id"]
        for unit in table_units:
            assert unit.source.text.startswith("table: "), document["id"]
            assert unit.evidence.format == "structured_table", document["id"]
            assert isinstance(unit.evidence.content, dict), document["id"]
