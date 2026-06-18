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


def test_scanned_cover_letter_pdf_is_not_marked_hwp_body_comparable():
    documents = {document["id"]: document for document in _manifest_documents()}
    expected = documents["pdf-medical-aid-overpayment-deduction"]["expected"]

    assert expected["pair_comparable"] is False
    assert expected["comparison_scope"] == "scanned_cover_letter_pages"


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


def test_pdf_corpus_preserves_grouped_headers_and_reduces_fragmentation():
    from rag_document_parser import PdfBackend

    documents = {document["id"]: document for document in _manifest_documents()}
    backend = PdfBackend()

    benefit = backend.parse(
        (CORPUS_DIR / str(documents["pdf-benefit-criteria-2024-278"]["path"])).read_bytes(),
        ".pdf",
    )
    benefit_tables = [unit for unit in benefit.units if unit.type == "table"]
    assert len(benefit.units) == 14
    assert len(benefit_tables) <= 4
    assert [len(unit.evidence.content["rows"]) for unit in benefit_tables] == [1, 1, 4]
    assert all(
        column["text"].strip()
        for unit in benefit_tables
        for column in unit.evidence.content["columns"]
    )
    assert any(
        [column["text"] for column in unit.evidence.content["columns"]]
        == [
            "현행 / 항목",
            "현행 / 제목",
            "현행 / 세부인정사항",
            "개정 / 항목",
            "개정 / 제목",
            "개정 / 세부인정사항",
            "비고",
        ]
        for unit in benefit_tables
    )
    grouped_table = benefit_tables[2].evidence.content
    assert [
        (cell["text"], cell["rowspan"], cell["colspan"])
        for cell in grouped_table["header_rows"][0]["cells"]
    ] == [
        ("현행", 1, 3),
        ("개정", 1, 3),
        ("비고", 2, 1),
    ]
    assert [
        (cell["column_id"], cell["text"], cell["colspan"])
        for cell in grouped_table["rows"][0]["cells"]
    ] == [
        ("c1", "I. 행위 일반사항", 3),
        ("c4", "I. 행위 일반사항", 4),
    ]
    assert "개정 / 비고: I. 행위 일반사항" in benefit_tables[2].source.text
    assert not any(
        "col 2:" in line
        for unit in benefit_tables
        for line in unit.source.text.splitlines()
        if line.startswith("row ")
    )
    disease_table = next(
        child["content"]
        for row in benefit_tables[1].evidence.content["rows"]
        for cell in row["cells"]
        for child in cell["children"]
        if child["kind"] == "table"
    )
    assert [column["text"] for column in disease_table["columns"]] == [
        "질병코드 / A04.7",
        "질병코드 / G83.4",
        "질병코드 / L89.2",
        "질병코드 / M86~M87",
        "질병코드 / S38.1",
    ]
    assert len(disease_table["header_rows"]) == 2
    assert len(disease_table["rows"]) == 16

    cesarean = backend.parse(
        (CORPUS_DIR / str(documents["pdf-cesarean-copay-qa"]["path"])).read_bytes(),
        ".pdf",
    )
    cesarean_tables = [unit for unit in cesarean.units if unit.type == "table"]
    assert len(cesarean.units) == 11
    assert len(cesarean_tables) == 2
    assert [len(unit.evidence.content["rows"]) for unit in cesarean_tables] == [2, 9]
    assert all(unit.evidence.content["rows"] for unit in cesarean_tables)
    assert [
        (cell["text"], cell["rowspan"], cell["colspan"])
        for cell in cesarean_tables[0].evidence.content["header_rows"][0]["cells"]
    ] == [
        ("연번", 2, 1),
        ("본인부담률 인하(5%) 관련", 1, 2),
        ("본인부담률 개정(5%→0%) 관련", 1, 1),
    ]
    assert [
        (cell["column_id"], cell["rowspan"], cell["colspan"])
        for cell in cesarean_tables[0].evidence.content["rows"][0]["cells"]
    ][:2] == [("c1", 2, 1), ("c2", 2, 1)]
    assert all(
        [column["text"] for column in unit.evidence.content["columns"]]
        == [
            "연번",
            "본인부담률 인하(5%) 관련 / 질의",
            "본인부담률 인하(5%) 관련 / 답변",
            "본인부담률 개정(5%→0%) 관련 / 개정(안)",
        ]
        for unit in cesarean_tables
    )
    assert not any(
        "col 3:" in line
        for unit in cesarean_tables
        for line in unit.source.text.splitlines()
        if line.startswith("row ")
    )
    assert not any(
        unit.type == "text" and unit.source.text in {"질병군 적용 대상", "행위 적용 대상"}
        for unit in cesarean.units
    )


def test_pdf_ultrasound_promotes_revision_history_text_to_table():
    from rag_document_parser import PdfBackend

    documents = {document["id"]: document for document in _manifest_documents()}
    parsed = PdfBackend().parse(
        (CORPUS_DIR / str(documents["pdf-ultrasound-qa"]["path"])).read_bytes(),
        ".pdf",
    )
    tables = [unit for unit in parsed.units if unit.type == "table"]

    assert [unit.type for unit in parsed.units[:3]] == ["text", "text", "table"]
    assert parsed.units[0].source.text == "초음파 검사 질의응답"
    assert parsed.units[1].source.text == "2024. 2."
    revision_table = tables[0]
    assert revision_table.metadata["table"]["row_count"] == 14
    assert [column["text"] for column in revision_table.evidence.content["columns"]] == [
        "개정일",
        "고시",
        "시행일",
        "관련 근거",
    ]
    assert "고시 제2016-149호" in revision_table.source.text
    assert "<하복부, 비뇨기 초음파 검사 급여기준 개선 관련 Q&A>" in (
        revision_table.source.text
    )
    toc_table = tables[1].evidence.content
    assert [cell["text"] for cell in toc_table["rows"][0]["cells"]] == [
        "1",
        "일반사항",
        "1",
    ]
    assert [cell["text"] for cell in toc_table["rows"][-1]["cells"]] == [
        "15",
        "경부 초음파 관련 Q&A",
        "39",
    ]
    upper_abdomen = next(
        unit
        for unit in parsed.units
        if unit.type == "table" and "1 기존 90번" in unit.source.text
    )
    code_table = upper_abdomen.evidence.content["rows"][0]["cells"][2]["children"][0][
        "content"
    ]
    assert [
        (cell["column_id"], cell["text"], cell["rowspan"], cell["colspan"])
        for cell in code_table["header_rows"][1]["cells"]
    ] == [
        ("c1", "기본 초음파", 2, 1),
        ("c2", "단순초음파(Ⅰ)", 1, 1),
        ("c3", "EB401", 1, 1),
    ]
    assert [
        [
            (cell["column_id"], cell["text"], cell["rowspan"], cell["colspan"])
            for cell in row["cells"]
        ]
        for row in code_table["rows"]
    ] == [
        [
            ("c1", "진단 초음파", 2, 1),
            ("c2", "간·담낭·담도·비장·췌장(일반)", 1, 1),
            ("c3", "EB441", 1, 1),
        ],
        [
            ("c2", "간·담낭·담도·비장·췌장(정밀)", 1, 1),
            ("c3", "EB442", 1, 1),
        ],
        [
            ("c1", "제한적 초음파", 2, 1),
            ("c2", "간·담낭·담도·비장·췌장(일반)", 1, 1),
            ("c3", "EB441001", 1, 1),
        ],
        [
            ("c2", "간·담낭·담도·비장·췌장(정밀)", 1, 1),
            ("c3", "EB442001", 1, 1),
        ],
    ]
