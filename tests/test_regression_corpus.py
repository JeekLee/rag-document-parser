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

    assert len(documents) == 11
    assert {str(document["id"]) for document in documents} == {
        "hwpx-ultrasound-qa",
        "pdf-ultrasound-qa",
        "hwpx-benefit-criteria-2024-278",
        "pdf-benefit-criteria-2024-278",
        "hwpx-cesarean-copay-qa",
        "pdf-cesarean-copay-qa",
        "hwpx-medical-fee-criteria-2022-139",
        "hwp-benefit-criteria-2009-214-annex",
        "hwp-medical-aid-overpayment-deduction",
        "pdf-medical-aid-overpayment-deduction",
        "pdf-infection-prevention-management-fee-qa",
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
        expected = document.get("expected", {})
        if isinstance(expected, dict) and expected.get("standalone_regression"):
            continue
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
    assert [column["text"] for column in benefit_table.content["columns"]] == [
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
    assert [column["text"] for column in cesarean_table.content["columns"]] == [
        "연번",
        "본인부담률 인하(5%) 관련 / 질의",
        "본인부담률 인하(5%) 관련 / 답변",
        "본인부담률 개정(5%→0%) 관련 / 개정(안)",
    ]

    ultrasound_path = CORPUS_DIR / str(documents["hwpx-ultrasound-qa"]["path"])
    ultrasound = HwpxBackend().parse(ultrasound_path.read_bytes(), ".hwpx")
    upper_abdomen = next(
        unit
        for unit in ultrasound.units
        if unit.type == "table" and "1 기존 90번" in unit.source.text
    )
    code_table = upper_abdomen.content["rows"][0]["cells"][2]["children"][0]["content"]
    assert [column["text"] for column in code_table["columns"]] == [
        "구분",
        "구분",
        "EDI코드",
    ]
    assert len(code_table["header_rows"]) == 1


def test_supported_hwpx_corpus_uses_common_evidence_contracts():
    from rag_document_parser import HwpxBackend

    documents = [
        document
        for document in _manifest_documents()
        if document["format"] == "hwpx" and document["parser_supported"]
    ]
    assert documents

    for document in documents:
        path = CORPUS_DIR / str(document["path"])
        expected = document["expected"]
        parsed = HwpxBackend().parse(path.read_bytes(), ".hwpx")

        assert isinstance(parsed.quality_warnings, list), document["id"]
        for warning in parsed.quality_warnings:
            assert warning["type"].startswith("hwpx_"), document["id"]
            assert warning["message"], document["id"]

        for unit in parsed.units:
            assert unit.type in {"text", "table", "image", "diagram"}, document["id"]
            assert "common" in unit.metadata, document["id"]
            if unit.type != "diagram":
                continue
            assert unit.format == "structured_diagram", document["id"]
            assert set(unit.content) >= {
                "nodes",
                "edges",
                "connectors",
                "mermaid",
            }, document["id"]
            assert unit.content["mermaid"] is None, document["id"]
        assert (
            sum(1 for unit in parsed.units if unit.type == "diagram")
            >= expected.get("min_diagram_units", 0)
        ), document["id"]


def test_hwpx_minio_diagram_fixture_extracts_real_file_diagram_evidence():
    from rag_document_parser import HwpxBackend

    path = CORPUS_DIR / "hwpx" / "medical-fee-criteria-2022-139.hwpx"
    assert path.is_file()

    parsed = HwpxBackend().parse(path.read_bytes(), ".hwpx")
    diagram = next(
        unit
        for unit in parsed.units
        if unit.type == "diagram" and "조산아 및 저체중 출생아 등록절차" in unit.source.text
    )

    assert diagram.format == "structured_diagram"
    assert diagram.source.kind == "diagram"
    node_texts = [str(node["text"]) for node in diagram.content["nodes"]]
    assert "조산아 및 저체중 출생아 등록절차" in node_texts
    assert "의료기관" in node_texts
    assert "건강보험심사평가원" in node_texts
    assert "시군구 (읍면동 포함)" in node_texts
    assert "지원대상자" in node_texts
    assert len(diagram.content["edges"]) >= 4
    assert all(edge["confidence"] == "inferred_table_grid" for edge in diagram.content["edges"])
    assert all(edge["connector_id"] for edge in diagram.content["edges"])
    assert len(diagram.content["connectors"]) >= 4
    assert diagram.content["mermaid"] is None


def test_hwpx_minio_diagram_fixture_preserves_flowchart_geometry():
    from rag_document_parser import HwpxBackend

    path = CORPUS_DIR / "hwpx" / "medical-fee-criteria-2022-139.hwpx"
    assert path.is_file()

    parsed = HwpxBackend().parse(path.read_bytes(), ".hwpx")
    diagram = next(
        unit
        for unit in parsed.units
        if unit.type == "diagram" and "조산아 및 저체중 출생아 등록절차" in unit.source.text
    )

    nodes_by_text = {
        str(node["text"]): node
        for node in diagram.content["nodes"]
    }
    assert nodes_by_text["조산아 및 저체중 출생아 등록절차"]["bbox"] == {
        "x": 0,
        "y": 0,
        "width": 23,
        "height": 1,
        "unit": "hwpx_table_grid",
    }
    assert nodes_by_text["건강보험공단 (자격관리시스템)"]["bbox"] == {
        "x": 6,
        "y": 2,
        "width": 6,
        "height": 2,
        "unit": "hwpx_table_grid",
    }
    assert nodes_by_text["지원대상자"]["bbox"] == {
        "x": 13,
        "y": 11,
        "width": 4,
        "height": 2,
        "unit": "hwpx_table_grid",
    }
    assert "④자료전송" in nodes_by_text
    assert nodes_by_text["④자료전송"]["bbox"] is None

    connectors = diagram.content["connectors"]
    assert len(connectors) >= 8
    assert connectors[0] == {
        "id": "c1",
        "type": "arrow",
        "bbox": {
            "x": 13,
            "y": 2,
            "width": 7,
            "height": 1,
            "unit": "hwpx_table_grid",
        },
        "points": [{"x": 20, "y": 2.5}, {"x": 13, "y": 2.5}],
        "arrow": True,
        "metadata": {
            "source": "hwpx_table_flowchart",
            "label": "④자료전송",
            "raw_label": "← (④자료전송)",
        },
    }
    assert len(diagram.content["edges"]) >= 8
    assert all(edge["connector_id"] for edge in diagram.content["edges"])
    assert all(edge["confidence"] == "inferred_table_grid" for edge in diagram.content["edges"])
    assert "relations:" in diagram.source.text


def test_hwpx_minio_diagram_fixture_does_not_duplicate_flowchart_rows_in_table():
    from rag_document_parser import HwpxBackend

    path = CORPUS_DIR / "hwpx" / "medical-fee-criteria-2022-139.hwpx"
    assert path.is_file()

    parsed = HwpxBackend().parse(path.read_bytes(), ".hwpx")
    form_table = next(
        unit
        for unit in parsed.units
        if unit.type == "table"
        and "의료급여 2종 조산아 및 저체중출생아 등록 신청서" in unit.source.text
    )
    diagram = next(
        unit
        for unit in parsed.units
        if unit.type == "diagram"
        and "조산아 및 저체중 출생아 등록절차" in unit.source.text
    )

    assert "조산아 및 저체중 출생아 등록절차" not in form_table.source.text
    assert "건강보험공단 (자격관리시스템)" not in form_table.source.text
    assert "지원대상자" not in form_table.source.text
    assert len(form_table.content["rows"]) == 11
    assert "조산아 및 저체중 출생아 등록절차" in diagram.source.text
    assert "지원대상자" in diagram.source.text


def test_hwpx_medical_fee_form_table_preserves_blank_body_rows():
    from rag_document_parser import HwpxBackend

    path = CORPUS_DIR / "hwpx" / "medical-fee-criteria-2022-139.hwpx"
    assert path.is_file()

    parsed = HwpxBackend().parse(path.read_bytes(), ".hwpx")
    table = next(unit for unit in parsed.units if unit.id == "b705")

    assert len(table.content["columns"]) == 100
    assert len(table.content["header_rows"]) == 15
    assert len(table.content["rows"]) == 47
    assert table.metadata["table"]["row_count"] == 47
    assert all(
        row["index"] == index
        for index, row in enumerate(table.content["rows"], start=1)
    )
    assert not any(
        str(cell["text"]).strip() or cell["children"]
        for cell in table.content["rows"][1]["cells"]
    )
    assert "row 2:" not in table.source.text


def test_supported_hwp5_and_pdf_corpus_extracts_evidence_units():
    from rag_document_parser import Hwp5Backend, PdfBackend
    from rag_document_parser.models import StructuredDiagramContent, StructuredTableContent

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
            assert unit.type in {"text", "table", "image", "diagram"}, document["id"]
            assert "common" in unit.metadata, document["id"]
            if document["format"] == "pdf" and unit.type in {
                "table",
                "image",
                "diagram",
            }:
                assert "confidence" in unit.metadata.get("pdf", {}), document["id"]
        for unit in table_units:
            assert unit.source.text.startswith("table: "), document["id"]
            assert unit.format == "structured_table", document["id"]
            assert isinstance(unit.content, StructuredTableContent), document["id"]
            assert not isinstance(unit.content, dict), document["id"]
        for unit in parsed.units:
            if unit.type != "diagram":
                continue
            assert unit.format == "structured_diagram", document["id"]
            assert isinstance(unit.content, StructuredDiagramContent), document["id"]
            assert not isinstance(unit.content, dict), document["id"]
            assert "nodes" in unit.content, document["id"]


def test_hwp5_complex_annex_fixture_preserves_span_heavy_table():
    from rag_document_parser import Hwp5Backend

    documents = {document["id"]: document for document in _manifest_documents()}
    document = documents["hwp-benefit-criteria-2009-214-annex"]
    parsed = Hwp5Backend().parse(
        (CORPUS_DIR / str(document["path"])).read_bytes(),
        ".hwp",
    )

    table = next(unit for unit in parsed.units if unit.type == "table")
    content = table.content
    all_cells = [
        cell
        for row in [*content["header_rows"], *content["rows"]]
        for cell in row["cells"]
    ]
    span_cells = [
        cell
        for cell in all_cells
        if cell["rowspan"] > 1 or cell["colspan"] > 1
    ]

    assert [column["text"] for column in content["columns"]] == [
        "처방명",
        "현 행 / 한 방",
        "현 행 / 양 방",
        "현 행 / 분류기호",
        "개정(안) / 적응증",
    ]
    assert len(content["header_rows"]) == 2
    assert len(content["rows"]) == 424
    assert len(span_cells) >= 100
    assert "header 1: col 1: 처방명; cols 2-4: 현 행; col 5: 개정(안)" in (
        table.source.text
    )
    assert "row 1: 처방명: 1. 가미소요산" in table.source.text
    assert parsed.quality_warnings == []


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
    assert [len(unit.content["rows"]) for unit in benefit_tables] == [1, 1, 4]
    assert all(
        column["text"].strip()
        for unit in benefit_tables
        for column in unit.content["columns"]
    )
    assert any(
        [column["text"] for column in unit.content["columns"]]
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
    grouped_table = benefit_tables[2].content
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
        for row in benefit_tables[1].content["rows"]
        for cell in row["cells"]
        for child in cell["children"]
        if child.get("type", child.get("kind")) == "table"
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
    assert [len(unit.content["rows"]) for unit in cesarean_tables] == [2, 9]
    assert all(unit.content["rows"] for unit in cesarean_tables)
    assert [
        (cell["text"], cell["rowspan"], cell["colspan"])
        for cell in cesarean_tables[0].content["header_rows"][0]["cells"]
    ] == [
        ("연번", 2, 1),
        ("본인부담률 인하(5%) 관련", 1, 2),
        ("본인부담률 개정(5%→0%) 관련", 1, 1),
    ]
    assert [
        (cell["column_id"], cell["rowspan"], cell["colspan"])
        for cell in cesarean_tables[0].content["rows"][0]["cells"]
    ][:2] == [("c1", 2, 1), ("c2", 2, 1)]
    assert all(
        [column["text"] for column in unit.content["columns"]]
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
    assert [column["text"] for column in revision_table.content["columns"]] == [
        "개정일",
        "고시",
        "시행일",
        "관련 근거",
    ]
    assert "고시 제2016-149호" in revision_table.source.text
    assert "<하복부, 비뇨기 초음파 검사 급여기준 개선 관련 Q&A>" in (
        revision_table.source.text
    )
    toc_table = tables[1].content
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
    code_table = upper_abdomen.content["rows"][0]["cells"][2]["children"][0][
        "content"
    ]
    assert [column["text"] for column in code_table["columns"]] == [
        "구분",
        "구분",
        "EDI코드",
    ]
    assert [
        (cell["column_id"], cell["text"], cell["rowspan"], cell["colspan"])
        for cell in code_table["header_rows"][0]["cells"]
    ] == [
        ("c1", "구분", 1, 2),
        ("c3", "EDI코드", 1, 1),
    ]
    assert len(code_table["header_rows"]) == 1
    assert [
        (cell["column_id"], cell["text"], cell["rowspan"], cell["colspan"])
        for cell in code_table["rows"][0]["cells"]
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
            ("c1", "기본 초음파", 2, 1),
            ("c2", "단순초음파(Ⅰ)", 1, 1),
            ("c3", "EB401", 1, 1),
        ],
        [
            ("c2", "단순초음파(Ⅱ)", 1, 1),
            ("c3", "EB402", 1, 1),
        ],
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


def test_pdf_corpus_does_not_duplicate_nested_tables_as_diagrams():
    from rag_document_parser import PdfBackend

    documents = {document["id"]: document for document in _manifest_documents()}
    document = documents["pdf-infection-prevention-management-fee-qa"]
    parsed = PdfBackend().parse(
        (CORPUS_DIR / str(document["path"])).read_bytes(),
        ".pdf",
    )

    expected = document["expected"]
    assert len(parsed.units) >= expected["min_units"]
    assert (
        sum(1 for unit in parsed.units if unit.type == "table")
        >= expected["min_tables"]
    )
    assert len(parsed.assets) >= expected["min_assets"]
    assert not any(
        unit.type == "text" and str(unit.content).startswith("INSID")
        for unit in parsed.units
    )

    nested_table_children = []
    diagram_children = []
    for unit in parsed.units:
        if unit.type != "table":
            continue
        for row in unit.content["rows"]:
            for cell in row["cells"]:
                for child in cell.get("children", []):
                    if child.get("type") == "table":
                        nested_table_children.append((unit, cell, child))
                    if child.get("type") == "diagram":
                        diagram_children.append((unit, cell, child))

    assert nested_table_children
    nested_table_text_parts = []
    for _, _, child in nested_table_children:
        content = child["content"]
        nested_table_text_parts.extend(
            str(column["text"])
            for column in content.get("columns", [])
        )
        for row in content.get("rows", []):
            nested_table_text_parts.extend(
                str(cell["text"])
                for cell in row.get("cells", [])
            )
    nested_table_text = " ".join(nested_table_text_parts)
    assert "최초운영일" in nested_table_text
    assert "최초분기 적용기준일" in nested_table_text
    assert "차기분기 적용기준일" in nested_table_text

    first_qa_table = next(
        unit
        for unit in parsed.units
        if unit.type == "table"
        and any(
            "감염관리 의사와 감염관리 전담간호사" in str(cell["text"])
            for row in unit.content["rows"]
            for cell in row["cells"]
        )
    )
    answer_cell = first_qa_table.content["rows"][0]["cells"][2]
    answer_tables = [
        child for child in answer_cell["children"] if child.get("type") == "table"
    ]
    assert len(answer_tables) == 3
    assert [
        [column["text"] for column in child["content"]["columns"]]
        for child in answer_tables
    ] == [
        [
            "최초운영일이 속한 최초분기 등급",
            "최초운영분기의 차기분기 등급",
        ],
        [
            "최초운영일",
            "최초운영분기",
            "최초분기 적용기준일 / 인력",
            "최초분기 적용기준일 / 병상수",
        ],
        [
            "최초운영일",
            "차기적용분기",
            "차기분기 적용기준일 / 인력",
            "차기분기 적용기준일 / 병상수",
        ],
    ]
    for child in answer_tables[1:]:
        header_rows = child["content"]["header_rows"]
        assert [
            (cell["column_id"], cell["text"], cell["rowspan"], cell["colspan"])
            for cell in header_rows[0]["cells"]
        ] == [
            ("c1", child["content"]["columns"][0]["text"], 2, 1),
            ("c2", child["content"]["columns"][1]["text"], 2, 1),
            ("c3", child["content"]["columns"][2]["text"].split(" / ")[0], 1, 2),
        ]
        assert [
            (cell["column_id"], cell["text"], cell["rowspan"], cell["colspan"])
            for cell in header_rows[1]["cells"]
        ] == [
            ("c3", "인력", 1, 1),
            ("c4", "병상수", 1, 1),
        ]
    assert all(
        len(row["cells"]) == len(child["content"]["columns"])
        for child in answer_tables
        for row in child["content"]["rows"]
    )

    assert diagram_children == []
    assert not any(
        asset.metadata.get("source") == "table_cell_diagram_fallback"
        for asset in parsed.assets
    )
    assert not any(
        warning["type"] == "pdf_table_cell_diagram_inferred"
        for warning in parsed.quality_warnings
    )
    assert not any(
        "diagram: 최초운영일" in unit.source.text
        for unit in parsed.units
        if unit.type == "table"
    )


def test_pdf_corpus_merges_continued_table_rows_across_artifact_text():
    from rag_document_parser import PdfBackend

    documents = {document["id"]: document for document in _manifest_documents()}
    document = documents["pdf-infection-prevention-management-fee-qa"]
    parsed = PdfBackend().parse(
        (CORPUS_DIR / str(document["path"])).read_bytes(),
        ".pdf",
    )

    qa_tables = [
        unit
        for unit in parsed.units
        if unit.type == "table"
        and [column["text"] for column in unit.content["columns"]]
        == ["연번", "질의", "답변"]
    ]
    assert len(qa_tables) == 2
    assert not any(
        row["cells"][0]["text"] == ""
        and row["cells"][1]["text"] == ""
        and row["cells"][2]["text"] == "의미함"
        for table in qa_tables
        for row in table.content["rows"]
    )

    row_five = next(
        row
        for table in qa_tables
        for row in table.content["rows"]
        if row["cells"][0]["text"] == "5"
        and "감염관리지침" in row["cells"][1]["text"]
    )
    assert row_five["cells"][2]["text"].endswith("활동을\n의미함")


def test_pdf_real_scanned_fixture_restores_aligned_ocr_table():
    from rag_document_parser import PdfBackend

    def _ocr_text(_png: bytes, page_idx: int) -> str:
        if page_idx == 0:
            return (
                "구분    금액    비고\n"
                "외래    1000    당일\n"
                "입원    2000    익일"
            )
        return f"본문 {page_idx}"

    parsed = PdfBackend(
        max_ocr_workers=1,
        ocr_fn=_ocr_text,
    ).parse(
        (CORPUS_DIR / "pdf/medical-aid-overpayment-deduction.pdf").read_bytes(),
        ".pdf",
    )

    assert [unit.type for unit in parsed.units] == ["table", "text", "text"]
    table = next(unit for unit in parsed.units if unit.type == "table")
    assert [column["text"] for column in table.content["columns"]] == [
        "구분",
        "금액",
        "비고",
    ]
    assert [
        [cell["text"] for cell in row["cells"]]
        for row in table.content["rows"]
    ] == [["외래", "1000", "당일"], ["입원", "2000", "익일"]]
    assert table.metadata["pdf"]["ocr"] is True
    assert parsed.quality_warnings == [
        {
            "type": "pdf_ocr_table_inferred",
            "severity": "low",
            "page": 1,
            "message": "OCR text was converted to structured_table from a detected text table.",
        }
    ]
