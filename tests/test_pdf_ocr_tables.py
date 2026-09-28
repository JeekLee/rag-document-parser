from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from rag_document_parser import OcrResult, PdfBackend, RagDocumentParser


@pytest.fixture(scope="module")
def scanned_pdf() -> bytes:
    output = BytesIO()
    pdf = canvas.Canvas(output, pagesize=(200, 200))
    image = ImageReader(Image.new("RGB", (200, 200), "white"))
    for _ in range(2):
        pdf.drawImage(image, 0, 0, 200, 200)
        pdf.showPage()
    pdf.save()
    return output.getvalue()


@pytest.fixture(params=["backend", "parser"])
def parse_ocr(request, scanned_pdf):
    def parse(
        text: str, *, legacy: bool = False, first_page_text: str = "앞 페이지 본문"
    ):
        calls = []

        def cached_ocr(image: bytes, page_index: int):
            assert image.startswith(b"\x89PNG")
            calls.append(page_index)
            page_text = text if page_index == 1 else first_page_text
            if legacy:
                return page_text
            return OcrResult(status="extracted", text=page_text, reason="test fixture")

        backend = PdfBackend(ocr_fn=cached_ocr, max_ocr_workers=1)
        if request.param == "backend":
            result = backend.parse(scanned_pdf, ".pdf")
        else:
            result = RagDocumentParser(backends={".pdf": backend}).parse(
                scanned_pdf, suffix=".pdf", source_name="cached-ocr.pdf"
            )
            assert result.source.name == "cached-ocr.pdf"
        assert calls == [0, 1]
        return result

    return parse


@pytest.mark.parametrize(
    ("headers", "rows"),
    [
        (["구분", "판단"], [["대상", "반려", "허가 및 안전성 판단 배경"]]),
        (
            ["구분", "입·퇴원 시기", "입원 본인부담률"],
            [
                [
                    "신생아",
                    "정상아",
                    "시행일 이전 입원·이후 퇴원",
                    "동일 입원기간 전부 면제",
                ]
            ],
        ),
        (
            ["항목", "대상", "시기", "비고"],
            [
                ["A", "B", "C", "D", "24.1.1. 0시부터 종료"],
                ["E", "F", "G", "H", "시범사업", "별도 조건"],
            ],
        ),
        (["구분", "조건", "기간"], [["대상", "중간 셀 누락 위치 불명"]]),
        (["구분", "조건"], [["하나의 셀만 판독됨"]]),
    ],
    ids=["2-3", "3-4", "4-5-6", "short-row", "single-cell-row"],
)
def test_ocr_table_width_mismatch_preserves_complete_source(parse_ocr, headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    text = "\n".join(lines)
    result = parse_ocr(text)
    fallback = result.units[1]

    assert len(result.units) == 2
    assert fallback.format == "plain"
    assert fallback.content == text
    assert fallback.source.text == text
    assert fallback.metadata["pdf"] == {"page": 2, "ocr": True, "table_fallback": True}
    (warning,) = result.quality_warnings
    assert warning.type == "pdf_ocr_table_unstructured"
    assert warning.severity == "medium"
    assert warning.details["page"] == 2
    assert warning.details["reason"] == "column_count_mismatch"
    assert warning.details["line_column_counts"] == [
        len(headers),
        len(headers),
        *map(len, rows),
    ]
    assert result.to_dict()["quality_warnings"][0]["page"] == 2


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("| A | B |\n| --- | --- | --- |\n| one | two |", "column_count_mismatch"),
        ("| A || C |\n| --- | --- | --- |\n| one | two | three |", "ambiguous_header"),
        ("| 상위 헤더 ||\n| A | B |\n| --- | --- |\n| one | two |", "ambiguous_header"),
        ("| A | B |\n| --- | --- |", "missing_body"),
        ("| A | B |\n| --- | --- |\n| --- | --- |", "unexpected_separator"),
    ],
    ids=[
        "separator-width",
        "empty-header",
        "merged-header",
        "no-body",
        "extra-separator",
    ],
)
def test_ocr_ambiguous_headers_preserve_text_with_reason(parse_ocr, text, reason):
    result = parse_ocr(text)
    assert result.units[1].format == "plain"
    assert result.units[1].source.text == text
    assert result.quality_warnings[0].details["reason"] == reason


@pytest.mark.parametrize(
    "legacy", [False, True], ids=["model-callback", "string-callback"]
)
def test_ocr_empty_and_escaped_cells_keep_their_columns(parse_ocr, legacy):
    text = (
        "| 구분 | 조건 | 기간 |\n"
        "| :--- | ---: | :---: |\n"
        "|신생아|정상아|전부 면제|\n"
        "||영유아가 아닌 때부터|20%|\n"
        "|외래||당일|\n"
        "|기타|확인 필요||\n"
        "|기호|A\\|B|원문|"
    )
    result = parse_ocr(text, legacy=legacy)
    unit = result.units[1]
    assert unit.format == "structured_table"
    assert [c.text for c in unit.content.columns] == ["구분", "조건", "기간"]
    assert [[c.text for c in row.cells] for row in unit.content.rows] == [
        ["신생아", "정상아", "전부 면제"],
        ["", "영유아가 아닌 때부터", "20%"],
        ["외래", "", "당일"],
        ["기타", "확인 필요", ""],
        ["기호", "A|B", "원문"],
    ]
    assert all(
        c.column_id == f"c{i}" and c.rowspan == c.colspan == 1
        for row in unit.content.rows
        for i, c in enumerate(row.cells, 1)
    )
    assert unit.metadata["pdf"] == {"page": 2, "ocr": True, "confidence": "medium"}
    assert [w.type for w in result.quality_warnings] == ["pdf_ocr_table_inferred"]


@pytest.mark.parametrize("gap", ["\n", "\n\n"], ids=["adjacent", "blank-line"])
@pytest.mark.parametrize(
    "same_headers", [False, True], ids=["different-headers", "same-headers"]
)
def test_ocr_adjacent_tables_keep_separate_header_mappings(
    parse_ocr, gap, same_headers
):
    first = "| 항목 | 금액 |\n| --- | --- |\n| 외래 | 1000 |"
    second = (
        "| 항목 | 금액 |\n| --- | --- |\n| 입원 | 2000 |"
        if same_headers
        else "| 코드 | 설명 | 근거 |\n| --- | --- | --- |\n| JT001 | 설명 보존 | 개정 사유 |"
    )
    result = parse_ocr(first + gap + second)
    assert [u.format for u in result.units] == [
        "plain",
        "structured_table",
        "structured_table",
    ]
    first_table, second_table = [u.content for u in result.units[1:]]
    assert [[c.text for c in row.cells] for row in first_table.rows] == [
        ["외래", "1000"]
    ]
    assert [[c.text for c in row.cells] for row in second_table.rows] == (
        [["입원", "2000"]] if same_headers else [["JT001", "설명 보존", "개정 사유"]]
    )
    assert [c.text for c in second_table.columns] == (
        ["항목", "금액"] if same_headers else ["코드", "설명", "근거"]
    )
    assert len(result.quality_warnings) == 2


def test_ocr_nested_table_with_ambiguous_outer_structure_preserves_entire_block(
    parse_ocr,
):
    text = (
        "| 구분 | 개정 내용 | 사유 |\n"
        "| --- | --- | --- |\n"
        "| 청구 | 상위 항목 | 세부항목 | 내용 | 추가 | 조건 | 개정 사유 |\n"
        "| 코드 | 설명 | 대상 | 비고 |\n"
        "| --- | --- | --- | --- |\n"
        "| JT001 | 첫 설명 | 대상 A | 보존 |\n"
        "| JT006 | 다른 설명 | 대상 B | 보존 |\n"
        "| JX999 | 나머지 설명 | 대상 C | 보존 |"
    )
    result = parse_ocr(text)
    assert len(result.units) == 2
    assert result.units[1].format == "plain"
    assert result.units[1].content == result.units[1].source.text == text
    assert result.quality_warnings[0].details["reason"] == "ambiguous_table_boundary"


def test_ocr_fallback_is_not_reinterpreted_as_official_letter_text(parse_ocr):
    text = (
        "| 제목 | 내용 |\n"
        "| --- | --- |\n"
        "| 수신자 담당자 | 제목 관련 근거 | 보건복지부 고시 | 24.1.1. 종료 |"
    )
    result = parse_ocr("앞 문단\n\n" + text + "\n\n뒤 문단")
    assert [u.source.text for u in result.units] == [
        "앞 페이지 본문",
        "앞 문단",
        text,
        "뒤 문단",
    ]
    assert result.units[2].metadata["pdf"]["table_fallback"] is True


def test_ocr_fallback_preserves_spacing_and_does_not_swallow_separate_table(parse_ocr):
    malformed = "  | 항목 | 조건 |  \n  | --- | --- |\n  | 대상 | 기간 | 금액 |  "
    normal = "| 항목 | 조건 |\n| --- | --- |\n| 대상 | 보존 |"
    result = parse_ocr(malformed + "\n\n" + normal)
    assert [u.format for u in result.units] == ["plain", "plain", "structured_table"]
    assert result.units[1].content == result.units[1].source.text == malformed
    assert result.units[2].content.rows[0].cells[1].text == "보존"
    assert [w.type for w in result.quality_warnings] == [
        "pdf_ocr_table_unstructured",
        "pdf_ocr_table_inferred",
    ]


def test_ocr_page_number_only_still_produces_no_evidence(parse_ocr):
    result = parse_ocr("\n - 2 - \n")
    assert len(result.units) == 1
    assert result.quality_warnings == []


def test_ocr_continuation_does_not_absorb_next_table_on_the_same_page(parse_ocr):
    header = "| 항목 | 금액 |\n| --- | --- |\n"
    result = parse_ocr(
        header + "| B | 200 |\n\n" + header + "| C | 300 |",
        first_page_text=header + "| A | 100 |",
    )
    assert [u.format for u in result.units] == ["structured_table", "structured_table"]
    assert [[c.text for c in row.cells] for row in result.units[0].content.rows] == [
        ["A", "100"],
        ["B", "200"],
    ]
    assert [[c.text for c in row.cells] for row in result.units[1].content.rows] == [
        ["C", "300"],
    ]
    assert result.units[1].metadata["pdf"]["page"] == 2
