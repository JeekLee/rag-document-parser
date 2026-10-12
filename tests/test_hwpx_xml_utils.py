from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from rag_document_parser import HwpxBackend
from rag_document_parser.evidence_unit_extraction.formats.hwpx.xml_utils import (
    element_text,
    paragraph_text,
    run_text,
)

HP = "http://www.hancom.co.kr/hwpml/2011/paragraph"
CORPUS_DIR = Path(__file__).resolve().parent / "fixtures" / "corpus" / "hwpx"


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("앞<hp:lineBreak/>뒤 문장", "앞\n뒤 문장"),
        (
            "앞<hp:fwSpace/>가운데<hp:nbSpace/>뒤<hp:tab/>끝",
            "앞 가운데 뒤\t끝",
        ),
        (
            "<hp:lineBreak/>앞<hp:lineBreak/><hp:lineBreak/>뒤<hp:lineBreak/>",
            "\n앞\n\n뒤\n",
        ),
        ("앞<hp:markpenBegin/>강조<hp:markpenEnd/>뒤", "앞강조뒤"),
        ("<hp:fwSpace/><hp:nbSpace/>", "  "),
        ("", ""),
    ],
)
def test_inline_control_text_is_preserved_in_document_order(content, expected):
    run = ET.fromstring(f'<hp:run xmlns:hp="{HP}"><hp:t>{content}</hp:t></hp:run>')

    assert run_text(run) == expected
    assert element_text(run) == " ".join(expected.split())


def test_run_text_does_not_collect_nested_structural_text_or_references():
    run = ET.fromstring(
        f'<hp:run xmlns:hp="{HP}">'
        "<hp:t>앞<hp:lineBreak/>뒤</hp:t>"
        "<hp:tbl><hp:tr><hp:tc><hp:subList><hp:p><hp:run>"
        "<hp:t>중첩 표<hp:lineBreak/>내용</hp:t>"
        "</hp:run></hp:p></hp:subList></hp:tc></hp:tr></hp:tbl>"
        '<hp:pic><hp:img binaryItemIDRef="img1"/></hp:pic>'
        "<hp:ctrl><hp:footNote><hp:subList><hp:p><hp:run>"
        "<hp:t>각주</hp:t>"
        "</hp:run></hp:p></hp:subList></hp:footNote></hp:ctrl>"
        "<hp:t>끝</hp:t>"
        "</hp:run>"
    )

    assert run_text(run) == "앞\n뒤끝"


def test_paragraph_text_preserves_mixed_content_across_runs():
    paragraph = ET.fromstring(
        f'<hp:p xmlns:hp="{HP}">'
        "<hp:run><hp:t>앞<hp:fwSpace/>가운데</hp:t><hp:t>문장</hp:t></hp:run>"
        "<hp:run><hp:t><hp:lineBreak/>뒤</hp:t></hp:run>"
        "</hp:p>"
    )

    assert paragraph_text(paragraph) == "앞 가운데문장\n뒤"


@pytest.mark.parametrize(
    ("fixture", "expected"),
    [
        ("medical-fee-criteria-2022-139.hwpx", "보건복지부 고시 제2022 -139호"),
        (
            "medical-fee-criteria-2022-139.hwpx",
            "위탁에 관한 기준” 중 급여비용 청구․심사 및 지급절차는 다음 각 호와 같다.",
        ),
        (
            "ultrasound-qa.hwpx",
            "판독소견서는 \n진료기록부에만 \n기재해도 수가산정 \n가능한가?",
        ),
        (
            "benefit-criteria-2024-278.hwpx",
            "1. 국민건강보험법시행령 [별표2]\n"
            "제3호의 규정에 의하여 본인이 부담할 비용을 면제하는 자연분만은 자435분만, 자436둔위",
        ),
    ],
)
def test_hwpx_fixture_preserves_title_question_and_sentence_tails(fixture, expected):
    parsed = HwpxBackend().parse((CORPUS_DIR / fixture).read_bytes(), ".hwpx")

    assert any(expected in unit.source.text for unit in parsed.units)
