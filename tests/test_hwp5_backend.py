from __future__ import annotations

import sys
import struct
from pathlib import Path

import pytest


FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "corpus"
    / "hwp"
    / "medical-aid-overpayment-deduction.hwp"
)
PNG_BYTES = b"\x89PNG\r\n\x1a\nfake-png"


def test_hwp5_backend_supported_suffixes():
    from rag_document_parser.extract.formats.hwp5.backend import Hwp5Backend

    assert Hwp5Backend.supported_suffixes == (".hwp",)


def test_hwp5_backend_parses_real_fixture_text_and_tables():
    pytest.importorskip("olefile")

    from rag_document_parser.extract.formats.hwp5.backend import Hwp5Backend

    parsed = Hwp5Backend().parse(FIXTURE.read_bytes(), ".hwp")

    assert parsed.assets == []
    assert len(parsed.units) >= 20
    assert {unit.type for unit in parsed.units} >= {"text", "table"}

    canonical_text = "\n".join(unit.source.text for unit in parsed.units)
    assert "의료급여 과다본인부담금 공제의뢰 업무처리요령" in canonical_text
    assert "보건복지부" in canonical_text
    assert "의료급여법 제11조의3" in canonical_text
    assert "환불금지급요청서" in canonical_text
    assert "과다본인부담금 공제의뢰 및 지급통보" in canonical_text
    assert "```hwp-drawing" not in canonical_text
    assert "[[RHWP_IMAGE:" not in canonical_text

    tables = [unit for unit in parsed.units if unit.type == "table"]
    assert len(tables) >= 2
    first_table = tables[0]
    assert first_table.source.kind == "table"
    assert first_table.evidence.kind == "table"
    assert first_table.evidence.format == "structured_table"
    assert first_table.evidence.content["columns"] == [
        {"id": "c1", "text": "수급권자"},
        {"id": "c2", "text": "주민번호"},
        {"id": "c3", "text": "진료일자"},
        {"id": "c4", "text": "과다본인부담금"},
    ]
    assert first_table.source.text.startswith("table: 4 columns\nheader 1:")
    assert first_table.metadata["common"] == {
        "chunk_kind": "table",
        "section_path": [],
        "display_format": "structured_table",
    }

    warning_types = {warning["type"] for warning in parsed.quality_warnings}
    assert "hwp5_drawing_structure_unsupported" in warning_types


def test_hwp5_backend_keeps_source_and_evidence_payloads_separate():
    pytest.importorskip("olefile")

    from rag_document_parser.extract.formats.hwp5.backend import Hwp5Backend

    parsed = Hwp5Backend().parse(FIXTURE.read_bytes(), ".hwp")

    text_unit = next(unit for unit in parsed.units if unit.type == "text")
    assert text_unit.source.kind == "text"
    assert text_unit.evidence.kind == "text"
    assert text_unit.evidence.format == "plain"
    assert text_unit.source.text == text_unit.evidence.content

    table_unit = next(unit for unit in parsed.units if unit.type == "table")
    assert isinstance(table_unit.source.text, str)
    assert isinstance(table_unit.evidence.content, dict)
    assert table_unit.source.text != table_unit.evidence.content


def test_hwp5_backend_reports_missing_olefile_dependency(monkeypatch):
    from rag_document_parser.extract.formats.hwp5.backend import Hwp5Backend

    monkeypatch.setitem(sys.modules, "olefile", None)

    with pytest.raises(NotImplementedError, match="olefile"):
        Hwp5Backend().parse(FIXTURE.read_bytes(), ".hwp")


def _make_header(tag_id: int, level: int, size: int) -> bytes:
    header = (tag_id & 0x3FF) | ((level & 0x3FF) << 10) | ((size & 0xFFF) << 20)
    return struct.pack("<I", header)


def _make_record(tag_id: int, level: int, payload: bytes) -> bytes:
    size = len(payload)
    if size < 0xFFF:
        return _make_header(tag_id, level, size) + payload
    return _make_header(tag_id, level, 0xFFF) + struct.pack("<I", size) + payload


def _list_header_payload(row_addr: int, col_addr: int) -> bytes:
    return struct.pack("<6H", 0, 0, 0, 0, col_addr, row_addr)


def _list_header_payload_with_span(
    row_addr: int,
    col_addr: int,
    *,
    rowspan: int = 1,
    colspan: int = 1,
) -> bytes:
    return struct.pack("<8H", 0, 0, 0, 0, col_addr, row_addr, colspan, rowspan)


def _out_of_bounds_short_list_header_payload() -> bytes:
    return struct.pack("<15H", 1, 0, 0, 0, 3, 0, 8504, 0, 850, 7171, 1, 0, 0, 0, 0)


def _table_body_payload(row_count: int, col_count: int) -> bytes:
    return b"\x00" * 4 + struct.pack("<2H", row_count, col_count)


def _u16(text: str) -> bytes:
    return text.encode("utf-16-le")


def _table_ctrl(level: int) -> bytes:
    return _make_record(0x47, level, b" lbt" + b"\x00" * 8)


def _gso_ctrl(level: int) -> bytes:
    return _make_record(0x47, level, b" osg" + b"\x00" * 8)


def test_hwp5_nested_table_is_structured_child_not_flattened_text():
    from rag_document_parser.extract.formats.hwp5.backend import _parse_section

    records = b""
    records += _table_ctrl(0)
    records += _make_record(0x48, 1, _list_header_payload(0, 0))
    records += _make_record(0x43, 2, _u16("구분"))
    records += _make_record(0x48, 1, _list_header_payload(0, 1))
    records += _make_record(0x43, 2, _u16("세부"))
    records += _make_record(0x48, 1, _list_header_payload(1, 0))
    records += _make_record(0x43, 2, _u16("본인부담"))
    records += _make_record(0x48, 1, _list_header_payload(1, 1))
    records += _make_record(0x43, 2, _u16("상세"))
    records += _table_ctrl(2)
    records += _make_record(0x48, 3, _list_header_payload(0, 0))
    records += _make_record(0x43, 4, _u16("항목"))
    records += _make_record(0x48, 3, _list_header_payload(0, 1))
    records += _make_record(0x43, 4, _u16("금액"))
    records += _make_record(0x48, 3, _list_header_payload(1, 0))
    records += _make_record(0x43, 4, _u16("외래"))
    records += _make_record(0x48, 3, _list_header_payload(1, 1))
    records += _make_record(0x43, 4, _u16("1000"))
    records += _make_record(0x42, 0, b"")

    parsed = _parse_section(records)
    document = parsed.to_document()
    table = document.units[0]
    detail_cell = table.evidence.content["rows"][0]["cells"][1]

    assert detail_cell["text"] == "상세"
    assert detail_cell["children"][0]["kind"] == "table"
    assert detail_cell["children"][0]["format"] == "structured_table"
    assert detail_cell["children"][0]["content"]["columns"] == [
        {"id": "c1", "text": "항목"},
        {"id": "c2", "text": "금액"},
    ]
    assert "nested table:" in table.source.text
    assert "상세" in table.source.text


def test_hwp5_table_preserves_column_addresses_with_blank_gaps():
    from rag_document_parser.extract.formats.hwp5.backend import _parse_section

    records = b""
    records += _table_ctrl(0)
    records += _make_record(0x48, 1, _list_header_payload(0, 0))
    records += _make_record(0x43, 2, _u16("A"))
    records += _make_record(0x48, 1, _list_header_payload(0, 2))
    records += _make_record(0x43, 2, _u16("C"))
    records += _make_record(0x48, 1, _list_header_payload(1, 0))
    records += _make_record(0x43, 2, _u16("x"))
    records += _make_record(0x48, 1, _list_header_payload(1, 2))
    records += _make_record(0x43, 2, _u16("z"))
    records += _make_record(0x42, 0, b"")

    table = _parse_section(records).to_document().units[0]

    assert table.evidence.content["columns"] == [
        {"id": "c1", "text": "A"},
        {"id": "c2", "text": ""},
        {"id": "c3", "text": "C"},
    ]
    assert [cell["text"] for cell in table.evidence.content["rows"][0]["cells"]] == [
        "x",
        "",
        "z",
    ]
    assert "row 1: A: x; C: z" in table.source.text


def test_hwp5_table_uses_table_body_dimensions_to_keep_blank_form_rows():
    from rag_document_parser.extract.formats.hwp5.backend import _parse_section

    records = b""
    records += _table_ctrl(0)
    records += _make_record(0x4D, 1, _table_body_payload(2, 4))
    records += _make_record(0x48, 1, _list_header_payload(0, 0))
    records += _make_record(0x43, 2, _u16("수급권자"))
    records += _make_record(0x48, 1, _list_header_payload(0, 1))
    records += _make_record(0x43, 2, _u16("주민번호"))
    records += _make_record(0x48, 1, _list_header_payload(0, 2))
    records += _make_record(0x43, 2, _u16("진료일자"))
    records += _make_record(0x48, 1, _list_header_payload(0, 3))
    records += _make_record(0x43, 2, _u16("과다본인부담금"))
    records += _make_record(0x48, 1, _list_header_payload(1, 0))
    records += _make_record(0x48, 1, _list_header_payload(1, 1))
    records += _make_record(0x48, 1, _list_header_payload(1, 2))
    records += _make_record(0x48, 1, _list_header_payload(1, 3))
    records += _make_record(0x42, 0, b"")

    table = _parse_section(records).to_document().units[0]
    content = table.evidence.content

    assert [column["text"] for column in content["columns"]] == [
        "수급권자",
        "주민번호",
        "진료일자",
        "과다본인부담금",
    ]
    assert len(content["rows"]) == 1
    assert [cell["text"] for cell in content["rows"][0]["cells"]] == ["", "", "", ""]
    assert table.metadata["table"]["row_count"] == 1


def test_hwp5_table_preserves_cell_spans_like_hwpx_tables():
    from rag_document_parser.extract.formats.hwp5.backend import _parse_section

    records = b""
    records += _table_ctrl(0)
    records += _make_record(0x4D, 1, _table_body_payload(3, 6))
    records += _make_record(
        0x48,
        1,
        _list_header_payload_with_span(0, 0, rowspan=2, colspan=2),
    )
    records += _make_record(0x43, 2, _u16("구분"))
    records += _make_record(0x48, 1, _list_header_payload_with_span(0, 2, colspan=2))
    records += _make_record(0x43, 2, _u16("산정요건"))
    records += _make_record(0x48, 1, _list_header_payload(1, 2))
    records += _make_record(0x43, 2, _u16("영상"))
    records += _make_record(0x48, 1, _list_header_payload(1, 3))
    records += _make_record(0x43, 2, _u16("판독"))
    records += _make_record(0x48, 1, _list_header_payload_with_span(2, 0, colspan=2))
    records += _make_record(0x43, 2, _u16("진단"))
    records += _make_record(0x48, 1, _list_header_payload(2, 2))
    records += _make_record(0x43, 2, _u16("필수"))
    records += _make_record(0x48, 1, _list_header_payload_with_span(2, 3, colspan=3))
    records += _make_record(0x43, 2, _u16("1차 QA"))
    records += _make_record(0x42, 0, b"")

    table = _parse_section(records).to_document().units[0]
    content = table.evidence.content

    assert len(content["columns"]) == 6
    assert [column["text"] for column in content["columns"]] == [
        "구분",
        "구분",
        "산정요건 / 영상",
        "산정요건 / 판독",
        "",
        "",
    ]
    assert len(content["header_rows"]) == 2
    assert content["header_rows"][0]["cells"][0]["rowspan"] == 2
    assert content["header_rows"][0]["cells"][0]["colspan"] == 2
    assert len(content["rows"]) == 1
    assert [
        (cell["column_id"], cell["text"], cell["colspan"])
        for cell in content["rows"][0]["cells"]
    ] == [
        ("c1", "진단", 2),
        ("c3", "필수", 1),
        ("c4", "1차 QA", 3),
    ]
    assert "header 1: cols 1-2: 구분; cols 3-4: 산정요건" in table.source.text
    assert "row 1: 구분: 진단; 산정요건 / 영상: 필수; cols 4-6: 1차 QA" in (
        table.source.text
    )


def test_hwp5_table_ignores_list_headers_outside_declared_dimensions():
    from rag_document_parser.extract.formats.hwp5.backend import _parse_section

    records = b""
    records += _table_ctrl(0)
    records += _make_record(0x4D, 1, _table_body_payload(3, 5))
    records += _make_record(0x48, 1, _out_of_bounds_short_list_header_payload())
    records += _make_record(0x48, 1, _list_header_payload_with_span(0, 0, colspan=2))
    records += _make_record(0x43, 2, _u16("현 행"))
    records += _make_record(0x48, 1, _list_header_payload_with_span(0, 2, colspan=2))
    records += _make_record(0x43, 2, _u16("개 정"))
    records += _make_record(0x48, 1, _list_header_payload_with_span(0, 4, rowspan=2))
    records += _make_record(0x43, 2, _u16("비고"))
    records += _make_record(0x48, 1, _list_header_payload(1, 0))
    records += _make_record(0x43, 2, _u16("제목"))
    records += _make_record(0x48, 1, _list_header_payload(1, 1))
    records += _make_record(0x43, 2, _u16("세부인정사항"))
    records += _make_record(0x48, 1, _list_header_payload(1, 2))
    records += _make_record(0x43, 2, _u16("제목"))
    records += _make_record(0x48, 1, _list_header_payload(1, 3))
    records += _make_record(0x43, 2, _u16("세부인정사항"))
    records += _make_record(0x48, 1, _list_header_payload(2, 0))
    records += _make_record(0x43, 2, _u16("A"))
    records += _make_record(0x48, 1, _list_header_payload(2, 1))
    records += _make_record(0x43, 2, _u16("B"))
    records += _make_record(0x48, 1, _list_header_payload(2, 2))
    records += _make_record(0x43, 2, _u16("C"))
    records += _make_record(0x48, 1, _list_header_payload(2, 3))
    records += _make_record(0x43, 2, _u16("D"))
    records += _make_record(0x48, 1, _list_header_payload(2, 4))
    records += _make_record(0x43, 2, _u16("E"))
    records += _make_record(0x42, 0, b"")

    table = _parse_section(records).to_document().units[0]
    content = table.evidence.content

    assert len(content["columns"]) == 5
    assert [column["text"] for column in content["columns"]] == [
        "현 행 / 제목",
        "현 행 / 세부인정사항",
        "개 정 / 제목",
        "개 정 / 세부인정사항",
        "비고",
    ]
    assert len(content["rows"]) == 1
    assert [cell["text"] for cell in content["rows"][0]["cells"]] == [
        "A",
        "B",
        "C",
        "D",
        "E",
    ]


def test_hwp5_real_fixture_keeps_form_table_blank_rows():
    pytest.importorskip("olefile")

    from rag_document_parser.extract.formats.hwp5.backend import Hwp5Backend

    parsed = Hwp5Backend().parse(FIXTURE.read_bytes(), ".hwp")
    tables = [unit for unit in parsed.units if unit.type == "table"]

    assert len(tables[0].evidence.content["rows"]) == 1
    assert [cell["text"] for cell in tables[0].evidence.content["rows"][0]["cells"]] == [
        "",
        "",
        "",
        "",
    ]
    assert len(tables[1].evidence.content["rows"]) == 3


def test_hwp5_groups_drawing_labels_with_interleaved_short_text():
    from rag_document_parser.extract.formats.hwp5.backend import _parse_section

    records = b""
    records += _make_record(0x43, 0, _u16("업무처리 흐름도"))
    records += _make_record(0x43, 0, _u16("< 확인절차 >"))
    records += _make_record(0x43, 0, _u16("①신청"))
    records += _gso_ctrl(0)
    records += _make_record(0x43, 1, _u16("수급권자"))
    records += _gso_ctrl(0)
    records += _make_record(0x43, 1, _u16("건강보험심사평가원"))
    records += _make_record(0x43, 0, _u16("②통보"))
    records += _gso_ctrl(0)
    records += _make_record(0x43, 1, _u16("의료급여기관"))
    records += _make_record(0x43, 0, _u16("< 붙임 1 >"))

    document = _parse_section(records).to_document()

    assert [unit.metadata["common"]["chunk_kind"] for unit in document.units] == [
        "diagram",
        "text",
    ]
    diagram = document.units[0]
    assert diagram.type == "diagram"
    assert diagram.source.kind == "diagram"
    assert diagram.evidence.kind == "diagram"
    assert diagram.evidence.format == "structured_diagram"
    assert diagram.metadata["common"]["display_format"] == "structured_diagram"
    assert diagram.source.text == (
        "업무처리 흐름도\n"
        "< 확인절차 >\n"
        "①신청\n"
        "수급권자\n"
        "건강보험심사평가원\n"
        "②통보\n"
        "의료급여기관"
    )
    assert [node["text"] for node in diagram.evidence.content["nodes"]] == [
        "업무처리 흐름도",
        "< 확인절차 >",
        "①신청",
        "수급권자",
        "건강보험심사평가원",
        "②통보",
        "의료급여기관",
    ]
    assert diagram.evidence.content["edges"] == []
    assert diagram.evidence.content["mermaid"] is None
    assert document.units[1].source.text == "< 붙임 1 >"


def test_hwp5_real_fixture_groups_flowchart_labels():
    pytest.importorskip("olefile")

    from rag_document_parser.extract.formats.hwp5.backend import Hwp5Backend

    parsed = Hwp5Backend().parse(FIXTURE.read_bytes(), ".hwp")
    drawing_units = [
        unit
        for unit in parsed.units
        if unit.metadata["common"]["chunk_kind"] == "diagram"
    ]
    normal_texts = [
        unit.source.text
        for unit in parsed.units
        if unit.metadata["common"]["chunk_kind"] == "text"
    ]

    assert len(drawing_units) == 1
    diagram = drawing_units[0]
    assert diagram.type == "diagram"
    assert diagram.evidence.format == "structured_diagram"
    assert diagram.evidence.content["edges"] == []
    assert diagram.evidence.content["mermaid"] is None
    assert "업무처리 흐름도" in diagram.source.text
    assert "< 과다본인부담금 확인절차 >" in diagram.source.text
    assert "건강보험심사평가원" in diagram.source.text
    assert "보장기관" in diagram.source.text
    assert "보장기관" not in normal_texts


def test_hwp5_picture_shape_becomes_image_asset_ref():
    from rag_document_parser.extract.formats.hwp5.backend import _BinEntry, _parse_section

    picture_payload = bytearray(80)
    struct.pack_into("<H", picture_payload, 71, 1)
    records = b""
    records += _make_record(0x47, 0, b" osg" + b"\x00" * 8)
    records += _make_record(0x55, 1, bytes(picture_payload))
    records += _make_record(0x42, 0, b"")

    parsed = _parse_section(
        records,
        bin_entries={1: _BinEntry(storage_id=7, ext="png")},
        bin_streams={7: (PNG_BYTES, "png")},
    )
    document = parsed.to_document()

    assert [unit.type for unit in document.units] == ["image"]
    assert document.units[0].evidence.content == {
        "asset_id": "img-0001",
        "caption": None,
    }
    assert document.assets[0].id == "img-0001"
    assert document.assets[0].data == PNG_BYTES
    assert document.assets[0].mime == "image/png"
    assert document.assets[0].ext == "png"
