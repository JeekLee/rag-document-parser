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


def test_hwp5_table_source_disambiguates_duplicate_header_labels():
    from rag_document_parser.extract.formats.hwp5 import backend as hwp5_backend

    def cell(
        column_id: str,
        text: str,
        *,
        rowspan: int = 1,
        colspan: int = 1,
    ) -> dict[str, object]:
        return {
            "column_id": column_id,
            "text": text,
            "rowspan": rowspan,
            "colspan": colspan,
            "children": [],
        }

    table = {
        "columns": [
            {"id": "c1", "text": "구분"},
            {"id": "c2", "text": "구분"},
            {"id": "c3", "text": "EDI코드"},
        ],
        "header_rows": [
            {
                "index": 1,
                "cells": [
                    cell("c1", "구분", colspan=2),
                    cell("c3", "EDI코드"),
                ],
            }
        ],
        "rows": [
            {
                "index": 1,
                "cells": [
                    cell("c1", "기본 초음파", rowspan=2),
                    cell("c2", "단순초음파(Ⅰ)"),
                    cell("c3", "EB401"),
                ],
            },
            {
                "index": 2,
                "cells": [
                    cell("c2", "단순초음파(Ⅱ)"),
                    cell("c3", "EB402"),
                ],
            },
        ],
    }

    assert hwp5_backend._table_source_text(table) == (
        "table: 3 columns\n"
        "header 1: cols 1-2: 구분; col 3: EDI코드\n"
        "row 1: 구분 [1]: 기본 초음파; 구분 [2]: 단순초음파(Ⅰ); EDI코드: EB401\n"
        "row 2: 구분 [2]: 단순초음파(Ⅱ); EDI코드: EB402"
    )


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


def _u16(text: str) -> bytes:
    return text.encode("utf-16-le")


def _table_ctrl(level: int) -> bytes:
    return _make_record(0x47, level, b" lbt" + b"\x00" * 8)


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
