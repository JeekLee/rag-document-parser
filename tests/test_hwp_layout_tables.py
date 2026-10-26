"""Layout tables must expose ordered blocks without flattening real data tables."""
from __future__ import annotations

import io
import struct
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from rag_document_parser import EvidenceUnitAgenticChunker, HwpxBackend
from rag_document_parser.chunk.tokens import count_tokens
from rag_document_parser.evidence_unit_extraction.formats.hwp5.backend import _parse_section
from test_hwp5_backend import (
    _gso_ctrl, _list_header_payload_with_span, _make_record, _shape_ctrl,
    _table_body_payload, _table_ctrl, _u16,
)
from test_hwpx_backend import HP, PNG_BYTES, _make_hwpx, _run


def table(*rows):
    return list(rows)


def wrapper(*blocks):
    return table([list(blocks)])


def data_table(label="검사", size=2):
    return table([["항목"], ["진료내역"]], *[
        [[f"{label}{i}"], [f"{label}{i} 기록 " + "구체적인 진료 내용. " * 12]]
        for i in range(size)
    ])


def hwp_table(rows, level=0):
    records = _table_ctrl(level)
    records += _make_record(0x4D, level + 1, _table_body_payload(len(rows), len(rows[0])))
    for r, row in enumerate(rows):
        for c, blocks in enumerate(row):
            records += _make_record(0x48, level + 1, _list_header_payload_with_span(r, c))
            for block in blocks:
                if isinstance(block, str):
                    records += _make_record(0x43, level + 2, _u16(block))
                else:
                    records += hwp_table(block, level + 2)
                    records += _make_record(0x42, level + 1, b"")
    return records


def hwpx_table(rows):
    parts = [f'<hp:tbl rowCnt="{len(rows)}" colCnt="{len(rows[0])}">']
    for r, row in enumerate(rows):
        parts.append("<hp:tr>")
        for c, blocks in enumerate(row):
            parts.append(f'<hp:tc><hp:cellAddr rowAddr="{r}" colAddr="{c}" />'
                         '<hp:cellSpan rowSpan="1" colSpan="1" /><hp:subList>')
            for block in blocks:
                content = _run(block) if isinstance(block, str) else f"<hp:run>{hwpx_table(block)}</hp:run>"
                parts.append(f"<hp:p>{content}</hp:p>")
            parts.append("</hp:subList></hp:tc>")
        parts.append("</hp:tr>")
    parts.append("</hp:tbl>")
    return "".join(parts)


@pytest.fixture(params=["hwp", "hwpx"])
def parse_layout(request):
    def parse(rows):
        if request.param == "hwp":
            return _parse_section(hwp_table(rows) + _make_record(0x42, 0, b"")).to_document()
        xml = f'<hp:sec xmlns:hp="{HP}"><hp:p><hp:run>{hwpx_table(rows)}</hp:run></hp:p></hp:sec>'
        return HwpxBackend().parse(_make_hwpx(xml), ".hwpx")
    return parse


def test_layout_table_exposes_text_and_nested_tables_in_original_order(parse_layout):
    parsed = parse_layout(wrapper("앞 문단", data_table("첫째"), "중간 문단", data_table("둘째"), "뒤 문단"))
    assert [u.type for u in parsed.units] == ["text", "table", "text", "table", "text"]
    assert [u.content for u in parsed.units if u.type == "text"] == ["앞 문단", "중간 문단", "뒤 문단"]
    assert "첫째0" in parsed.units[1].source.text
    assert "둘째0" in parsed.units[3].source.text
    assert [u.id for u in parsed.units] == [f"b{i}" for i in range(1, 6)]
    assert [u.metadata["table"]["table_id"] for u in parsed.units if u.type == "table"] == ["t1", "t2"]


def test_layout_tables_unwrap_recursively_but_keep_data_cell_nesting(parse_layout):
    inner = data_table("안쪽")
    real = table([["구분"], ["세부"]], [["항목"], ["설명", inner, "결론"]])
    parsed = parse_layout(wrapper("시작", wrapper("중첩 테두리", real), "끝"))
    assert [u.type for u in parsed.units] == ["text", "text", "table", "text"]
    nested = parsed.units[2].content["rows"][0]["cells"][1]["children"]
    assert len(nested) == 1 and nested[0]["type"] == "table"
    assert "안쪽0" in parsed.units[2].source.text


def test_plain_single_cell_and_multicell_tables_are_unchanged(parse_layout):
    text = parse_layout(wrapper("한 칸의 텍스트"))
    assert len(text.units) == 1 and text.units[0].type == "text"
    assert text.units[0].content == "한 칸의 텍스트"
    real = parse_layout(table([["항목"], ["세부"]], [["가"], [data_table()]]))
    assert len(real.units) == 1 and real.units[0].type == "table"
    assert real.units[0].content["rows"][0]["cells"][1]["children"][0]["type"] == "table"


@pytest.mark.parametrize("table_rows", [1, 20])
def test_unwrapped_cases_and_tables_use_existing_chunk_boundaries(parse_layout, table_rows):
    blocks = []
    for i in range(1, 4):
        blocks.extend([f"▶사례{i}", f"청구내역 및 진료내역 ({i})", data_table(f"사례{i}검사", table_rows)])
    parsed = parse_layout(wrapper(*blocks))
    chunks = EvidenceUnitAgenticChunker(
        plan_fn=lambda units, *_: [{"operations": [{"unit_id": u.id, "action": "include"} for u in units]}],
        enrich_final_chunks=False,
    ).chunk(parsed.units)
    assert len(chunks) >= 3
    assert not any(c.metadata.get("_fallback_reason") for c in chunks)
    assert all(count_tokens(c.source.text) <= 2048 for c in chunks)
    warning_types = {w["type"] for c in chunks for w in c.metadata.get("_warnings", [])}
    if table_rows == 1:
        assert len(chunks) == 3
        assert "agentic_case_boundary_split" in warning_types
    else:
        assert len(chunks) > 3
        assert "agentic_table_split_by_token_budget" in warning_types
    for unit in parsed.units:
        if unit.type == "table":
            actual = [row for c in chunks for e in c.evidence.items if unit.id in e.source_unit_ids
                      for row in e.content["rows"]]
            assert actual == unit.content["rows"]


def test_hwpx_layout_preserves_inline_text_and_multiple_tables_in_one_paragraph():
    inner = hwpx_table(data_table())
    content = f"<hp:p><hp:run><hp:t>앞</hp:t>{inner}<hp:t>사이</hp:t>{inner}<hp:t>뒤</hp:t></hp:run></hp:p>"
    outer = f'<hp:tbl rowCnt="1" colCnt="1"><hp:tr><hp:tc><hp:subList>{content}</hp:subList></hp:tc></hp:tr></hp:tbl>'
    xml = f'<hp:sec xmlns:hp="{HP}"><hp:p><hp:run>{outer}</hp:run></hp:p></hp:sec>'
    parsed = HwpxBackend().parse(_make_hwpx(xml), ".hwpx")
    assert [u.type for u in parsed.units] == ["text", "table", "text", "table", "text"]
    assert [u.content for u in parsed.units if u.type == "text"] == ["앞", "사이", "뒤"]


@pytest.mark.parametrize("attribute", ['rowCnt="2"', 'colCnt="2"'])
def test_hwpx_incomplete_multicell_grid_is_not_a_layout_wrapper(attribute):
    inner = hwpx_table(data_table())
    outer = f'<hp:tbl {attribute}><hp:tr><hp:tc><hp:subList><hp:p><hp:run>{inner}</hp:run></hp:p></hp:subList></hp:tc></hp:tr></hp:tbl>'
    xml = f'<hp:sec xmlns:hp="{HP}"><hp:p><hp:run>{outer}</hp:run></hp:p></hp:sec>'
    parsed = HwpxBackend().parse(_make_hwpx(xml), ".hwpx")
    assert len(parsed.units) == 1
    assert parsed.units[0].content["rows"][0]["cells"][0]["children"][0]["type"] == "table"


@pytest.mark.parametrize("rowspan,colspan", [(2, 1), (1, 2)])
def test_merged_cell_covering_a_larger_grid_is_not_unwrapped(rowspan, colspan):
    from rag_document_parser.evidence_unit_extraction.formats.hwp5.blocks import TableBlock, TextBlock
    from rag_document_parser.evidence_unit_extraction.formats.hwp5.layout_tables import expand_layout_tables
    from rag_document_parser.evidence_unit_extraction.formats.hwp5.table_extraction import Cell
    from rag_document_parser.evidence_unit_extraction.formats.hwpx.layout_tables import expand_layout_tables as expand_xml

    outer = TableBlock([[Cell(rowspan=rowspan, colspan=colspan, blocks=[TextBlock("앞"), TableBlock([])])]])
    assert list(expand_layout_tables([outer])) == [outer]
    xml = hwpx_table(wrapper(data_table())).replace('rowSpan="1" colSpan="1"',
                                                  f'rowSpan="{rowspan}" colSpan="{colspan}"', 1)
    paragraph = ET.fromstring(f'<hp:p xmlns:hp="{HP}"><hp:run>{xml}</hp:run></hp:p>')
    assert list(expand_xml([paragraph])) == [paragraph]


def test_hwpx_layout_keeps_inline_images_in_order_and_assets_once():
    inner = hwpx_table(data_table())
    picture = '<hp:pic><hc:img binaryItemIDRef="img1" /></hp:pic>'
    content = f"<hp:p><hp:run><hp:t>앞</hp:t>{picture}<hp:t>사이</hp:t>{inner}<hp:t>뒤</hp:t></hp:run></hp:p>"
    outer = f'<hp:tbl><hp:tr><hp:tc><hp:subList>{content}</hp:subList></hp:tc></hp:tr></hp:tbl>'
    xml = (f'<hp:sec xmlns:hp="{HP}" xmlns:hc="http://www.hancom.co.kr/hwpml/2011/core">'
           f"<hp:p><hp:run>{outer}</hp:run></hp:p></hp:sec>")
    parsed = HwpxBackend().parse(_make_hwpx(xml, image_bytes=PNG_BYTES), ".hwpx")
    assert [u.type for u in parsed.units] == ["text", "image", "text", "table", "text"]
    assert [u.content for u in parsed.units if u.type == "text"] == ["앞", "사이", "뒤"]
    assert len(parsed.assets) == 1 and parsed.assets[0].data == PNG_BYTES
    assert parsed.units[1].content["asset_id"] == parsed.assets[0].id


def test_hwp_layout_keeps_images_and_diagram_geometry_in_order():
    from rag_document_parser.evidence_unit_extraction.formats.hwp5.backend import _BinEntry

    picture = bytearray(80)
    struct.pack_into("<H", picture, 71, 1)
    records = _table_ctrl(0) + _make_record(0x4D, 1, _table_body_payload(1, 1))
    records += _make_record(0x48, 1, _list_header_payload_with_span(0, 0))
    records += _make_record(0x43, 2, _u16("앞"))
    records += _gso_ctrl(2) + _make_record(0x55, 3, bytes(picture))
    records += _make_record(0x43, 2, _u16("사이"))
    records += hwp_table(data_table(), 2) + _make_record(0x42, 1, b"")
    records += _shape_ctrl(2, b"cer$", instance_id=1, x=100, y=200, width=300, height=400)
    records += _make_record(0x43, 3, _u16("도형"))
    records += _make_record(0x43, 2, _u16("뒤")) + _make_record(0x42, 0, b"")
    parsed = _parse_section(records, bin_entries={1: _BinEntry(storage_id=7, ext="png")},
                            bin_streams={7: (PNG_BYTES, "png")}).to_document()
    assert [u.type for u in parsed.units] == ["text", "image", "text", "table", "diagram", "text"]
    assert [u.content for u in parsed.units if u.type == "text"] == ["앞", "사이", "뒤"]
    node = parsed.units[4].content["nodes"][0]
    assert node["text"] == "도형" and node["shape_type"] == "rectangle"
    assert node["bbox"]["width"] == 300
    assert len(parsed.assets) == 1 and parsed.assets[0].data == PNG_BYTES


def test_real_hwpx_wrapper_matches_its_cell_body_without_loss_or_reordering():
    from rag_document_parser.evidence_unit_extraction.formats.hwpx.xml_utils import q

    fixture = next((Path(__file__).parent / "fixtures/corpus/hwpx_size_samples/medium").glob("medium-03-*.hwpx"))
    raw = fixture.read_bytes()
    parsed = HwpxBackend().parse(raw, ".hwpx")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        section = ET.fromstring(archive.read("Contents/section0.xml"))
    # Independent oracle: place the original cell's paragraphs directly in a
    # document body, without the production expander or canonical table payload.
    paragraphs = section.findall(f'{q("p")}/{q("run")}/{q("tbl")}/{q("tr")}/{q("tc")}/{q("subList")}/{q("p")}')
    body = ET.Element(q("sec"))
    body.extend(paragraphs)
    expected = HwpxBackend().parse(_make_hwpx(ET.tostring(body).decode()), ".hwpx")
    assert len(parsed.units) == 25
    assert parsed.units == expected.units
    assert sum(u.type == "table" for u in parsed.units) == 9
    assert parsed.assets == [] and parsed.quality_warnings == []


def test_real_hwp_wrapper_preserves_all_text_and_the_inner_table(monkeypatch):
    pytest.importorskip("olefile")
    from rag_document_parser import Hwp5Backend
    from rag_document_parser.evidence_unit_extraction.formats.hwp5 import document_assembly

    fixture = next((Path(__file__).parent / "fixtures/corpus/hwp/minio-size-samples/medium").glob("medium-05-*.hwp"))
    raw = fixture.read_bytes()
    actual = Hwp5Backend().parse(raw, ".hwp")
    # Compare to the legacy cell projection independently of ordered blocks.
    with monkeypatch.context() as patch:
        patch.setattr(document_assembly, "expand_layout_tables", lambda blocks: blocks)
        before = Hwp5Backend().parse(raw, ".hwp")
    start = next(i for i, u in enumerate(actual.units) if u.type == "text" and u.content == "가. 실시기관 기준")
    promoted = actual.units[start:start + 25]
    original_cell = before.units[start].content["rows"][0]["cells"][0]
    assert [u.type for u in promoted] == ["text"] * 18 + ["table"] + ["text"] * 6
    assert "\n".join(u.content for u in promoted if u.type == "text") == original_cell["text"]
    assert promoted[17].content == "※ 면역관용요법 결과의 정의"
    assert promoted[18].content == original_cell["children"][0]["content"]
    assert promoted[19].content.startswith("라. 의학적 판단")
    assert [u.source for u in actual.units[:start]] == [u.source for u in before.units[:start]]
    assert [u.source for u in actual.units[start + 25:]] == [u.source for u in before.units[start + 1:]]
    assert actual.assets == before.assets == []
    assert actual.quality_warnings == before.quality_warnings == []
