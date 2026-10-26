from pathlib import Path

import pytest

from rag_document_parser import Hwp5Backend
from rag_document_parser.evidence_unit_extraction.formats.hwp5.table_extraction import Cell, _header_row_count
from rag_document_parser.evidence_unit_extraction.formats.hwpx.table_extraction import _header_row_count as hwpx_headers


@pytest.mark.parametrize("format", ["hwp", "hwpx"])
@pytest.mark.parametrize("outside_value", ["2020.08.17.", ""])
def test_repeated_colspan_data_is_not_a_header_refinement(format, outside_value):
    rows = [
        [Cell("일자", col_addr=0), Cell("내역", col_addr=1, colspan=2)],
        [Cell(outside_value, col_addr=0), Cell("진료내역", col_addr=1, colspan=2)],
        [Cell("2020.09.01.", col_addr=0), Cell("추가내역", col_addr=1, colspan=2)],
    ]
    for index, row in enumerate(rows):
        for cell in row:
            cell.row_addr = index
    if format == "hwp":
        assert _header_row_count(rows) == 1
    else:
        raw = [[{"row_addr": i, "col_addr": c.col_addr, "rowspan": c.rowspan,
                 "colspan": c.colspan, "text": c.text, "children": []} for c in row]
               for i, row in enumerate(rows)]
        assert hwpx_headers(raw) == 1


def test_hwp_values_outside_a_refined_group_stay_in_body():
    rows = [
        [Cell("구분", col_addr=0, colspan=2), Cell("코드", col_addr=2)],
        [Cell("항목", col_addr=0), Cell("내용", col_addr=1), Cell("EB401", col_addr=2)],
    ]
    for index, row in enumerate(rows):
        for cell in row:
            cell.row_addr = index
    assert _header_row_count(rows) == 1


def test_issue75_dates_and_diagnoses_are_body_rows_not_column_labels():
    pytest.importorskip("olefile")
    path = Path(__file__).parent / "fixtures/corpus/hwp/hira-spine-1a-5.hwp"
    doc = Hwp5Backend().parse(path.read_bytes(), ".hwp")
    units = {u.id: u for u in doc.units}
    for uid, rows in [("b21", 10), ("b84", 13)]:
        table = units[uid].content
        assert len(table["header_rows"]) == 1
        assert len(table["rows"]) == rows
        assert table["columns"][0]["text"] == "요양\n개시일자"
        assert "2020." in table["rows"][0]["cells"][0]["text"]
