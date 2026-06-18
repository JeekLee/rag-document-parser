from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


PNG_BYTES = b"\x89PNG\r\n\x1a\nfake-png"


class _FakeCrop:
    def __init__(self, text: str):
        self._text = text

    def extract_text(self, **kwargs):
        return self._text


class _FakeRow:
    def __init__(self, cells):
        self.cells = cells


class _FakeTable:
    def __init__(self, bbox, row_cells, extracted):
        self.bbox = bbox
        self.rows = [_FakeRow(cells) for cells in row_cells]
        self._extracted = extracted

    def extract(self):
        return self._extracted


class _FakePage:
    width = 300.0
    height = 400.0

    def __init__(self, *, chars, images, tables=(), crop_text=None, full_text=None):
        self.chars = chars
        self.images = images
        self._tables = list(tables)
        self._crop_text = crop_text or {}
        self._full_text = full_text

    def crop(self, bbox):
        top = int(round(bbox[1]))
        bottom = int(round(bbox[3]))
        return _FakeCrop(self._crop_text.get((top, bottom), ""))

    def extract_text(self, **kwargs):
        return self._full_text

    def find_tables(self):
        return list(self._tables)


class _FakePdf:
    def __init__(self, pages):
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _cell_span_summary(rows):
    return [
        [
            (
                cell["column_id"],
                cell["text"],
                cell["rowspan"],
                cell["colspan"],
            )
            for cell in row["cells"]
        ]
        for row in rows
    ]


def test_pdf_backend_extracts_evidence_units_from_text_tables_images_and_ocr(
    monkeypatch,
):
    from rag_document_parser.extract.formats.pdf import backend as pdf_backend
    from rag_document_parser.extract.formats.pdf import PdfBackend

    outer = _FakeTable(
        bbox=(0.0, 80.0, 300.0, 220.0),
        row_cells=[
            [(0.0, 80.0, 100.0, 120.0), (100.0, 80.0, 300.0, 120.0)],
            [(0.0, 120.0, 100.0, 220.0), (100.0, 120.0, 300.0, 220.0)],
        ],
        extracted=[
            ["구분", "세부"],
            ["본인부담", "PARENT_FLAT_TEXT"],
        ],
    )
    nested = _FakeTable(
        bbox=(120.0, 140.0, 280.0, 180.0),
        row_cells=[
            [(120.0, 140.0, 200.0, 160.0), (200.0, 140.0, 280.0, 160.0)],
            [(120.0, 160.0, 200.0, 180.0), (200.0, 160.0, 280.0, 180.0)],
        ],
        extracted=[
            ["항목", "금액"],
            ["외래", "1000"],
        ],
    )
    text_page = _FakePage(
        chars=[{"text": "가"} for _ in range(40)],
        images=[{"name": "Im1", "x0": 10, "x1": 90, "y0": 10, "y1": 70, "top": 240}],
        tables=[outer, nested],
        crop_text={
            (0, 80): "요양급여 안내\n- 1 -",
            (120, 140): "기재형식",
            (180, 220): "예시",
        },
    )
    scanned_page = _FakePage(
        chars=[],
        images=[{"x0": 0, "x1": 300, "y0": 0, "y1": 400}],
    )
    fake_pdf = _FakePdf([text_page, scanned_page])
    monkeypatch.setitem(
        sys.modules,
        "pdfplumber",
        SimpleNamespace(open=lambda stream: fake_pdf),
    )
    monkeypatch.setattr(pdf_backend, "_pdf_reader", lambda data: object())
    monkeypatch.setattr(
        pdf_backend,
        "_extract_page_images",
        lambda data, page_idx, page, start_idx, reader=None: [
            (
                240.0,
                SimpleNamespace(
                    data=PNG_BYTES,
                    mime="image/png",
                    ext="png",
                    is_diagram=False,
                ),
            )
        ],
        raising=False,
    )
    monkeypatch.setattr(
        pdf_backend,
        "_render_page_to_png",
        lambda data, page_idx, bbox, scale=2.0: b"rendered-page",
        raising=False,
    )

    parsed = PdfBackend(
        max_ocr_workers=1,
        ocr_fn=lambda png, page_idx: "스캔 OCR 본문",
    ).parse(b"%PDF-1.4 fake", ".pdf")

    assert [unit.type for unit in parsed.units] == ["text", "table", "image", "text"]
    text_unit = parsed.units[0]
    assert text_unit.source.kind == "text"
    assert text_unit.source.text == "요양급여 안내"
    assert text_unit.type == "text"
    assert text_unit.format == "plain"
    assert text_unit.content == "요양급여 안내"
    assert text_unit.metadata["common"] == {
        "chunk_kind": "text",
        "section_path": [],
        "display_format": "plain",
    }

    table = parsed.units[1]
    assert table.source.kind == "table"
    assert table.type == "table"
    assert table.format == "structured_table"
    assert table.content["columns"] == [
        {"id": "c1", "text": "구분"},
        {"id": "c2", "text": "세부"},
    ]
    assert table.content["rows"][0]["cells"][0] == {
        "column_id": "c1",
        "text": "본인부담",
        "rowspan": 1,
        "colspan": 1,
        "children": [],
    }
    detail_cell = table.content["rows"][0]["cells"][1]
    assert detail_cell["text"] == "기재형식 예시"
    assert detail_cell["children"][0]["type"] == "table"
    assert detail_cell["children"][0]["format"] == "structured_table"
    assert detail_cell["children"][0]["content"]["columns"] == [
        {"id": "c1", "text": "항목"},
        {"id": "c2", "text": "금액"},
    ]
    assert table.metadata["common"] == {
        "chunk_kind": "table",
        "section_path": [],
        "display_format": "structured_table",
    }
    assert table.metadata["table"] == {
        "table_id": "t1",
        "headers": ["구분", "세부"],
        "row_count": 1,
    }
    assert "PARENT_FLAT_TEXT" not in table.source.text
    assert "nested table:" in table.source.text

    image = parsed.units[2]
    assert image.source.kind == "image"
    assert image.type == "image"
    assert image.format == "asset_ref"
    assert image.content == {"asset_id": "img-0001", "caption": None}
    assert parsed.assets[0].id == "img-0001"
    assert parsed.assets[0].data == PNG_BYTES
    assert parsed.assets[0].mime == "image/png"
    assert parsed.assets[0].ext == "png"

    ocr_unit = parsed.units[3]
    assert ocr_unit.source.text == "스캔 OCR 본문"
    assert ocr_unit.format == "plain"
    assert ocr_unit.content == "스캔 OCR 본문"
    assert parsed.quality_warnings == []


def test_pdf_backend_merges_nested_table_continuations(monkeypatch):
    from rag_document_parser.extract.formats.pdf import PdfBackend

    outer = _FakeTable(
        bbox=(0.0, 80.0, 300.0, 260.0),
        row_cells=[
            [(0.0, 80.0, 100.0, 120.0), (100.0, 80.0, 300.0, 120.0)],
            [(0.0, 120.0, 100.0, 260.0), (100.0, 120.0, 300.0, 260.0)],
        ],
        extracted=[
            ["항목", "세부"],
            ["질병코드", "PARENT_FLAT_TEXT"],
        ],
    )
    first_child = _FakeTable(
        bbox=(120.0, 140.0, 280.0, 180.0),
        row_cells=[
            [(120.0, 140.0, 200.0, 160.0), (200.0, 140.0, 280.0, 160.0)],
            [(120.0, 160.0, 200.0, 180.0), (200.0, 160.0, 280.0, 180.0)],
        ],
        extracted=[
            ["질병코드", ""],
            ["A04.7", "G83.4"],
        ],
    )
    continuation_child = _FakeTable(
        bbox=(120.0, 190.0, 280.0, 230.0),
        row_cells=[
            [(120.0, 190.0, 200.0, 210.0), (200.0, 190.0, 280.0, 210.0)],
            [(120.0, 210.0, 200.0, 230.0), (200.0, 210.0, 280.0, 230.0)],
        ],
        extracted=[
            ["E11.5", "J15.2"],
            ["E11.7", "J18.2"],
        ],
    )
    page = _FakePage(
        chars=[{"text": "가"} for _ in range(40)],
        images=[],
        tables=[outer, first_child, continuation_child],
        crop_text={(120, 140): "기준", (180, 190): "중간", (230, 260): "끝"},
    )
    monkeypatch.setitem(
        sys.modules,
        "pdfplumber",
        SimpleNamespace(open=lambda stream: _FakePdf([page])),
    )

    parsed = PdfBackend().parse(b"%PDF-1.4 fake", ".pdf")

    child_tables = parsed.units[0].content["rows"][0]["cells"][1]["children"]
    assert len(child_tables) == 1
    child = child_tables[0]["content"]
    assert child["columns"] == [
        {"id": "c1", "text": "질병코드 / A04.7"},
        {"id": "c2", "text": "질병코드 / G83.4"},
    ]
    assert child["header_rows"] == [
        {
            "index": 1,
            "cells": [
                {
                    "column_id": "c1",
                    "text": "질병코드",
                    "rowspan": 1,
                    "colspan": 2,
                    "children": [],
                }
            ],
        },
        {
            "index": 2,
            "cells": [
                {
                    "column_id": "c1",
                    "text": "A04.7",
                    "rowspan": 1,
                    "colspan": 1,
                    "children": [],
                },
                {
                    "column_id": "c2",
                    "text": "G83.4",
                    "rowspan": 1,
                    "colspan": 1,
                    "children": [],
                },
            ],
        },
    ]
    assert [
        [cell["text"] for cell in row["cells"]]
        for row in child["rows"]
    ] == [
        ["E11.5", "J15.2"],
        ["E11.7", "J18.2"],
    ]


def test_pdf_backend_combines_grouped_header_rows(monkeypatch):
    from rag_document_parser.extract.formats.pdf import PdfBackend

    grouped = _FakeTable(
        bbox=(0.0, 80.0, 300.0, 220.0),
        row_cells=[
            [
                (0.0, 80.0, 42.0, 100.0),
                (42.0, 80.0, 84.0, 100.0),
                (84.0, 80.0, 126.0, 100.0),
                (126.0, 80.0, 168.0, 100.0),
                (168.0, 80.0, 210.0, 100.0),
                (210.0, 80.0, 252.0, 100.0),
                (252.0, 80.0, 300.0, 100.0),
            ],
            [
                (0.0, 100.0, 42.0, 120.0),
                (42.0, 100.0, 84.0, 120.0),
                (84.0, 100.0, 126.0, 120.0),
                (126.0, 100.0, 168.0, 120.0),
                (168.0, 100.0, 210.0, 120.0),
                (210.0, 100.0, 252.0, 120.0),
                (252.0, 100.0, 300.0, 120.0),
            ],
            [
                (0.0, 120.0, 42.0, 220.0),
                (42.0, 120.0, 84.0, 220.0),
                (84.0, 120.0, 126.0, 220.0),
                (126.0, 120.0, 168.0, 220.0),
                (168.0, 120.0, 210.0, 220.0),
                (210.0, 120.0, 252.0, 220.0),
                (252.0, 120.0, 300.0, 220.0),
            ],
        ],
        extracted=[
            ["현행", "", "", "개정", "", "", "비고"],
            ["항목", "제목", "세부인정사항", "항목", "제목", "세부인정사항", ""],
            ["일반사항", "자연분만", "현행 내용", "일반사항", "자연분만", "개정 내용", "수정"],
        ],
    )
    page = _FakePage(chars=[{"text": "가"} for _ in range(40)], images=[], tables=[grouped])
    monkeypatch.setitem(
        sys.modules,
        "pdfplumber",
        SimpleNamespace(open=lambda stream: _FakePdf([page])),
    )

    parsed = PdfBackend().parse(b"%PDF-1.4 fake", ".pdf")

    table = parsed.units[0]
    assert table.content["columns"] == [
        {"id": "c1", "text": "현행 / 항목"},
        {"id": "c2", "text": "현행 / 제목"},
        {"id": "c3", "text": "현행 / 세부인정사항"},
        {"id": "c4", "text": "개정 / 항목"},
        {"id": "c5", "text": "개정 / 제목"},
        {"id": "c6", "text": "개정 / 세부인정사항"},
        {"id": "c7", "text": "비고"},
    ]
    assert len(table.content["header_rows"]) == 2
    assert table.metadata["table"]["headers"] == [
        "현행 / 항목",
        "현행 / 제목",
        "현행 / 세부인정사항",
        "개정 / 항목",
        "개정 / 제목",
        "개정 / 세부인정사항",
        "비고",
    ]
    assert "현행 / 제목: 자연분만" in table.source.text
    assert not any(
        "col 2:" in line
        for line in table.source.text.splitlines()
        if line.startswith("row ")
    )


def test_pdf_backend_restores_pdf_header_cell_spans_from_missing_slots(monkeypatch):
    from rag_document_parser.extract.formats.pdf import PdfBackend

    table = _FakeTable(
        bbox=(0.0, 80.0, 300.0, 180.0),
        row_cells=[
            [
                (0.0, 80.0, 200.0, 110.0),
                None,
                (200.0, 80.0, 300.0, 140.0),
            ],
            [
                (0.0, 110.0, 100.0, 140.0),
                (100.0, 110.0, 200.0, 140.0),
                None,
            ],
            [
                (0.0, 140.0, 100.0, 180.0),
                (100.0, 140.0, 200.0, 180.0),
                (200.0, 140.0, 300.0, 180.0),
            ],
        ],
        extracted=[
            ["본인부담률 인하 관련", None, "개정 관련"],
            ["질의", "답변", None],
            ["질문", "답변", "개정안"],
        ],
    )
    page = _FakePage(chars=[{"text": "가"} for _ in range(40)], images=[], tables=[table])
    monkeypatch.setitem(
        sys.modules,
        "pdfplumber",
        SimpleNamespace(open=lambda stream: _FakePdf([page])),
    )

    parsed = PdfBackend().parse(b"%PDF-1.4 fake", ".pdf")

    content = parsed.units[0].content
    assert [column["text"] for column in content["columns"]] == [
        "본인부담률 인하 관련 / 질의",
        "본인부담률 인하 관련 / 답변",
        "개정 관련",
    ]
    assert [
        (cell["column_id"], cell["text"], cell["rowspan"], cell["colspan"])
        for cell in content["header_rows"][0]["cells"]
    ] == [
        ("c1", "본인부담률 인하 관련", 1, 2),
        ("c3", "개정 관련", 2, 1),
    ]
    assert [
        (cell["column_id"], cell["text"], cell["rowspan"], cell["colspan"])
        for cell in content["header_rows"][1]["cells"]
    ] == [
        ("c1", "질의", 1, 1),
        ("c2", "답변", 1, 1),
    ]


def test_pdf_backend_restores_pdf_body_rowspans_from_missing_slots(monkeypatch):
    from rag_document_parser.extract.formats.pdf import PdfBackend

    table = _FakeTable(
        bbox=(0.0, 80.0, 300.0, 180.0),
        row_cells=[
            [
                (0.0, 80.0, 50.0, 100.0),
                (50.0, 80.0, 150.0, 100.0),
                (150.0, 80.0, 300.0, 100.0),
            ],
            [
                (0.0, 100.0, 50.0, 180.0),
                (50.0, 100.0, 150.0, 180.0),
                (150.0, 100.0, 300.0, 140.0),
            ],
            [
                None,
                None,
                (150.0, 140.0, 300.0, 180.0),
            ],
        ],
        extracted=[
            ["연번", "질의", "답변"],
            ["1", "질문", "첫 답변"],
            [None, None, "추가 답변"],
        ],
    )
    page = _FakePage(chars=[{"text": "가"} for _ in range(40)], images=[], tables=[table])
    monkeypatch.setitem(
        sys.modules,
        "pdfplumber",
        SimpleNamespace(open=lambda stream: _FakePdf([page])),
    )

    parsed = PdfBackend().parse(b"%PDF-1.4 fake", ".pdf")

    rows = parsed.units[0].content["rows"]
    assert [
        (cell["column_id"], cell["text"], cell["rowspan"], cell["colspan"])
        for cell in rows[0]["cells"]
    ] == [
        ("c1", "1", 2, 1),
        ("c2", "질문", 2, 1),
        ("c3", "첫 답변", 1, 1),
    ]
    assert [
        (cell["column_id"], cell["text"], cell["rowspan"], cell["colspan"])
        for cell in rows[1]["cells"]
    ] == [("c3", "추가 답변", 1, 1)]


def test_pdf_backend_uses_semantic_source_label_for_mixed_colspan_headers():
    from rag_document_parser.extract.formats.pdf import backend as pdf_backend

    label = pdf_backend._cell_source_label(
        {"column_id": "c1", "colspan": 3},
        {
            "c1": "개정 / 항목",
            "c2": "개정 / 제목",
            "c3": "비고",
        },
        use_header_labels=True,
    )

    assert label == "개정 / 비고"


def test_pdf_backend_disambiguates_duplicate_source_labels():
    from rag_document_parser.extract.formats.pdf import backend as pdf_backend

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
                    pdf_backend._simple_cell("c1", "구분", colspan=2),
                    pdf_backend._simple_cell("c3", "EDI코드"),
                ],
            }
        ],
        "rows": [
            {
                "index": 1,
                "cells": [
                    pdf_backend._simple_cell("c1", "기본 초음파", rowspan=2),
                    pdf_backend._simple_cell("c2", "단순초음파(Ⅰ)"),
                    pdf_backend._simple_cell("c3", "EB401"),
                ],
            },
            {
                "index": 2,
                "cells": [
                    pdf_backend._simple_cell("c2", "단순초음파(Ⅱ)"),
                    pdf_backend._simple_cell("c3", "EB402"),
                ],
            },
            {
                "index": 3,
                "cells": [
                    pdf_backend._simple_cell("c1", "통합 구분", colspan=2),
                    pdf_backend._simple_cell("c3", "EB499"),
                ],
            },
        ],
    }

    assert pdf_backend._table_source_text(table) == (
        "table: 3 columns\n"
        "header 1: cols 1-2: 구분; col 3: EDI코드\n"
        "row 1: 구분 [1]: 기본 초음파; 구분 [2]: 단순초음파(Ⅰ); EDI코드: EB401\n"
        "row 2: 구분 [2]: 단순초음파(Ⅱ); EDI코드: EB402\n"
        "row 3: 구분: 통합 구분; EDI코드: EB499"
    )


def test_pdf_backend_combines_single_group_header_rows(monkeypatch):
    from rag_document_parser.extract.formats.pdf import PdfBackend

    table = _FakeTable(
        bbox=(0.0, 80.0, 300.0, 180.0),
        row_cells=[
            [
                (0.0, 80.0, 100.0, 100.0),
                (100.0, 80.0, 200.0, 100.0),
                (200.0, 80.0, 300.0, 100.0),
            ],
            [
                (0.0, 100.0, 100.0, 120.0),
                (100.0, 100.0, 200.0, 120.0),
                (200.0, 100.0, 300.0, 120.0),
            ],
            [
                (0.0, 120.0, 100.0, 180.0),
                (100.0, 120.0, 200.0, 180.0),
                (200.0, 120.0, 300.0, 180.0),
            ],
        ],
        extracted=[
            ["진료내역", "", ""],
            ["줄번호", "항", "목"],
            ["0001", "09", "01"],
        ],
    )
    page = _FakePage(chars=[{"text": "가"} for _ in range(40)], images=[], tables=[table])
    monkeypatch.setitem(
        sys.modules,
        "pdfplumber",
        SimpleNamespace(open=lambda stream: _FakePdf([page])),
    )

    parsed = PdfBackend().parse(b"%PDF-1.4 fake", ".pdf")

    content = parsed.units[0].content
    assert [column["text"] for column in content["columns"]] == [
        "진료내역 / 줄번호",
        "진료내역 / 항",
        "진료내역 / 목",
    ]
    assert len(content["header_rows"]) == 2
    assert [[cell["text"] for cell in row["cells"]] for row in content["rows"]] == [
        ["0001", "09", "01"]
    ]


def test_pdf_backend_converts_multi_cell_title_table_to_text(monkeypatch):
    from rag_document_parser.extract.formats.pdf import PdfBackend

    title = _FakeTable(
        bbox=(0.0, 80.0, 300.0, 100.0),
        row_cells=[
            [(0.0, 80.0, 250.0, 100.0), (250.0, 80.0, 300.0, 100.0)],
        ],
        extracted=[["질병군 적용 대", "상"]],
    )
    page = _FakePage(chars=[{"text": "가"} for _ in range(40)], images=[], tables=[title])
    monkeypatch.setitem(
        sys.modules,
        "pdfplumber",
        SimpleNamespace(open=lambda stream: _FakePdf([page])),
    )

    parsed = PdfBackend().parse(b"%PDF-1.4 fake", ".pdf")

    assert [unit.type for unit in parsed.units] == ["text"]
    assert parsed.units[0].source.text == "질병군 적용 대상"


def test_pdf_backend_merges_wrapped_table_rows(monkeypatch):
    from rag_document_parser.extract.formats.pdf import PdfBackend

    table = _FakeTable(
        bbox=(0.0, 80.0, 300.0, 180.0),
        row_cells=[
            [
                (0.0, 80.0, 60.0, 100.0),
                (60.0, 80.0, 160.0, 100.0),
                (160.0, 80.0, 300.0, 100.0),
            ],
            [
                (0.0, 100.0, 60.0, 130.0),
                (60.0, 100.0, 160.0, 130.0),
                (160.0, 100.0, 300.0, 130.0),
            ],
            [
                (0.0, 130.0, 60.0, 155.0),
                (60.0, 130.0, 160.0, 155.0),
                (160.0, 130.0, 300.0, 155.0),
            ],
            [
                (0.0, 155.0, 60.0, 180.0),
                (60.0, 155.0, 160.0, 180.0),
                (160.0, 155.0, 300.0, 180.0),
            ],
        ],
        extracted=[
            ["항목", "제목", "세부인정사항"],
            ["가2 입원료", "장기입원", "가. 입원 진료가 필요한 경우"],
            ["", "", "나. 계속 입원 진료가 필요한 경우"],
            ["", "경우의 범주", "다. 그 밖의 세부 기준"],
        ],
    )
    page = _FakePage(chars=[{"text": "가"} for _ in range(40)], images=[], tables=[table])
    monkeypatch.setitem(
        sys.modules,
        "pdfplumber",
        SimpleNamespace(open=lambda stream: _FakePdf([page])),
    )

    parsed = PdfBackend().parse(b"%PDF-1.4 fake", ".pdf")

    rows = parsed.units[0].content["rows"]
    assert len(rows) == 1
    assert rows[0]["index"] == 1
    assert rows[0]["cells"][1]["text"] == "장기입원\n경우의 범주"
    assert rows[0]["cells"][2]["text"] == (
        "가. 입원 진료가 필요한 경우\n"
        "나. 계속 입원 진료가 필요한 경우\n"
        "다. 그 밖의 세부 기준"
    )
    assert parsed.units[0].metadata["table"]["row_count"] == 1


def test_pdf_backend_keeps_bullet_subrows_separate(monkeypatch):
    from rag_document_parser.extract.formats.pdf import PdfBackend

    table = _FakeTable(
        bbox=(0.0, 80.0, 300.0, 160.0),
        row_cells=[
            [
                (0.0, 80.0, 40.0, 100.0),
                (40.0, 80.0, 130.0, 100.0),
                (130.0, 80.0, 215.0, 100.0),
                (215.0, 80.0, 300.0, 100.0),
            ],
            [
                (0.0, 100.0, 40.0, 130.0),
                (40.0, 100.0, 130.0, 130.0),
                (130.0, 100.0, 215.0, 130.0),
                (215.0, 100.0, 300.0, 130.0),
            ],
            [
                (0.0, 130.0, 40.0, 160.0),
                (40.0, 130.0, 130.0, 160.0),
                (130.0, 130.0, 215.0, 160.0),
                (215.0, 130.0, 300.0, 160.0),
            ],
        ],
        extracted=[
            ["연번", "질의", "답변", "개정(안)"],
            ["1", "질의 본문", "답변 본문", "<현행 유지>"],
            ["", "", "- 별도 청구 기준", "- 별도 청구 기준"],
        ],
    )
    page = _FakePage(chars=[{"text": "가"} for _ in range(40)], images=[], tables=[table])
    monkeypatch.setitem(
        sys.modules,
        "pdfplumber",
        SimpleNamespace(open=lambda stream: _FakePdf([page])),
    )

    parsed = PdfBackend().parse(b"%PDF-1.4 fake", ".pdf")

    rows = parsed.units[0].content["rows"]
    assert len(rows) == 2
    assert rows[1]["cells"][2]["text"] == "- 별도 청구 기준"


def test_pdf_backend_keeps_table_of_contents_rows_separate(monkeypatch):
    from rag_document_parser.extract.formats.pdf import PdfBackend

    table = _FakeTable(
        bbox=(0.0, 80.0, 300.0, 160.0),
        row_cells=[
            [
                (0.0, 80.0, 40.0, 100.0),
                (40.0, 80.0, 240.0, 100.0),
                (240.0, 80.0, 300.0, 100.0),
            ],
            [
                (0.0, 100.0, 40.0, 130.0),
                (40.0, 100.0, 240.0, 130.0),
                (240.0, 100.0, 300.0, 130.0),
            ],
            [
                (0.0, 130.0, 40.0, 160.0),
                (40.0, 130.0, 240.0, 160.0),
                (240.0, 130.0, 300.0, 160.0),
            ],
        ],
        extracted=[
            ["연번", "제목", "페이지"],
            ["", "일반사항", ""],
            ["", "초음파 검사 Q&A", ""],
        ],
    )
    page = _FakePage(chars=[{"text": "가"} for _ in range(40)], images=[], tables=[table])
    monkeypatch.setitem(
        sys.modules,
        "pdfplumber",
        SimpleNamespace(open=lambda stream: _FakePdf([page])),
    )

    parsed = PdfBackend().parse(b"%PDF-1.4 fake", ".pdf")

    rows = parsed.units[0].content["rows"]
    assert len(rows) == 2
    assert [row["cells"][1]["text"] for row in rows] == [
        "일반사항",
        "초음파 검사 Q&A",
    ]


def test_pdf_backend_restores_table_of_contents_numbers(monkeypatch):
    from rag_document_parser.extract.formats.pdf import PdfBackend

    table = _FakeTable(
        bbox=(0.0, 80.0, 300.0, 160.0),
        row_cells=[
            [
                (0.0, 80.0, 40.0, 100.0),
                (40.0, 80.0, 240.0, 100.0),
                (240.0, 80.0, 300.0, 100.0),
            ],
            [
                (0.0, 100.0, 40.0, 130.0),
                (40.0, 100.0, 240.0, 130.0),
                (240.0, 100.0, 300.0, 130.0),
            ],
            [
                (0.0, 130.0, 40.0, 160.0),
                (40.0, 130.0, 240.0, 160.0),
                (240.0, 130.0, 300.0, 160.0),
            ],
        ],
        extracted=[
            ["연번", "제목", "페이지"],
            ["", "일반사항", ""],
            ["", "초음파 검사 Q&A", ""],
        ],
    )
    page = _FakePage(
        chars=[{"text": "가"} for _ in range(40)],
        images=[],
        tables=[table],
        full_text="\n".join(
            [
                "목 차",
                "연번 제목 페이지",
                "1 일반사항 1",
                "2 초음파 검사 Q&A 5",
            ]
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "pdfplumber",
        SimpleNamespace(open=lambda stream: _FakePdf([page])),
    )

    parsed = PdfBackend().parse(b"%PDF-1.4 fake", ".pdf")

    rows = parsed.units[0].content["rows"]
    assert [cell["text"] for cell in rows[0]["cells"]] == ["1", "일반사항", "1"]
    assert [cell["text"] for cell in rows[1]["cells"]] == [
        "2",
        "초음파 검사 Q&A",
        "5",
    ]


def test_pdf_backend_restores_cell_line_break_spaces():
    from rag_document_parser.extract.formats.pdf import backend as pdf_backend

    assert pdf_backend._clean_cell(
        "급여 확대되는\n초음파 검사에도\n면허종류와"
    ) == "급여 확대되는 초음파 검사에도 면허종류와"
    assert pdf_backend._clean_cell("급여대\n상") == "급여대상"


def test_pdf_backend_promotes_ultrasound_code_matrix_rows():
    from rag_document_parser.extract.formats.pdf import backend as pdf_backend

    table = {
        "caption": None,
        "columns": [
            {"id": "c1", "text": "구분"},
            {"id": "c2", "text": "EDI코드"},
        ],
        "header_rows": [
            {
                "index": 1,
                "cells": [
                    pdf_backend._simple_cell("c1", "구분"),
                    pdf_backend._simple_cell("c2", "EDI코드"),
                ],
            }
        ],
        "rows": [
            {
                "index": 1,
                "cells": [
                    pdf_backend._simple_cell(
                        "c1",
                        "기본 단순초음파(Ⅰ) 초음파 단순초음파(Ⅱ)",
                    ),
                    pdf_backend._simple_cell("c2", "EB401 EB402"),
                ],
            },
            {
                "index": 2,
                "cells": [
                    pdf_backend._simple_cell(
                        "c1",
                        "유방·액와부-일반 진단 유방·액와부-정밀 초음파 자동유방초음파 흉벽, 흉막, 늑골 등",
                    ),
                    pdf_backend._simple_cell("c2", "EB421 EB423 EB424 EB422"),
                ],
            },
            {
                "index": 3,
                "cells": [
                    pdf_backend._simple_cell(
                        "c1",
                        "유방·액와부-일반 제한적 유방·액와부-정밀 초음파 자동유방초음파 흉벽, 흉막, 늑골 등",
                    ),
                    pdf_backend._simple_cell(
                        "c2",
                        "EB421001 EB423001 EB424001 EB422001",
                    ),
                ],
            },
        ],
    }

    pdf_backend._promote_ultrasound_code_matrix(table)

    assert [column["text"] for column in table["columns"]] == [
        "구분",
        "구분",
        "EDI코드",
    ]
    assert _cell_span_summary(table["header_rows"]) == [
        [("c1", "구분", 1, 2), ("c3", "EDI코드", 1, 1)],
    ]
    assert _cell_span_summary(table["rows"]) == [
        [("c1", "기본 초음파", 2, 1), ("c2", "단순초음파(Ⅰ)", 1, 1), ("c3", "EB401", 1, 1)],
        [("c2", "단순초음파(Ⅱ)", 1, 1), ("c3", "EB402", 1, 1)],
        [("c1", "진단 초음파", 4, 1), ("c2", "유방·액와부-일반", 1, 1), ("c3", "EB421", 1, 1)],
        [("c2", "유방·액와부-정밀", 1, 1), ("c3", "EB423", 1, 1)],
        [("c2", "자동유방초음파", 1, 1), ("c3", "EB424", 1, 1)],
        [("c2", "흉벽, 흉막, 늑골 등", 1, 1), ("c3", "EB422", 1, 1)],
        [("c1", "제한적 초음파", 4, 1), ("c2", "유방·액와부-일반", 1, 1), ("c3", "EB421001", 1, 1)],
        [("c2", "유방·액와부-정밀", 1, 1), ("c3", "EB423001", 1, 1)],
        [("c2", "자동유방초음파", 1, 1), ("c3", "EB424001", 1, 1)],
        [("c2", "흉벽, 흉막, 늑골 등", 1, 1), ("c3", "EB422001", 1, 1)],
    ]


def test_pdf_backend_promotes_ultrasound_code_matrix_rows_with_leading_group():
    from rag_document_parser.extract.formats.pdf import backend as pdf_backend

    table = {
        "caption": None,
        "columns": [
            {"id": "c1", "text": "구분"},
            {"id": "c2", "text": "EDI코드"},
        ],
        "header_rows": [
            {
                "index": 1,
                "cells": [
                    pdf_backend._simple_cell("c1", "구분"),
                    pdf_backend._simple_cell("c2", "EDI코드"),
                ],
            }
        ],
        "rows": [
            {
                "index": 1,
                "cells": [
                    pdf_backend._simple_cell(
                        "c1",
                        "기본 단순초음파(Ⅰ) 초음파 단순초음파(Ⅱ)",
                    ),
                    pdf_backend._simple_cell("c2", "EB401 EB402"),
                ],
            },
            {
                "index": 2,
                "cells": [
                    pdf_backend._simple_cell(
                        "c1",
                        "진단 간·담낭·담도·비장·췌장(일반) 초음파 간·담낭·담도·비장·췌장(정밀)",
                    ),
                    pdf_backend._simple_cell("c2", "EB441 EB442"),
                ],
            },
            {
                "index": 3,
                "cells": [
                    pdf_backend._simple_cell(
                        "c1",
                        "제한적 간·담낭·담도·비장·췌장(일반) 초음파 간·담낭·담도·비장·췌장(정밀)",
                    ),
                    pdf_backend._simple_cell("c2", "EB441001 EB442001"),
                ],
            },
        ],
    }

    pdf_backend._promote_ultrasound_code_matrix(table)

    assert _cell_span_summary(table["rows"]) == [
        [("c1", "기본 초음파", 2, 1), ("c2", "단순초음파(Ⅰ)", 1, 1), ("c3", "EB401", 1, 1)],
        [("c2", "단순초음파(Ⅱ)", 1, 1), ("c3", "EB402", 1, 1)],
        [("c1", "진단 초음파", 2, 1), ("c2", "간·담낭·담도·비장·췌장(일반)", 1, 1), ("c3", "EB441", 1, 1)],
        [("c2", "간·담낭·담도·비장·췌장(정밀)", 1, 1), ("c3", "EB442", 1, 1)],
        [("c1", "제한적 초음파", 2, 1), ("c2", "간·담낭·담도·비장·췌장(일반)", 1, 1), ("c3", "EB441001", 1, 1)],
        [("c2", "간·담낭·담도·비장·췌장(정밀)", 1, 1), ("c3", "EB442001", 1, 1)],
    ]


def test_pdf_backend_promotes_ultrasound_code_matrix_rows_with_known_heart_labels():
    from rag_document_parser.extract.formats.pdf import backend as pdf_backend

    table = {
        "caption": None,
        "columns": [
            {"id": "c1", "text": "구분"},
            {"id": "c2", "text": "EDI코드"},
        ],
        "header_rows": [
            {
                "index": 1,
                "cells": [
                    pdf_backend._simple_cell("c1", "구분"),
                    pdf_backend._simple_cell("c2", "EDI코드"),
                ],
            }
        ],
        "rows": [
            {
                "index": 1,
                "cells": [
                    pdf_backend._simple_cell(
                        "c1",
                        "기본 단순초음파(Ⅰ) 초음파 단순초음파(Ⅱ)",
                    ),
                    pdf_backend._simple_cell("c2", "EB401 EB402"),
                ],
            },
            {
                "index": 2,
                "cells": [
                    pdf_backend._simple_cell(
                        "c1",
                        "선천성 심질환 경흉부 경흉부-단순 경흉부-일반 진단 경흉부-전문 초음파 부하-약물부하 부하-운동부하 태아정밀",
                    ),
                    pdf_backend._simple_cell(
                        "c2",
                        "EB430 EB431 EB432 EB433 EB434 EB435 EB436",
                    ),
                ],
            },
            {
                "index": 3,
                "cells": [
                    pdf_backend._simple_cell(
                        "c1",
                        "선천성 심질환 경식도 특수 경식도 초음파 심장내",
                    ),
                    pdf_backend._simple_cell("c2", "EB610 EB611 EB612"),
                ],
            },
        ],
    }

    pdf_backend._promote_ultrasound_code_matrix(table)

    assert _cell_span_summary(table["rows"]) == [
        [("c1", "기본 초음파", 2, 1), ("c2", "단순초음파(Ⅰ)", 1, 1), ("c3", "EB401", 1, 1)],
        [("c2", "단순초음파(Ⅱ)", 1, 1), ("c3", "EB402", 1, 1)],
        [("c1", "진단 초음파", 7, 1), ("c2", "선천성 심질환 경흉부", 1, 1), ("c3", "EB430", 1, 1)],
        [("c2", "경흉부-단순", 1, 1), ("c3", "EB431", 1, 1)],
        [("c2", "경흉부-일반", 1, 1), ("c3", "EB432", 1, 1)],
        [("c2", "경흉부-전문", 1, 1), ("c3", "EB433", 1, 1)],
        [("c2", "부하-약물부하", 1, 1), ("c3", "EB434", 1, 1)],
        [("c2", "부하-운동부하", 1, 1), ("c3", "EB435", 1, 1)],
        [("c2", "태아정밀", 1, 1), ("c3", "EB436", 1, 1)],
        [("c1", "특수 초음파", 3, 1), ("c2", "선천성 심질환 경식도", 1, 1), ("c3", "EB610", 1, 1)],
        [("c2", "경식도", 1, 1), ("c3", "EB611", 1, 1)],
        [("c2", "심장내", 1, 1), ("c3", "EB612", 1, 1)],
    ]


def test_pdf_backend_expands_parallel_code_action_rows():
    from rag_document_parser.extract.formats.pdf import backend as pdf_backend

    table = {
        "caption": None,
        "columns": [
            {"id": "c1", "text": "분류"},
            {"id": "c2", "text": "코드"},
            {"id": "c3", "text": "행위명"},
        ],
        "rows": [
            {
                "index": 1,
                "cells": [
                    pdf_backend._simple_cell("c1", "(Ⅰ)"),
                    pdf_backend._simple_cell("c2", "M6850 / C8040 / C8060"),
                    pdf_backend._simple_cell(
                        "c3",
                        "낭종흡인요법 / 흉막천자 / 심낭천자",
                    ),
                ],
            },
            {
                "index": 2,
                "cells": [
                    pdf_backend._simple_cell("c1", ""),
                    pdf_backend._simple_cell("c2", "C8100"),
                    pdf_backend._simple_cell("c3", "더글라스와천자"),
                ],
            },
        ],
    }

    pdf_backend._expand_parallel_code_action_rows(table)

    assert [[cell["text"] for cell in row["cells"]] for row in table["rows"]] == [
        ["(Ⅰ)", "M6850", "낭종흡인요법"],
        ["", "C8040", "흉막천자"],
        ["", "C8060", "심낭천자"],
        ["", "C8100", "더글라스와천자"],
    ]


def test_pdf_backend_expands_parallel_code_action_rows_split_by_lines():
    from rag_document_parser.extract.formats.pdf import backend as pdf_backend

    table = {
        "caption": None,
        "columns": [
            {"id": "c1", "text": "분류"},
            {"id": "c2", "text": "코드"},
            {"id": "c3", "text": "행위명"},
        ],
        "rows": [
            {
                "index": 1,
                "cells": [
                    pdf_backend._simple_cell("c1", "(Ⅰ)"),
                    pdf_backend._simple_cell(
                        "c2",
                        "M6850\nC8040\nO1901 O1903 O1905",
                    ),
                    pdf_backend._simple_cell(
                        "c3",
                        "낭종흡인요법\n흉막천자\n부분체외순환",
                    ),
                ],
            }
        ],
    }

    pdf_backend._expand_parallel_code_action_rows(table)

    assert [[cell["text"] for cell in row["cells"]] for row in table["rows"]] == [
        ["(Ⅰ)", "M6850", "낭종흡인요법"],
        ["", "C8040", "흉막천자"],
        ["", "O1901 O1903 O1905", "부분체외순환"],
    ]


def test_pdf_backend_promotes_revision_history_text_to_table(monkeypatch):
    from rag_document_parser.extract.formats.pdf import PdfBackend

    text = "\n".join(
        [
            "초음파 검사 질의응답",
            "2024. 2.",
            "관련 근거",
            "개정 ’16.11.7. 고시 제2016-149호 (2016.10.01.시행) <1차 Q&A>",
            "개정 ’18.3.29. 고시 제2018-66호 (2018.04.01.시행) <상복부 Q&A>",
            "※ 이 자료는 2016년부터 현재까지 합본한 것입니다.",
            "* 난임치료 시술 관련 초음파 검사는 별도 참고",
        ]
    )
    page = _FakePage(
        chars=[{"text": "가"} for _ in range(40)],
        images=[],
        tables=[],
        crop_text={(0, 400): text},
    )
    monkeypatch.setitem(
        sys.modules,
        "pdfplumber",
        SimpleNamespace(open=lambda stream: _FakePdf([page])),
    )

    parsed = PdfBackend().parse(b"%PDF-1.4 fake", ".pdf")

    assert [unit.type for unit in parsed.units] == [
        "text",
        "text",
        "table",
        "text",
        "text",
    ]
    assert parsed.units[0].source.text == "초음파 검사 질의응답"
    assert parsed.units[1].source.text == "2024. 2."
    table = parsed.units[2]
    assert table.metadata["table"]["row_count"] == 2
    assert table.content["columns"] == [
        {"id": "c1", "text": "개정일"},
        {"id": "c2", "text": "고시"},
        {"id": "c3", "text": "시행일"},
        {"id": "c4", "text": "관련 근거"},
    ]
    assert table.content["rows"][0]["cells"][1]["text"] == "고시 제2016-149호"
    assert table.content["rows"][1]["cells"][3]["text"] == "<상복부 Q&A>"
    assert parsed.units[3].source.text == "※ 이 자료는 2016년부터 현재까지 합본한 것입니다."
    assert parsed.units[4].source.text == "* 난임치료 시술 관련 초음파 검사는 별도 참고"


def test_pdf_backend_splits_official_notice_text_into_paragraphs(monkeypatch):
    from rag_document_parser.extract.formats.pdf import PdfBackend

    text = "\n".join(
        [
            "보건복지부 고시 제2024 - 278호",
            "「국민건강보험법」제41조제3항 및 제4항, 「국민건강보험법 시행령」",
            "제19조제1항 관련 별표2 및 「국민건강보험 요양급여의 기준에 관한",
            "규칙」제5조제2항에 의한 「요양급여의 적용기준 및 방법에 관한 세부",
            "사항」을 다음과 같이 개정",
            "ㆍ발령합니다.",
            "2024년 12월 27일",
            "보건복지부 장관",
            "「요양급여의 적용기준 및 방법에 관한 세부사항」일부개정",
            "요양급여의 적용기준 및 방법에 관한 세부사항 일부를 다음과 같이",
            "개정한다.",
            "Ⅰ. 행위 일반사항 중 일반사항의 자연분만시 본인부담금 면제대상",
            "적용범주란을 다음과 같이 한다.",
        ]
    )
    page = _FakePage(
        chars=[{"text": "가"} for _ in range(40)],
        images=[],
        tables=[],
        crop_text={(0, 400): text},
    )
    monkeypatch.setitem(
        sys.modules,
        "pdfplumber",
        SimpleNamespace(open=lambda stream: _FakePdf([page])),
    )

    parsed = PdfBackend().parse(b"%PDF-1.4 fake", ".pdf")

    assert [unit.source.text for unit in parsed.units] == [
        "보건복지부 고시 제2024 - 278호",
        (
            "「국민건강보험법」제41조제3항 및 제4항, 「국민건강보험법 시행령」 "
            "제19조제1항 관련 별표2 및 「국민건강보험 요양급여의 기준에 관한 "
            "규칙」제5조제2항에 의한 「요양급여의 적용기준 및 방법에 관한 세부"
            "사항」을 다음과 같이 개정 ㆍ발령합니다."
        ),
        "2024년 12월 27일",
        "보건복지부 장관",
        "「요양급여의 적용기준 및 방법에 관한 세부사항」일부개정",
        "요양급여의 적용기준 및 방법에 관한 세부사항 일부를 다음과 같이 개정한다.",
        "Ⅰ. 행위 일반사항 중 일반사항의 자연분만시 본인부담금 면제대상 적용범주란을 다음과 같이 한다.",
    ]


def test_pdf_backend_splits_short_heading_lines(monkeypatch):
    from rag_document_parser.extract.formats.pdf import PdfBackend

    page = _FakePage(
        chars=[{"text": "가"} for _ in range(40)],
        images=[],
        tables=[],
        crop_text={(0, 400): "요양급여의 적용기준 및 방법에 관한 세부사항\n신구조문 대비표"},
    )
    monkeypatch.setitem(
        sys.modules,
        "pdfplumber",
        SimpleNamespace(open=lambda stream: _FakePdf([page])),
    )

    parsed = PdfBackend().parse(b"%PDF-1.4 fake", ".pdf")

    assert [unit.source.text for unit in parsed.units] == [
        "요양급여의 적용기준 및 방법에 관한 세부사항",
        "신구조문 대비표",
    ]


def test_pdf_backend_splits_related_basis_bullets(monkeypatch):
    from rag_document_parser.extract.formats.pdf import PdfBackend

    text = "\n".join(
        [
            "제왕절개분만 입원환자의 본인부담률 개정(5%→0%)관련 질의응답",
            "1. 관련 근거",
            "○「국민건강보험법 시행령」 [별표 2] (대통령령 제35054호)",
            "○「건강보험 행위 급여․비급여 목록표 및 급여 상대가치점수」(보건복지부고시 제2024-280호)",
        ]
    )
    page = _FakePage(
        chars=[{"text": "가"} for _ in range(40)],
        images=[],
        tables=[],
        crop_text={(0, 400): text},
    )
    monkeypatch.setitem(
        sys.modules,
        "pdfplumber",
        SimpleNamespace(open=lambda stream: _FakePdf([page])),
    )

    parsed = PdfBackend().parse(b"%PDF-1.4 fake", ".pdf")

    assert [unit.source.text for unit in parsed.units] == [
        "제왕절개분만 입원환자의 본인부담률 개정(5%→0%)관련 질의응답",
        "1. 관련 근거",
        "○「국민건강보험법 시행령」 [별표 2] (대통령령 제35054호)",
        "○「건강보험 행위 급여․비급여 목록표 및 급여 상대가치점수」(보건복지부고시 제2024-280호)",
    ]


def test_pdf_backend_drops_duplicate_short_title_before_full_heading(monkeypatch):
    from rag_document_parser.extract.formats.pdf import PdfBackend

    title = _FakeTable(
        bbox=(0.0, 80.0, 300.0, 100.0),
        row_cells=[
            [(0.0, 80.0, 250.0, 100.0), (250.0, 80.0, 300.0, 100.0)],
        ],
        extracted=[["질병군 적용 대", "상"]],
    )
    page = _FakePage(
        chars=[{"text": "가"} for _ in range(40)],
        images=[],
        tables=[title],
        crop_text={
            (100, 400): "가. 「건강보험 행위 급여․비급여 목록표」고시 제2편 질병군 적용 대상",
        },
    )
    monkeypatch.setitem(
        sys.modules,
        "pdfplumber",
        SimpleNamespace(open=lambda stream: _FakePdf([page])),
    )

    parsed = PdfBackend().parse(b"%PDF-1.4 fake", ".pdf")

    assert [unit.source.text for unit in parsed.units] == [
        "가. 「건강보험 행위 급여․비급여 목록표」고시 제2편 질병군 적용 대상",
    ]


def test_pdf_backend_splits_sectioned_text_blocks(monkeypatch):
    from rag_document_parser.extract.formats.pdf import PdfBackend

    text = "\n".join(
        [
            "일반사항",
            "다음의 수가산정방법 및 청구방법은 「초음파 검사의 급여기준」,",
            "「상복부 초음파 검사의 급여기준」에서 정하는 세부 내용임.",
            "□ 수가산정방법",
        ]
    )
    page = _FakePage(
        chars=[{"text": "가"} for _ in range(40)],
        images=[],
        tables=[],
        crop_text={(0, 400): text},
    )
    monkeypatch.setitem(
        sys.modules,
        "pdfplumber",
        SimpleNamespace(open=lambda stream: _FakePdf([page])),
    )

    parsed = PdfBackend().parse(b"%PDF-1.4 fake", ".pdf")

    assert [unit.source.text for unit in parsed.units] == [
        "일반사항",
        "다음의 수가산정방법 및 청구방법은 「초음파 검사의 급여기준」, 「상복부 초음파 검사의 급여기준」에서 정하는 세부 내용임.",
        "□ 수가산정방법",
    ]


def test_pdf_backend_splits_scanned_official_letter_ocr(monkeypatch):
    from rag_document_parser.extract.formats.pdf import backend as pdf_backend
    from rag_document_parser.extract.formats.pdf import PdfBackend

    scanned_page = _FakePage(
        chars=[],
        images=[{"x0": 0, "x1": 300, "y0": 0, "y1": 400}],
    )
    text = "\n".join(
        [
            '"긴급지원은 지역번호없이 129번으로"',
            "보건복지부",
            "수신자 수신자 참조",
            "(경유)",
            "제목 의료급여 과다본인부담금 공제 업무처리 요령",
            "1. 적정의료급여 서비스 제공을 위해 노력하시는 귀 기관에 감사드립니다.",
            "2. 의료급여법 제11조의3(급여대상 여부의 확인 등)의 신설로",
            "의료급여기관이 수급권자에게 비용을 과다하게 징수한 것으로 확인되었습니다.",
            "3. 이에 따라 붙임과 같이 업무처리요령을 송부합니다.",
            "붙임 07115-의료급여 과다 본인부담금 공제 업무처리 요령. 끝.",
        ]
    )
    monkeypatch.setitem(
        sys.modules,
        "pdfplumber",
        SimpleNamespace(open=lambda stream: _FakePdf([scanned_page])),
    )
    monkeypatch.setattr(
        pdf_backend,
        "_render_page_to_png",
        lambda data, page_idx, bbox, scale=2.0: b"rendered-page",
    )

    parsed = PdfBackend(max_ocr_workers=1, ocr_fn=lambda png, page_idx: text).parse(
        b"%PDF-1.4 fake",
        ".pdf",
    )

    assert [unit.source.text for unit in parsed.units] == [
        '"긴급지원은 지역번호없이 129번으로"',
        "보건복지부",
        "수신자 수신자 참조",
        "(경유)",
        "제목 의료급여 과다본인부담금 공제 업무처리 요령",
        "1. 적정의료급여 서비스 제공을 위해 노력하시는 귀 기관에 감사드립니다.",
        (
            "2. 의료급여법 제11조의3(급여대상 여부의 확인 등)의 신설로 "
            "의료급여기관이 수급권자에게 비용을 과다하게 징수한 것으로 확인되었습니다."
        ),
        "3. 이에 따라 붙임과 같이 업무처리요령을 송부합니다.",
        "붙임 07115-의료급여 과다 본인부담금 공제 업무처리 요령. 끝.",
    ]


def test_pdf_backend_splits_scanned_official_letter_continuation_ocr(monkeypatch):
    from rag_document_parser.extract.formats.pdf import backend as pdf_backend
    from rag_document_parser.extract.formats.pdf import PdfBackend

    scanned_page = _FakePage(
        chars=[],
        images=[{"x0": 0, "x1": 300, "y0": 0, "y1": 400}],
    )
    text = (
        "보건복지부 수신자 보건복지콜센터장, 국민건강보험공단이사장, "
        "건강보험심사평가원장 주무관 황영원 사회복지사무 김혜래 "
        "시행 기초의료보장팀-4995 (2007.11.16.) 접수 의료급여1부-2970 "
        "우 427-721 경기도 과천시 중앙동 1번지 / www.mohw.go.kr "
        "전화 02-2110-6234 전송 02-503-5395 / na852@mohw.go.kr / 공개"
    )
    monkeypatch.setitem(
        sys.modules,
        "pdfplumber",
        SimpleNamespace(open=lambda stream: _FakePdf([scanned_page])),
    )
    monkeypatch.setattr(
        pdf_backend,
        "_render_page_to_png",
        lambda data, page_idx, bbox, scale=2.0: b"rendered-page",
    )

    parsed = PdfBackend(max_ocr_workers=1, ocr_fn=lambda png, page_idx: text).parse(
        b"%PDF-1.4 fake",
        ".pdf",
    )

    assert [unit.source.text for unit in parsed.units] == [
        "보건복지부",
        "수신자 보건복지콜센터장, 국민건강보험공단이사장, 건강보험심사평가원장",
        "주무관 황영원 사회복지사무 김혜래",
        "시행 기초의료보장팀-4995 (2007.11.16.)",
        "접수 의료급여1부-2970",
        "우 427-721 경기도 과천시 중앙동 1번지 / www.mohw.go.kr",
        "전화 02-2110-6234 전송 02-503-5395 / na852@mohw.go.kr / 공개",
    ]


def test_pdf_backend_splits_multiline_official_letter_continuation_markers(monkeypatch):
    from rag_document_parser.extract.formats.pdf import backend as pdf_backend
    from rag_document_parser.extract.formats.pdf import PdfBackend

    scanned_page = _FakePage(
        chars=[],
        images=[{"x0": 0, "x1": 300, "y0": 0, "y1": 400}],
    )
    text = "\n".join(
        [
            "보건복지부",
            "수신자 보건복지콜센터장, 국민건강보험공단이사장",
            "시행 기초의료보장팀-4995 (2007.11.16.) 접수 의료급여1부-2970",
            "우 427-721 경기도 과천시 중앙동 1번지 / www.mohw.go.kr",
            "전화 02-2110-6234 전송 02-503-5395 / na852@mohw.go.kr / 공개",
        ]
    )
    monkeypatch.setitem(
        sys.modules,
        "pdfplumber",
        SimpleNamespace(open=lambda stream: _FakePdf([scanned_page])),
    )
    monkeypatch.setattr(
        pdf_backend,
        "_render_page_to_png",
        lambda data, page_idx, bbox, scale=2.0: b"rendered-page",
    )

    parsed = PdfBackend(max_ocr_workers=1, ocr_fn=lambda png, page_idx: text).parse(
        b"%PDF-1.4 fake",
        ".pdf",
    )

    assert [unit.source.text for unit in parsed.units] == [
        "보건복지부",
        "수신자 보건복지콜센터장, 국민건강보험공단이사장",
        "시행 기초의료보장팀-4995 (2007.11.16.)",
        "접수 의료급여1부-2970",
        "우 427-721 경기도 과천시 중앙동 1번지 / www.mohw.go.kr",
        "전화 02-2110-6234 전송 02-503-5395 / na852@mohw.go.kr / 공개",
    ]


def test_pdf_backend_reports_missing_pdfplumber_dependency(monkeypatch):
    from rag_document_parser.extract.formats.pdf import PdfBackend

    monkeypatch.setitem(sys.modules, "pdfplumber", None)

    with pytest.raises(NotImplementedError, match="pdfplumber"):
        PdfBackend().parse(b"%PDF-1.4 fake", ".pdf")


def test_pdf_backend_isolates_ocr_failures_as_quality_warnings(monkeypatch):
    from rag_document_parser.extract.formats.pdf import backend as pdf_backend
    from rag_document_parser.extract.formats.pdf import PdfBackend

    scanned_page = _FakePage(
        chars=[],
        images=[{"x0": 0, "x1": 300, "y0": 0, "y1": 400}],
    )
    fake_pdf = _FakePdf([scanned_page])
    monkeypatch.setitem(
        sys.modules,
        "pdfplumber",
        SimpleNamespace(open=lambda stream: fake_pdf),
    )
    monkeypatch.setattr(
        pdf_backend,
        "_render_page_to_png",
        lambda data, page_idx, bbox, scale=2.0: b"rendered-page",
    )

    def fail_ocr(png, page_idx):
        raise RuntimeError("ocr boom")

    parsed = PdfBackend(max_ocr_workers=1, ocr_fn=fail_ocr).parse(
        b"%PDF-1.4 fake",
        ".pdf",
    )

    assert parsed.units == []
    assert parsed.assets == []
    assert parsed.quality_warnings == [
        {
            "type": "pdf_ocr_failed",
            "severity": "medium",
            "page": 1,
            "stage": "ocr",
            "message": "ocr boom",
        }
    ]


def test_pdf_backend_renders_scanned_pages_at_ocr_scale(monkeypatch):
    from rag_document_parser.extract.formats.pdf import backend as pdf_backend

    calls = []

    def fake_render(data, page_idx, bbox, scale):
        calls.append((data, page_idx, bbox, scale))
        return b"high-resolution-page"

    monkeypatch.setattr(pdf_backend, "_render_page_to_png", fake_render)

    warnings = []
    rendered = pdf_backend._render_scanned_page_for_ocr(
        b"%PDF-1.4 fake",
        2,
        SimpleNamespace(width=200, height=400),
        warnings,
    )

    assert rendered == b"high-resolution-page"
    assert warnings == []
    assert calls == [
        (
            b"%PDF-1.4 fake",
            2,
            (0.0, 0.0, 200.0, 400.0),
            3.0,
        )
    ]


def test_pdf_backend_uses_openai_compatible_vision_ocr(monkeypatch):
    from rag_document_parser.extract.formats.pdf import backend as pdf_backend
    from rag_document_parser.extract.formats.pdf import PdfBackend, PdfOcrConfig

    scanned_page = _FakePage(
        chars=[],
        images=[{"x0": 0, "x1": 300, "y0": 0, "y1": 400}],
    )
    fake_pdf = _FakePdf([scanned_page])
    requests = []
    monkeypatch.setitem(
        sys.modules,
        "pdfplumber",
        SimpleNamespace(open=lambda stream: fake_pdf),
    )
    monkeypatch.setattr(
        pdf_backend,
        "_render_page_to_png",
        lambda data, page_idx, bbox, scale=2.0: b"rendered-page",
    )

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return '{"choices":[{"message":{"content":"스캔 OCR"}}]}'.encode("utf-8")

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return FakeResponse()

    monkeypatch.setattr(pdf_backend.request, "urlopen", fake_urlopen)

    parsed = PdfBackend(
        max_ocr_workers=1,
        ocr_llm=PdfOcrConfig(
            url="http://spark.test/v1",
            api_key="secret",
            model="qwen3-vl-30b-a3b",
            timeout=7.0,
        ),
    ).parse(b"%PDF-1.4 fake", ".pdf")

    assert [unit.source.text for unit in parsed.units] == ["스캔 OCR"]
    assert parsed.quality_warnings == []
    request, timeout = requests[0]
    assert request.full_url == "http://spark.test/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer secret"
    assert timeout == 7.0
    body = request.data.decode("utf-8")
    assert '"model": "qwen3-vl-30b-a3b"' in body
    assert "data:image/png;base64," in body


def test_pdf_backend_falls_back_when_vision_ocr_is_empty(monkeypatch):
    from rag_document_parser.extract.formats.pdf import backend as pdf_backend
    from rag_document_parser.extract.formats.pdf import PdfBackend, PdfOcrConfig

    scanned_page = _FakePage(
        chars=[],
        images=[{"x0": 0, "x1": 300, "y0": 0, "y1": 400}],
    )
    monkeypatch.setitem(
        sys.modules,
        "pdfplumber",
        SimpleNamespace(open=lambda stream: _FakePdf([scanned_page])),
    )
    monkeypatch.setattr(
        pdf_backend,
        "_render_page_to_png",
        lambda data, page_idx, bbox, scale=2.0: b"rendered-page",
    )
    monkeypatch.setattr(pdf_backend, "_vision_ocr_png", lambda png, cfg: "")
    monkeypatch.setattr(
        pdf_backend,
        "_ocr_page",
        lambda data, png, page_idx: "fallback OCR",
    )

    parsed = PdfBackend(
        max_ocr_workers=1,
        ocr_llm=PdfOcrConfig(
            url="http://spark.test/v1",
            api_key="secret",
            model="qwen3-vl-30b-a3b",
        ),
    ).parse(b"%PDF-1.4 fake", ".pdf")

    assert [unit.source.text for unit in parsed.units] == ["fallback OCR"]


def test_pdf_backend_does_not_render_scanned_pages_without_image_ocr(monkeypatch):
    from rag_document_parser.extract.formats.pdf import backend as pdf_backend
    from rag_document_parser.extract.formats.pdf import PdfBackend

    scanned_page = _FakePage(
        chars=[],
        images=[{"x0": 0, "x1": 300, "y0": 0, "y1": 400}],
    )
    fake_pdf = _FakePdf([scanned_page])
    monkeypatch.setitem(
        sys.modules,
        "pdfplumber",
        SimpleNamespace(open=lambda stream: fake_pdf),
    )

    def fail_render(data, page_idx, bbox, scale=2.0):
        raise AssertionError("rendering should be skipped without image OCR")

    monkeypatch.setattr(pdf_backend, "_render_page_to_png", fail_render)
    monkeypatch.setattr(
        pdf_backend,
        "_ocr_page",
        lambda data, png, page_idx: "embedded-image OCR",
    )

    parsed = PdfBackend(max_ocr_workers=1).parse(b"%PDF-1.4 fake", ".pdf")

    assert [unit.source.text for unit in parsed.units] == ["embedded-image OCR"]
    assert parsed.quality_warnings == []


def test_pdf_backend_reuses_pdf_reader_for_multiple_image_pages(monkeypatch):
    from rag_document_parser.extract.formats.pdf import PdfBackend

    class FakePilImage:
        format = "PNG"
        mode = "RGBA"
        width = 100
        height = 100

        def save(self, buffer, format):
            buffer.write(PNG_BYTES)

    class FakeImageFile:
        name = "Im1.png"
        image = FakePilImage()

    class FakePdfPage:
        images = [FakeImageFile()]

    class FakePdfReader:
        calls = 0

        def __init__(self, stream):
            type(self).calls += 1
            self.pages = [FakePdfPage(), FakePdfPage()]

    page_1 = _FakePage(
        chars=[{"text": "가"} for _ in range(40)],
        images=[{"name": "Im1", "x0": 10, "x1": 90, "y0": 10, "y1": 70, "top": 20}],
        crop_text={(21, 400): "page 1 text"},
    )
    page_2 = _FakePage(
        chars=[{"text": "나"} for _ in range(40)],
        images=[{"name": "Im1", "x0": 10, "x1": 90, "y0": 10, "y1": 70, "top": 20}],
        crop_text={(21, 400): "page 2 text"},
    )
    monkeypatch.setitem(
        sys.modules,
        "pdfplumber",
        SimpleNamespace(open=lambda stream: _FakePdf([page_1, page_2])),
    )
    monkeypatch.setitem(
        sys.modules,
        "pypdf",
        SimpleNamespace(PdfReader=FakePdfReader),
    )

    parsed = PdfBackend().parse(b"%PDF-1.4 fake", ".pdf")

    assert FakePdfReader.calls == 1
    assert len(parsed.assets) == 2
    assert [unit.type for unit in parsed.units].count("image") == 2
