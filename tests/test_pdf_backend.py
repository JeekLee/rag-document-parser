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

    def __init__(self, *, chars, images, tables=(), crop_text=None):
        self.chars = chars
        self.images = images
        self._tables = list(tables)
        self._crop_text = crop_text or {}

    def crop(self, bbox):
        top = int(round(bbox[1]))
        bottom = int(round(bbox[3]))
        return _FakeCrop(self._crop_text.get((top, bottom), ""))

    def find_tables(self):
        return list(self._tables)


class _FakePdf:
    def __init__(self, pages):
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


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
        lambda data, page_idx, bbox: b"rendered-page",
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
        lambda data, page_idx, bbox: b"rendered-page",
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

    def fail_render(data, page_idx, bbox):
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
