from __future__ import annotations

import hashlib
import io
import zipfile


HP = "http://www.hancom.co.kr/hwpml/2011/paragraph"
OPF = "http://www.idpf.org/2007/opf/"
PNG_BYTES = b"\x89PNG\r\n\x1a\nfake-png"


def _s3_config():
    from rag_document_parser import S3Config

    return S3Config(
        endpoint="http://minio.test",
        bucket="rag-assets",
        access_key="access",
        secret_key="secret",
        prefix="documents",
    )


def _run(text: str) -> str:
    return f"<hp:run><hp:t>{text}</hp:t></hp:run>"


def _text_cell(
    text: str,
    *,
    rowspan: int = 1,
    colspan: int = 1,
    row_addr: int | None = None,
    col_addr: int | None = None,
) -> str:
    addr_xml = (
        f'<hp:cellAddr rowAddr="{row_addr}" colAddr="{col_addr}" />'
        if row_addr is not None and col_addr is not None
        else ""
    )
    return (
        f'<hp:tc rowSpan="{rowspan}" colSpan="{colspan}">'
        f"{addr_xml}"
        f"<hp:cellSpan rowSpan=\"{rowspan}\" colSpan=\"{colspan}\" />"
        "<hp:subList><hp:p>"
        f"{_run(text)}"
        "</hp:p></hp:subList></hp:tc>"
    )


def _image_cell(ref: str = "img1") -> str:
    return (
        "<hp:tc><hp:subList><hp:p><hp:run>"
        f'<hp:pic><hc:img binaryItemIDRef="{ref}" /></hp:pic>'
        "</hp:run></hp:p></hp:subList></hp:tc>"
    )


def _table_cell(table_xml: str) -> str:
    return (
        "<hp:tc><hp:subList><hp:p><hp:run>"
        f"{table_xml}"
        "</hp:run></hp:p></hp:subList></hp:tc>"
    )


def _table(*rows: list[str]) -> str:
    body = "".join(f"<hp:tr>{''.join(row)}</hp:tr>" for row in rows)
    return f"<hp:tbl>{body}</hp:tbl>"


def _make_hwpx(section_xml: str, *, image_bytes: bytes | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("Contents/section0.xml", section_xml)
        if image_bytes is not None:
            z.writestr(
                "Contents/content.hpf",
                (
                    f'<opf:package xmlns:opf="{OPF}" xmlns:hp="{HP}" '
                    'xmlns:hc="http://www.hancom.co.kr/hwpml/2011/core">'
                    "<opf:manifest>"
                    '<opf:item id="img1" href="BinData/image001.png" />'
                    "</opf:manifest></opf:package>"
                ),
            )
            z.writestr("BinData/image001.png", image_bytes)
    return buf.getvalue()


def test_hwpx_backend_parses_text_table_nested_table_and_image_asset():
    from rag_document_parser import HwpxBackend

    nested = _table(
        [_text_cell("항목"), _text_cell("금액")],
        [_text_cell("외래"), _text_cell("1000")],
    )
    outer = _table(
        [_text_cell("구분"), _text_cell("세부")],
        [_text_cell("본인부담"), _table_cell(nested)],
    )
    xml = (
        f'<hp:sec xmlns:hp="{HP}">'
        f"<hp:p>{_run('요양급여 기준 안내')}</hp:p>"
        f"<hp:p><hp:run>{outer}</hp:run></hp:p>"
        '<hp:p><hp:run><hp:pic><hp:img binaryItemIDRef="img1" /></hp:pic></hp:run></hp:p>'
        "</hp:sec>"
    )

    parsed = HwpxBackend().parse(_make_hwpx(xml, image_bytes=PNG_BYTES), ".hwpx")

    assert [unit.type for unit in parsed.units] == ["text", "table", "image"]
    assert parsed.units[0].source.text == "요양급여 기준 안내"
    assert parsed.units[0].evidence.content == "요양급여 기준 안내"

    table = parsed.units[1]
    assert table.evidence.format == "structured_table"
    assert table.evidence.content["columns"] == [
        {"id": "c1", "text": "구분"},
        {"id": "c2", "text": "세부"},
    ]
    assert table.evidence.content["rows"][0]["cells"][0] == {
        "column_id": "c1",
        "text": "본인부담",
        "rowspan": 1,
        "colspan": 1,
        "children": [],
    }
    nested_child = table.evidence.content["rows"][0]["cells"][1]["children"][0]
    assert nested_child["kind"] == "table"
    assert nested_child["format"] == "structured_table"
    assert nested_child["content"]["columns"] == [
        {"id": "c1", "text": "항목"},
        {"id": "c2", "text": "금액"},
    ]
    assert "nested table" in table.source.text
    assert "외래" in table.source.text

    image = parsed.units[2]
    assert image.source.kind == "image"
    assert image.evidence.format == "asset_ref"
    assert image.evidence.content == {"asset_id": "img-0001", "caption": None}
    assert parsed.assets[0].id == "img-0001"
    assert parsed.assets[0].data == PNG_BYTES
    assert parsed.assets[0].mime == "image/png"
    assert parsed.assets[0].ext == "png"


def test_parser_registers_hwpx_backend_and_uploads_hwpx_images(monkeypatch):
    from rag_document_parser import RagDocumentParser

    uploads = []

    def fake_put_object(cfg, key, data, content_type):
        uploads.append((key, data, content_type))
        return f"s3://{cfg.bucket}/{cfg.prefix}/{key}"

    monkeypatch.setattr("rag_document_parser.parser._put_object", fake_put_object)

    xml = (
        f'<hp:sec xmlns:hp="{HP}">'
        '<hp:p><hp:run><hp:pic><hp:img binaryItemIDRef="img1" /></hp:pic></hp:run></hp:p>'
        "</hp:sec>"
    )
    raw = _make_hwpx(xml, image_bytes=PNG_BYTES)
    document_hash = hashlib.sha256(raw).hexdigest()

    result = RagDocumentParser(object_storage=_s3_config()).parse(
        raw,
        suffix=".hwpx",
    )

    assert uploads == [
        (
            f"{document_hash}/assets/img-0001.png",
            PNG_BYTES,
            "image/png",
        )
    ]
    assert result.assets[0].uri == (
        f"s3://rag-assets/documents/{document_hash}/assets/img-0001.png"
    )
    assert result.units[0].evidence.content["uri"] == result.assets[0].uri
    assert not hasattr(result.units[0], "summary")


def test_hwpx_table_cell_image_is_preserved_as_nested_asset_ref():
    from rag_document_parser import HwpxBackend

    table = _table(
        [_text_cell("구분"), _text_cell("이미지")],
        [_text_cell("급여기준"), _image_cell()],
    )
    xml = (
        f'<hp:sec xmlns:hp="{HP}" xmlns:hc="http://www.hancom.co.kr/hwpml/2011/core">'
        f"<hp:p><hp:run>{table}</hp:run></hp:p>"
        "</hp:sec>"
    )

    parsed = HwpxBackend().parse(_make_hwpx(xml, image_bytes=PNG_BYTES), ".hwpx")

    table_unit = parsed.units[0]
    image_child = table_unit.evidence.content["rows"][0]["cells"][1]["children"][0]
    assert image_child == {
        "kind": "image",
        "format": "asset_ref",
        "content": {"asset_id": "img-0001", "caption": None},
    }
    assert parsed.assets[0].id == "img-0001"


def test_nested_asset_refs_are_uploaded_and_resolved_in_table_evidence(monkeypatch):
    from rag_document_parser import RagDocumentParser

    uploads = []

    def fake_put_object(cfg, key, data, content_type):
        uploads.append((key, data, content_type))
        return f"s3://{cfg.bucket}/{cfg.prefix}/{key}"

    monkeypatch.setattr("rag_document_parser.parser._put_object", fake_put_object)

    table = _table(
        [_text_cell("구분"), _text_cell("이미지")],
        [_text_cell("급여기준"), _image_cell()],
    )
    xml = (
        f'<hp:sec xmlns:hp="{HP}" xmlns:hc="http://www.hancom.co.kr/hwpml/2011/core">'
        f"<hp:p><hp:run>{table}</hp:run></hp:p>"
        "</hp:sec>"
    )
    raw = _make_hwpx(xml, image_bytes=PNG_BYTES)
    document_hash = hashlib.sha256(raw).hexdigest()

    result = RagDocumentParser(object_storage=_s3_config()).parse(
        raw,
        suffix=".hwpx",
    )

    assert uploads == [
        (
            f"{document_hash}/assets/img-0001.png",
            PNG_BYTES,
            "image/png",
        )
    ]
    image_child = result.units[0].evidence.content["rows"][0]["cells"][1]["children"][0]
    assert image_child["content"]["uri"] == (
        f"s3://rag-assets/documents/{document_hash}/assets/img-0001.png"
    )
    assert image_child["content"]["sha256"] == hashlib.sha256(PNG_BYTES).hexdigest()


def test_hwpx_table_first_row_with_image_is_not_lost_as_header():
    from rag_document_parser import HwpxBackend

    table = _table(
        [_image_cell(), _text_cell("문서 제목")],
        [_text_cell("구분"), _text_cell("내용")],
    )
    xml = (
        f'<hp:sec xmlns:hp="{HP}" xmlns:hc="http://www.hancom.co.kr/hwpml/2011/core">'
        f"<hp:p><hp:run>{table}</hp:run></hp:p>"
        "</hp:sec>"
    )

    parsed = HwpxBackend().parse(_make_hwpx(xml, image_bytes=PNG_BYTES), ".hwpx")

    table_content = parsed.units[0].evidence.content
    assert table_content["columns"] == [
        {"id": "c1", "text": ""},
        {"id": "c2", "text": ""},
    ]
    image_child = table_content["rows"][0]["cells"][0]["children"][0]
    assert image_child["format"] == "asset_ref"
    assert image_child["content"]["asset_id"] == "img-0001"
    assert table_content["rows"][0]["cells"][1]["text"] == "문서 제목"


def test_hwpx_table_uses_cell_addresses_and_spans_for_grid_width():
    from rag_document_parser import HwpxBackend

    table = _table(
        [
            _text_cell("", row_addr=0, col_addr=0),
            _text_cell("", row_addr=0, col_addr=1),
            _text_cell("관련 근거", rowspan=2, colspan=2, row_addr=0, col_addr=2),
            _text_cell("", row_addr=0, col_addr=4),
            _text_cell("", row_addr=0, col_addr=5),
        ],
        [
            _text_cell("", row_addr=1, col_addr=0),
            _text_cell("", row_addr=1, col_addr=1),
            _text_cell("", row_addr=1, col_addr=4),
            _text_cell("", row_addr=1, col_addr=5),
        ],
        [
            _text_cell("개정 ’16.11.7.", row_addr=2, col_addr=0),
            _text_cell("고시 제2016-149호", row_addr=2, col_addr=1),
            _text_cell("(2016.10.01.시행)", row_addr=2, col_addr=2),
            _text_cell("1차 QA", colspan=3, row_addr=2, col_addr=3),
        ],
    )
    xml = (
        f'<hp:sec xmlns:hp="{HP}">'
        f"<hp:p><hp:run>{table}</hp:run></hp:p>"
        "</hp:sec>"
    )

    parsed = HwpxBackend().parse(_make_hwpx(xml), ".hwpx")

    content = parsed.units[0].evidence.content
    assert len(content["columns"]) == 6
    assert [column["text"] for column in content["columns"]] == [
        "",
        "",
        "관련 근거",
        "",
        "",
        "",
    ]
    assert [row["index"] for row in content["rows"]] == [1]
    cells = content["rows"][0]["cells"]
    assert [(cell["column_id"], cell["text"], cell["colspan"]) for cell in cells] == [
        ("c1", "개정 ’16.11.7.", 1),
        ("c2", "고시 제2016-149호", 1),
        ("c3", "(2016.10.01.시행)", 1),
        ("c4", "1차 QA", 3),
    ]
    assert sum(cell["colspan"] for cell in cells) == len(content["columns"])
    assert "Column " not in parsed.units[0].source.text


def test_hwpx_single_cell_text_table_is_emitted_as_text_unit():
    from rag_document_parser import HwpxBackend

    table = _table([_text_cell("목  차", row_addr=0, col_addr=0)])
    xml = (
        f'<hp:sec xmlns:hp="{HP}">'
        f"<hp:p><hp:run>{table}</hp:run></hp:p>"
        "</hp:sec>"
    )

    parsed = HwpxBackend().parse(_make_hwpx(xml), ".hwpx")

    assert [unit.type for unit in parsed.units] == ["text"]
    assert parsed.units[0].source.text == "목  차"
    assert parsed.units[0].evidence.content == "목  차"


def test_hwpx_table_keeps_header_rows_covered_by_rowspan_out_of_body():
    from rag_document_parser import HwpxBackend

    table = _table(
        [
            _text_cell("구분", rowspan=2, colspan=2, row_addr=0, col_addr=0),
            _text_cell("산정요건", colspan=2, row_addr=0, col_addr=2),
        ],
        [
            _text_cell("영상", row_addr=1, col_addr=2),
            _text_cell("판독", row_addr=1, col_addr=3),
        ],
        [
            _text_cell("진단", colspan=2, row_addr=2, col_addr=0),
            _text_cell("필수", row_addr=2, col_addr=2),
            _text_cell("필수", row_addr=2, col_addr=3),
        ],
    )
    xml = (
        f'<hp:sec xmlns:hp="{HP}">'
        f"<hp:p><hp:run>{table}</hp:run></hp:p>"
        "</hp:sec>"
    )

    parsed = HwpxBackend().parse(_make_hwpx(xml), ".hwpx")

    content = parsed.units[0].evidence.content
    assert len(content["header_rows"]) == 2
    assert len(content["rows"]) == 1
    assert content["header_rows"][1]["cells"][0]["text"] == "영상"
    assert content["rows"][0]["cells"][0]["text"] == "진단"
