from __future__ import annotations

import hashlib
import io
import zipfile


HP = "http://www.hancom.co.kr/hwpml/2011/paragraph"
OPF = "http://www.idpf.org/2007/opf/"
PNG_BYTES = b"\x89PNG\r\n\x1a\nfake-png"


def _llm_config():
    from rag_document_parser import LlmConfig

    return LlmConfig(url="http://llm.test/v1", api_key="test", model="test-model")


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


def _text_cell(text: str, *, rowspan: int = 1, colspan: int = 1) -> str:
    return (
        f'<hp:tc rowSpan="{rowspan}" colSpan="{colspan}">'
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

    monkeypatch.setattr(
        "rag_document_parser.parser._chat_json",
        lambda prompt, cfg: {
            "summary": "HWPX chunk summary.",
            "keywords": ["hwpx"],
            "questions": ["HWPX 문서에 무엇이 있나요?"],
        },
    )

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

    result = RagDocumentParser(llm=_llm_config(), object_storage=_s3_config()).parse(
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
    assert result.chunks[0].evidence.content["uri"] == result.assets[0].uri
    assert result.chunks[0].summary == "HWPX chunk summary."


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

    monkeypatch.setattr(
        "rag_document_parser.parser._chat_json",
        lambda prompt, cfg: {
            "summary": "Table image summary.",
            "keywords": ["table", "image"],
            "questions": ["표 안 이미지는 무엇인가요?"],
        },
    )

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

    result = RagDocumentParser(llm=_llm_config(), object_storage=_s3_config()).parse(
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
    image_child = result.chunks[0].evidence.content["rows"][0]["cells"][1]["children"][0]
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
        {"id": "c1", "text": "Column 1"},
        {"id": "c2", "text": "Column 2"},
    ]
    image_child = table_content["rows"][0]["cells"][0]["children"][0]
    assert image_child["format"] == "asset_ref"
    assert image_child["content"]["asset_id"] == "img-0001"
    assert table_content["rows"][0]["cells"][1]["text"] == "문서 제목"
