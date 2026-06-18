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


def _rect(text: str, *, x: int, y: int, width: int, height: int) -> str:
    return (
        f'<hp:rect><hp:pos x="{x}" y="{y}" />'
        f'<hp:sz width="{width}" height="{height}" />'
        "<hp:drawText><hp:p>"
        f"{_run(text)}"
        "</hp:p></hp:drawText></hp:rect>"
    )


def _line(
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    arrow: bool = False,
) -> str:
    arrow_xml = '<hp:lineShape endArrowType="NORMAL" />' if arrow else ""
    return (
        f'<hp:line><hp:pos x="{x}" y="{y}" />'
        f'<hp:sz width="{width}" height="{height}" />'
        f"{arrow_xml}</hp:line>"
    )


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
    assert parsed.units[0].format == "plain"
    assert parsed.units[0].content == "요양급여 기준 안내"

    table = parsed.units[1]
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
    nested_child = table.content["rows"][0]["cells"][1]["children"][0]
    assert nested_child["type"] == "table"
    assert nested_child["format"] == "structured_table"
    assert nested_child["content"]["columns"] == [
        {"id": "c1", "text": "항목"},
        {"id": "c2", "text": "금액"},
    ]
    assert table.source.text == (
        "table: 2 columns\n"
        "header 1: col 1: 구분; col 2: 세부\n"
        "row 1: 구분: 본인부담; "
        "세부: nested table: table: 2 columns / "
        "header 1: col 1: 항목; col 2: 금액 / row 1: 항목: 외래; 금액: 1000"
    )

    image = parsed.units[2]
    assert image.source.kind == "image"
    assert image.format == "asset_ref"
    assert image.content == {"asset_id": "img-0001", "caption": None}
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

    monkeypatch.setattr("rag_document_parser.extract.assets._put_object", fake_put_object)

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
    assert result.units[0].content["uri"] == result.assets[0].uri
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
    image_child = table_unit.content["rows"][0]["cells"][1]["children"][0]
    assert image_child == {
        "type": "image",
        "format": "asset_ref",
        "content": {"asset_id": "img-0001", "caption": None},
    }
    assert parsed.assets[0].id == "img-0001"
    assert "image: img-0001" in table_unit.source.text


def test_nested_asset_refs_are_uploaded_and_resolved_in_table_evidence(monkeypatch):
    from rag_document_parser import RagDocumentParser

    uploads = []

    def fake_put_object(cfg, key, data, content_type):
        uploads.append((key, data, content_type))
        return f"s3://{cfg.bucket}/{cfg.prefix}/{key}"

    monkeypatch.setattr("rag_document_parser.extract.assets._put_object", fake_put_object)

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
    image_child = result.units[0].content["rows"][0]["cells"][1]["children"][0]
    assert image_child["content"]["uri"] == (
        f"s3://rag-assets/documents/{document_hash}/assets/img-0001.png"
    )
    assert image_child["content"]["sha256"] == hashlib.sha256(PNG_BYTES).hexdigest()


def test_legacy_nested_kind_asset_ref_is_resolved_and_canonicalized():
    from rag_document_parser.extract.assets import resolve_units
    from rag_document_parser.models import DocumentAsset, EvidenceUnit, SourceEvidence

    unit = EvidenceUnit(
        id="b1",
        type="table",
        format="structured_table",
        source=SourceEvidence(kind="table", text="table with legacy image"),
        content={
            "caption": None,
            "columns": [{"id": "c1", "text": "이미지"}],
            "rows": [
                {
                    "index": 1,
                    "cells": [
                        {
                            "column_id": "c1",
                            "text": "",
                            "rowspan": 1,
                            "colspan": 1,
                            "children": [
                                {
                                    "kind": "image",
                                    "format": "asset_ref",
                                    "content": {
                                        "asset_id": "img-0001",
                                        "caption": "legacy nested",
                                    },
                                }
                            ],
                        }
                    ],
                }
            ],
        },
    )
    asset = DocumentAsset(
        id="img-0001",
        kind="image",
        uri="s3://rag-assets/documents/doc-sha/assets/img-0001.png",
        mime="image/png",
        ext="png",
        sha256=hashlib.sha256(PNG_BYTES).hexdigest(),
        bytes=len(PNG_BYTES),
    )

    resolved = resolve_units([unit], [asset])

    image_child = resolved[0].content["rows"][0]["cells"][0]["children"][0]
    assert image_child == {
        "type": "image",
        "format": "asset_ref",
        "content": {
            "asset_id": "img-0001",
            "caption": "legacy nested",
            "uri": "s3://rag-assets/documents/doc-sha/assets/img-0001.png",
            "mime": "image/png",
            "ext": "png",
            "sha256": hashlib.sha256(PNG_BYTES).hexdigest(),
            "bytes": len(PNG_BYTES),
        },
    }


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

    table_content = parsed.units[0].content
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

    content = parsed.units[0].content
    assert len(content["columns"]) == 6
    assert [column["text"] for column in content["columns"]] == [
        "",
        "",
        "관련 근거",
        "관련 근거",
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
    assert parsed.units[0].source.text == (
        "table: 6 columns\n"
        "header 1: cols 3-4: 관련 근거\n"
        "row 1: col 1: 개정 ’16.11.7.; col 2: 고시 제2016-149호; "
        "관련 근거: (2016.10.01.시행); cols 4-6: 1차 QA"
    )


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
    assert parsed.units[0].format == "plain"
    assert parsed.units[0].content == "목  차"


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

    content = parsed.units[0].content
    assert len(content["header_rows"]) == 2
    assert len(content["rows"]) == 1
    assert content["header_rows"][1]["cells"][0]["text"] == "영상"
    assert content["rows"][0]["cells"][0]["text"] == "진단"


def test_hwpx_table_propagates_colspan_header_groups_to_leaf_columns():
    from rag_document_parser import HwpxBackend

    table = _table(
        [
            _text_cell("현행", colspan=3, row_addr=0, col_addr=0),
            _text_cell("개정", colspan=3, row_addr=0, col_addr=3),
            _text_cell("비고", rowspan=2, row_addr=0, col_addr=6),
        ],
        [
            _text_cell("항목", row_addr=1, col_addr=0),
            _text_cell("제목", row_addr=1, col_addr=1),
            _text_cell("세부인정사항", row_addr=1, col_addr=2),
            _text_cell("항목", row_addr=1, col_addr=3),
            _text_cell("제목", row_addr=1, col_addr=4),
            _text_cell("세부인정사항", row_addr=1, col_addr=5),
        ],
        [
            _text_cell("일반사항", row_addr=2, col_addr=0),
            _text_cell("자연분만", row_addr=2, col_addr=1),
            _text_cell("현행 기준", row_addr=2, col_addr=2),
            _text_cell("일반사항", row_addr=2, col_addr=3),
            _text_cell("제왕절개", row_addr=2, col_addr=4),
            _text_cell("개정 기준", row_addr=2, col_addr=5),
            _text_cell("문구 수정", row_addr=2, col_addr=6),
        ],
        [
            _text_cell("I. 행위 일반사항", colspan=3, row_addr=3, col_addr=0),
            _text_cell("I. 행위 일반사항", colspan=3, row_addr=3, col_addr=3),
            _text_cell("동일", row_addr=3, col_addr=6),
        ],
    )
    xml = (
        f'<hp:sec xmlns:hp="{HP}">'
        f"<hp:p><hp:run>{table}</hp:run></hp:p>"
        "</hp:sec>"
    )

    parsed = HwpxBackend().parse(_make_hwpx(xml), ".hwpx")

    content = parsed.units[0].content
    assert [column["text"] for column in content["columns"]] == [
        "현행 / 항목",
        "현행 / 제목",
        "현행 / 세부인정사항",
        "개정 / 항목",
        "개정 / 제목",
        "개정 / 세부인정사항",
        "비고",
    ]
    assert parsed.units[0].source.text == (
        "table: 7 columns\n"
        "header 1: cols 1-3: 현행; cols 4-6: 개정; col 7: 비고\n"
        "header 2: col 1: 항목; col 2: 제목; col 3: 세부인정사항; "
        "col 4: 항목; col 5: 제목; col 6: 세부인정사항\n"
        "row 1: 현행 / 항목: 일반사항; 현행 / 제목: 자연분만; "
        "현행 / 세부인정사항: 현행 기준; 개정 / 항목: 일반사항; "
        "개정 / 제목: 제왕절개; 개정 / 세부인정사항: 개정 기준; 비고: 문구 수정\n"
        "row 2: 현행: I. 행위 일반사항; 개정: I. 행위 일반사항; 비고: 동일"
    )


def test_hwpx_table_treats_colspan_only_second_row_as_header():
    from rag_document_parser import HwpxBackend

    table = _table(
        [
            _text_cell("검사", colspan=2, row_addr=0, col_addr=0),
            _text_cell("결과", colspan=2, row_addr=0, col_addr=2),
        ],
        [
            _text_cell("일반", row_addr=1, col_addr=0),
            _text_cell("정밀", row_addr=1, col_addr=1),
            _text_cell("판정", row_addr=1, col_addr=2),
            _text_cell("근거", row_addr=1, col_addr=3),
        ],
        [
            _text_cell("복부", row_addr=2, col_addr=0),
            _text_cell("심장", row_addr=2, col_addr=1),
            _text_cell("급여", row_addr=2, col_addr=2),
            _text_cell("고시", row_addr=2, col_addr=3),
        ],
    )
    xml = (
        f'<hp:sec xmlns:hp="{HP}">'
        f"<hp:p><hp:run>{table}</hp:run></hp:p>"
        "</hp:sec>"
    )

    parsed = HwpxBackend().parse(_make_hwpx(xml), ".hwpx")

    content = parsed.units[0].content
    assert [column["text"] for column in content["columns"]] == [
        "검사 / 일반",
        "검사 / 정밀",
        "결과 / 판정",
        "결과 / 근거",
    ]
    assert len(content["header_rows"]) == 2
    assert [row["index"] for row in content["rows"]] == [1]
    assert parsed.units[0].source.text == (
        "table: 4 columns\n"
        "header 1: cols 1-2: 검사; cols 3-4: 결과\n"
        "header 2: col 1: 일반; col 2: 정밀; col 3: 판정; col 4: 근거\n"
        "row 1: 검사 / 일반: 복부; 검사 / 정밀: 심장; "
        "결과 / 판정: 급여; 결과 / 근거: 고시"
    )


def test_hwpx_table_does_not_promote_grouped_code_rows_to_headers():
    from rag_document_parser import HwpxBackend

    table = _table(
        [
            _text_cell("구분", colspan=2, row_addr=0, col_addr=0),
            _text_cell("EDI코드", row_addr=0, col_addr=2),
        ],
        [
            _text_cell("기본 초음파", rowspan=2, row_addr=1, col_addr=0),
            _text_cell("단순초음파(Ⅰ)", row_addr=1, col_addr=1),
            _text_cell("EB401", row_addr=1, col_addr=2),
        ],
        [
            _text_cell("단순초음파(Ⅱ)", row_addr=2, col_addr=1),
            _text_cell("EB402", row_addr=2, col_addr=2),
        ],
        [
            _text_cell("진단 초음파", rowspan=2, row_addr=3, col_addr=0),
            _text_cell("간·담낭·담도·비장·췌장(일반)", row_addr=3, col_addr=1),
            _text_cell("EB441", row_addr=3, col_addr=2),
        ],
        [
            _text_cell("간·담낭·담도·비장·췌장(정밀)", row_addr=4, col_addr=1),
            _text_cell("EB442", row_addr=4, col_addr=2),
        ],
    )
    xml = (
        f'<hp:sec xmlns:hp="{HP}">'
        f"<hp:p><hp:run>{table}</hp:run></hp:p>"
        "</hp:sec>"
    )

    parsed = HwpxBackend().parse(_make_hwpx(xml), ".hwpx")

    content = parsed.units[0].content
    assert [column["text"] for column in content["columns"]] == [
        "구분",
        "구분",
        "EDI코드",
    ]
    assert len(content["header_rows"]) == 1
    assert [
        (cell["column_id"], cell["text"], cell["rowspan"], cell["colspan"])
        for cell in content["header_rows"][0]["cells"]
    ] == [
        ("c1", "구분", 1, 2),
        ("c3", "EDI코드", 1, 1),
    ]
    assert [
        [
            (cell["column_id"], cell["text"], cell["rowspan"], cell["colspan"])
            for cell in row["cells"]
        ]
        for row in content["rows"]
    ] == [
        [
            ("c1", "기본 초음파", 2, 1),
            ("c2", "단순초음파(Ⅰ)", 1, 1),
            ("c3", "EB401", 1, 1),
        ],
        [
            ("c2", "단순초음파(Ⅱ)", 1, 1),
            ("c3", "EB402", 1, 1),
        ],
        [
            ("c1", "진단 초음파", 2, 1),
            ("c2", "간·담낭·담도·비장·췌장(일반)", 1, 1),
            ("c3", "EB441", 1, 1),
        ],
        [
            ("c2", "간·담낭·담도·비장·췌장(정밀)", 1, 1),
            ("c3", "EB442", 1, 1),
        ],
    ]
    assert parsed.units[0].source.text == (
        "table: 3 columns\n"
        "header 1: cols 1-2: 구분; col 3: EDI코드\n"
        "row 1: 구분 [1]: 기본 초음파; 구분 [2]: 단순초음파(Ⅰ); EDI코드: EB401\n"
        "row 2: 구분 [2]: 단순초음파(Ⅱ); EDI코드: EB402\n"
        "row 3: 구분 [1]: 진단 초음파; 구분 [2]: 간·담낭·담도·비장·췌장(일반); "
        "EDI코드: EB441\n"
        "row 4: 구분 [2]: 간·담낭·담도·비장·췌장(정밀); EDI코드: EB442"
    )


def test_hwpx_drawing_shapes_become_structured_diagram():
    from rag_document_parser import HwpxBackend

    xml = (
        f'<hp:sec xmlns:hp="{HP}">'
        "<hp:p><hp:run>"
        f"{_rect('수급권자', x=100, y=100, width=200, height=100)}"
        f"{_line(x=300, y=145, width=200, height=10, arrow=True)}"
        f"{_rect('심사평가원', x=500, y=100, width=200, height=100)}"
        "</hp:run></hp:p>"
        "</hp:sec>"
    )

    parsed = HwpxBackend().parse(_make_hwpx(xml), ".hwpx")

    assert [unit.type for unit in parsed.units] == ["diagram"]
    diagram = parsed.units[0]
    assert diagram.format == "structured_diagram"
    assert diagram.source.kind == "diagram"
    assert diagram.metadata["common"]["display_format"] == "structured_diagram"
    assert [node["text"] for node in diagram.content["nodes"]] == [
        "수급권자",
        "심사평가원",
    ]
    assert diagram.content["nodes"][0]["bbox"] == {
        "x": 100,
        "y": 100,
        "width": 200,
        "height": 100,
        "unit": "hwpx",
    }
    assert diagram.content["connectors"] == [
        {
            "id": "c1",
            "type": "line",
            "bbox": {
                "x": 300,
                "y": 145,
                "width": 200,
                "height": 10,
                "unit": "hwpx",
            },
            "points": [{"x": 300, "y": 150}, {"x": 500, "y": 150}],
            "arrow": True,
            "metadata": {"source": "hwpx_line"},
        }
    ]
    assert diagram.content["edges"] == [
        {
            "from": "n1",
            "to": "n2",
            "type": "arrow",
            "label": "",
            "confidence": "inferred_geometry",
            "connector_id": "c1",
        }
    ]
    assert diagram.content["mermaid"] is None
    assert "relations:\nn1 -> n2" in diagram.source.text
    assert parsed.quality_warnings == []


def test_hwpx_single_text_box_stays_text_unit():
    from rag_document_parser import HwpxBackend

    xml = (
        f'<hp:sec xmlns:hp="{HP}">'
        "<hp:p><hp:run>"
        f"{_rect('단일 텍스트박스', x=100, y=100, width=200, height=100)}"
        "</hp:run></hp:p>"
        "</hp:sec>"
    )

    parsed = HwpxBackend().parse(_make_hwpx(xml), ".hwpx")

    assert [unit.type for unit in parsed.units] == ["text"]
    assert parsed.units[0].source.text == "단일 텍스트박스"
    assert parsed.units[0].format == "plain"
    assert parsed.quality_warnings == []


def test_hwpx_table_cell_diagram_is_preserved_as_nested_evidence():
    from rag_document_parser import HwpxBackend

    diagram_xml = (
        f"{_rect('수급권자', x=100, y=100, width=200, height=100)}"
        f"{_line(x=300, y=145, width=200, height=10, arrow=True)}"
        f"{_rect('심사평가원', x=500, y=100, width=200, height=100)}"
    )
    table = _table(
        [_text_cell("구분"), _text_cell("흐름도")],
        [_text_cell("처리"), _table_cell(diagram_xml)],
    )
    xml = (
        f'<hp:sec xmlns:hp="{HP}">'
        f"<hp:p><hp:run>{table}</hp:run></hp:p>"
        "</hp:sec>"
    )

    parsed = HwpxBackend().parse(_make_hwpx(xml), ".hwpx")

    table_unit = parsed.units[0]
    diagram_child = table_unit.content["rows"][0]["cells"][1]["children"][0]
    assert diagram_child["type"] == "diagram"
    assert diagram_child["format"] == "structured_diagram"
    assert [node["text"] for node in diagram_child["content"]["nodes"]] == [
        "수급권자",
        "심사평가원",
    ]
    assert diagram_child["content"]["edges"][0]["from"] == "n1"
    assert diagram_child["content"]["edges"][0]["to"] == "n2"
    assert "diagram: 수급권자 / 심사평가원" in table_unit.source.text


def test_hwpx_broken_image_reference_is_reported_as_quality_warning():
    from rag_document_parser import HwpxBackend

    xml = (
        f'<hp:sec xmlns:hp="{HP}" xmlns:hc="http://www.hancom.co.kr/hwpml/2011/core">'
        '<hp:p><hp:run><hp:pic><hc:img binaryItemIDRef="missing" /></hp:pic></hp:run></hp:p>'
        "</hp:sec>"
    )

    parsed = HwpxBackend().parse(_make_hwpx(xml), ".hwpx")

    assert parsed.units == []
    assert parsed.assets == []
    assert parsed.quality_warnings == [
        {
            "type": "hwpx_image_reference_unresolved",
            "severity": "medium",
            "ref": "missing",
            "message": "HWPX image reference could not be resolved: missing",
        }
    ]


def test_hwpx_unsupported_drawing_structure_is_reported_as_quality_warning():
    from rag_document_parser import HwpxBackend

    xml = (
        f'<hp:sec xmlns:hp="{HP}">'
        '<hp:p><hp:run><hp:arc><hp:pos x="10" y="20" />'
        '<hp:sz width="100" height="50" /></hp:arc></hp:run></hp:p>'
        "</hp:sec>"
    )

    parsed = HwpxBackend().parse(_make_hwpx(xml), ".hwpx")

    assert parsed.units == []
    assert parsed.quality_warnings == [
        {
            "type": "hwpx_drawing_structure_unsupported",
            "severity": "medium",
            "element": "arc",
            "message": "Unsupported HWPX drawing structure was skipped: arc",
        }
    ]


def test_hwpx_image_only_document_uses_ocr_fallback_without_duplicate_native_text():
    from rag_document_parser import HwpxBackend

    calls: list[tuple[bytes, int]] = []

    def ocr_fn(data: bytes, image_index: int) -> str:
        calls.append((data, image_index))
        return "스캔 이미지 OCR"

    image_only = (
        f'<hp:sec xmlns:hp="{HP}" xmlns:hc="http://www.hancom.co.kr/hwpml/2011/core">'
        '<hp:p><hp:run><hp:pic><hc:img binaryItemIDRef="img1" /></hp:pic></hp:run></hp:p>'
        "</hp:sec>"
    )
    parsed = HwpxBackend(ocr_fn=ocr_fn).parse(
        _make_hwpx(image_only, image_bytes=PNG_BYTES),
        ".hwpx",
    )

    assert [unit.type for unit in parsed.units] == ["image", "text"]
    assert parsed.units[1].source.text == "스캔 이미지 OCR"
    assert calls == [(PNG_BYTES, 0)]
    assert parsed.quality_warnings == []

    with_native_text = (
        f'<hp:sec xmlns:hp="{HP}" xmlns:hc="http://www.hancom.co.kr/hwpml/2011/core">'
        f"<hp:p>{_run('기존 본문')}</hp:p>"
        '<hp:p><hp:run><hp:pic><hc:img binaryItemIDRef="img1" /></hp:pic></hp:run></hp:p>'
        "</hp:sec>"
    )
    calls.clear()

    parsed = HwpxBackend(ocr_fn=ocr_fn).parse(
        _make_hwpx(with_native_text, image_bytes=PNG_BYTES),
        ".hwpx",
    )

    assert [unit.source.text for unit in parsed.units] == [
        "기존 본문",
        "image: img-0001",
    ]
    assert calls == []


def test_hwpx_ocr_failure_is_reported_as_quality_warning():
    from rag_document_parser import HwpxBackend

    def fail_ocr(data: bytes, image_index: int) -> str:
        raise RuntimeError("ocr boom")

    xml = (
        f'<hp:sec xmlns:hp="{HP}" xmlns:hc="http://www.hancom.co.kr/hwpml/2011/core">'
        '<hp:p><hp:run><hp:pic><hc:img binaryItemIDRef="img1" /></hp:pic></hp:run></hp:p>'
        "</hp:sec>"
    )

    parsed = HwpxBackend(ocr_fn=fail_ocr).parse(
        _make_hwpx(xml, image_bytes=PNG_BYTES),
        ".hwpx",
    )

    assert [unit.type for unit in parsed.units] == ["image"]
    assert parsed.quality_warnings == [
        {
            "type": "hwpx_ocr_failed",
            "severity": "medium",
            "image_index": 1,
            "asset_id": "img-0001",
            "message": "ocr boom",
        }
    ]
