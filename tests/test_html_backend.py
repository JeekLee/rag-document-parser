from __future__ import annotations

import base64

PNG_BYTES = b"png bytes"


def _data_uri(data: bytes = PNG_BYTES, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def test_html_backend_extracts_text_sections_links_and_lists():
    from rag_document_parser import HtmlBackend

    raw = b"""
    <html><body>
      <h1>Coverage Rules</h1>
      <p>Apply <!-- hidden comment --> the <a href="https://example.test/rule">rule</a> today.</p>
      <ul><li>First item</li><li>Second item</li></ul>
      <blockquote>Quoted guidance</blockquote>
      <pre>code line 1
code line 2</pre>
    </body></html>
    """

    parsed = HtmlBackend().parse(raw, ".html")

    assert [unit.type for unit in parsed.units] == [
        "text",
        "text",
        "text",
        "text",
        "text",
    ]
    assert [unit.content for unit in parsed.units] == [
        "Apply the rule (https://example.test/rule) today.",
        "First item",
        "Second item",
        "Quoted guidance",
        "code line 1\ncode line 2",
    ]
    assert all(unit.format == "plain" for unit in parsed.units)
    assert all(
        unit.metadata["common"]["section_path"] == ["Coverage Rules"]
        for unit in parsed.units
    )
    assert parsed.units[0].source.text == (
        "section: Coverage Rules\n"
        "Apply the rule (https://example.test/rule) today."
    )


def test_html_backend_extracts_standalone_sectioning_text():
    from rag_document_parser import HtmlBackend

    raw = b"""
    <main>
      <h1>Notice</h1>
      Loose intro text
      <section>
        Nested detail text
        <a href="https://example.test/more">More detail</a>
      </section>
      <script>ignored()</script>
    </main>
    """

    parsed = HtmlBackend().parse(raw, ".html")

    assert [unit.content for unit in parsed.units] == [
        "Loose intro text",
        "Nested detail text",
        "More detail (https://example.test/more)",
    ]
    assert all(
        unit.metadata["common"]["section_path"] == ["Notice"]
        for unit in parsed.units
    )


def test_html_backend_ignores_hira_popup_chrome_and_attachment_box():
    from rag_document_parser import HtmlBackend

    raw = """
    <html><body>
      <!-- popup header start -->
      <div id="popHeader"><h1>보험인정기준 상세내용</h1></div>
      <div class="btnAreaR"><a href="javascript:window.print();">인쇄</a></div>
      <div class="viewCont">
        <div class="title">세균배양검사의 항산균검사 수가산정방법</div>
        <ul class="writer">
          <li><span>분류</span>행정해석</li>
          <li><span>관련근거</span>급여1492-59659호</li>
        </ul>
        <div class="fileBox">
          <ul><li>본문내용.pdf <a href="#none">첨부파일 다운로드</a></li></ul>
        </div>
        <div class="view">
          * 세균배양검사의 항산균검사 수가산정방법<br>
          가. 도말검사는 직접도말을 실시한 경우 산정<br>
          나. 집균도말검사는 원심분리 후 실시한 경우 산정
        </div>
      </div>
      <div id="popFooter">popup footer <a href="javascript:self.close();">닫기</a></div>
      <!-- popup header end -->
    </body></html>
    """.encode()

    parsed = HtmlBackend().parse(raw, ".html")

    assert [unit.content for unit in parsed.units] == [
        "세균배양검사의 항산균검사 수가산정방법",
        "분류 행정해석",
        "관련근거 급여1492-59659호",
        "* 세균배양검사의 항산균검사 수가산정방법",
        "가. 도말검사는 직접도말을 실시한 경우 산정",
        "나. 집균도말검사는 원심분리 후 실시한 경우 산정",
    ]
    assert parsed.units[0].metadata["common"]["section_path"] == []
    assert all(
        unit.metadata["common"]["section_path"]
        == ["세균배양검사의 항산균검사 수가산정방법"]
        for unit in parsed.units[1:]
    )
    assert parsed.units[3].source.text == (
        "section: 세균배양검사의 항산균검사 수가산정방법\n"
        "* 세균배양검사의 항산균검사 수가산정방법"
    )
    joined = "\n".join(str(unit.content) for unit in parsed.units)
    assert "popup" not in joined
    assert "인쇄" not in joined
    assert "첨부파일 다운로드" not in joined
    assert "닫기" not in joined


def test_html_backend_extracts_structured_table_with_caption_and_spans():
    from rag_document_parser import HtmlBackend

    raw = b"""
    <h1>Fee Criteria</h1>
    <table>
      <caption>Copay Table</caption>
      <thead><tr><th>Type</th><th>Amount</th></tr></thead>
      <tbody>
        <tr><td rowspan="2">Clinic</td><td>1000</td></tr>
        <tr><td colspan="1">2000</td></tr>
      </tbody>
    </table>
    """

    parsed = HtmlBackend().parse(raw, ".html")

    assert [unit.type for unit in parsed.units] == ["table"]
    table = parsed.units[0]
    assert table.format == "structured_table"
    assert table.content["caption"] == "Copay Table"
    assert table.content["columns"] == [
        {"id": "c1", "text": "Type"},
        {"id": "c2", "text": "Amount"},
    ]
    assert table.content["rows"] == [
        {
            "index": 1,
            "cells": [
                {
                    "column_id": "c1",
                    "text": "Clinic",
                    "rowspan": 2,
                    "colspan": 1,
                    "children": [],
                },
                {
                    "column_id": "c2",
                    "text": "1000",
                    "rowspan": 1,
                    "colspan": 1,
                    "children": [],
                },
            ],
        },
        {
            "index": 2,
            "cells": [
                {
                    "column_id": "c2",
                    "text": "2000",
                    "rowspan": 1,
                    "colspan": 1,
                    "children": [],
                },
            ],
        },
    ]
    assert table.metadata["common"]["section_path"] == ["Fee Criteria"]
    assert table.metadata["table"] == {
        "table_id": "t1",
        "headers": ["Type", "Amount"],
        "row_count": 2,
    }
    assert table.source.text == (
        "section: Fee Criteria\n"
        "caption: Copay Table\n"
        "columns: Type | Amount\n"
        "row 1: Type=Clinic; Amount=1000\n"
        "row 2: Amount=2000"
    )


def test_html_backend_preserves_figure_data_uri_image_asset():
    from rag_document_parser import HtmlBackend

    raw = f"""
    <h1>Images</h1>
    <figure>
      <img src="{_data_uri()}" alt="chart alt">
      <figcaption>Chart caption</figcaption>
    </figure>
    """.encode()

    parsed = HtmlBackend().parse(raw, ".html")

    assert [unit.type for unit in parsed.units] == ["image"]
    image = parsed.units[0]
    assert image.source.kind == "image"
    assert image.source.text == (
        "section: Images\n"
        "image: img-0001\n"
        "caption: Chart caption\n"
        "alt: chart alt"
    )
    assert image.format == "asset_ref"
    assert image.content == {"asset_id": "img-0001", "caption": "Chart caption"}
    assert parsed.assets[0].id == "img-0001"
    assert parsed.assets[0].data == PNG_BYTES
    assert parsed.assets[0].mime == "image/png"
    assert parsed.assets[0].ext == "png"


def test_html_backend_warns_for_external_and_invalid_images():
    from rag_document_parser import HtmlBackend

    raw = b'''
    <img src="https://example.test/image.png" alt="remote">
    <img src="data:image/png;base64,not-valid" alt="bad">
    <img src="data:image/svg+xml;base64,PHN2Zy8+" alt="svg">
    '''

    parsed = HtmlBackend().parse(raw, ".html")

    assert parsed.units == []
    assert parsed.assets == []
    assert [warning["type"] for warning in parsed.quality_warnings] == [
        "html_image_external_reference",
        "html_image_data_uri_invalid",
        "html_image_mime_unsupported",
    ]


def test_html_backend_preserves_nested_table_as_table_child():
    from rag_document_parser import HtmlBackend

    raw = b"""
    <table>
      <tr><th>Item</th><th>Detail</th></tr>
      <tr>
        <td>Criteria</td>
        <td>
          <table>
            <tr><th>Subitem</th><th>Value</th></tr>
            <tr><td>A</td><td>1</td></tr>
          </table>
        </td>
      </tr>
    </table>
    """

    parsed = HtmlBackend().parse(raw, ".html")

    table = parsed.units[0]
    nested_child = table.content["rows"][0]["cells"][1]["children"][0]
    assert nested_child["type"] == "table"
    assert nested_child["format"] == "structured_table"
    assert nested_child["content"]["columns"] == [
        {"id": "c1", "text": "Subitem"},
        {"id": "c2", "text": "Value"},
    ]
    assert nested_child["content"]["rows"][0]["cells"][0]["text"] == "A"
    assert "nested table:" in table.source.text


def test_html_backend_preserves_table_cell_image_as_nested_asset_ref():
    from rag_document_parser import HtmlBackend

    raw = f"""
    <table>
      <tr><th>Item</th><th>Image</th></tr>
      <tr><td>Criteria</td><td><img src="{_data_uri()}" alt="cell chart"></td></tr>
    </table>
    """.encode()

    parsed = HtmlBackend().parse(raw, ".html")

    table = parsed.units[0]
    image_child = table.content["rows"][0]["cells"][1]["children"][0]
    assert image_child == {
        "type": "image",
        "format": "asset_ref",
        "content": {"asset_id": "img-0001", "caption": "cell chart"},
    }
    assert parsed.assets[0].id == "img-0001"
    assert "image: img-0001" in table.source.text
