from __future__ import annotations


def test_html_backend_extracts_text_sections_links_and_lists():
    from rag_document_parser import HtmlBackend

    raw = b"""
    <html><body>
      <h1>Coverage Rules</h1>
      <p>Apply the <a href="https://example.test/rule">rule</a> today.</p>
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
