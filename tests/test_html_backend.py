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
