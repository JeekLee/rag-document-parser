from __future__ import annotations


def test_render_evidence_units_as_html_without_llm_enrichment():
    from rag_document_parser.evidence_html import render_evidence_units_html

    units = [
        {
            "id": "b1",
            "type": "table",
            "format": "structured_table",
            "source": {"kind": "table", "text": "columns: 구분 | 세부"},
            "content": {
                "caption": None,
                "columns": [
                    {"id": "c1", "text": "구분"},
                    {"id": "c2", "text": "세부"},
                ],
                "rows": [
                    {
                        "index": 1,
                        "cells": [
                            {
                                "column_id": "c1",
                                "text": "본인부담",
                                "rowspan": 1,
                                "colspan": 1,
                                "children": [],
                            },
                            {
                                "column_id": "c2",
                                "text": "",
                                "rowspan": 1,
                                "colspan": 1,
                                "children": [
                                    {
                                        "type": "table",
                                        "format": "structured_table",
                                        "content": {
                                            "caption": None,
                                            "columns": [
                                                {"id": "c1", "text": "항목"},
                                                {"id": "c2", "text": "금액"},
                                            ],
                                            "rows": [
                                                {
                                                    "index": 1,
                                                    "cells": [
                                                        {
                                                            "column_id": "c1",
                                                            "text": "외래",
                                                            "rowspan": 1,
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
                                                }
                                            ],
                                        },
                                    }
                                ],
                            },
                        ],
                    }
                ],
            },
            "metadata": {},
        },
        {
            "id": "b2",
            "type": "image",
            "format": "asset_ref",
            "source": {"kind": "image", "text": "image: img-0001"},
            "content": {
                "asset_id": "img-0001",
                "caption": "첨부 이미지",
            },
            "metadata": {},
        },
    ]

    html = render_evidence_units_html(
        units,
        title="HWPX evidence",
        assets=[
            {
                "id": "img-0001",
                "uri": "s3://rag-assets/doc/assets/img-0001.png",
                "mime": "image/png",
                "ext": "png",
                "sha256": "def456",
                "bytes": 10,
            }
        ],
    )

    assert "<!doctype html>" in html
    assert "HWPX evidence" in html
    assert html.count("<table") == 2
    assert "본인부담" in html
    assert "외래" in html
    assert "s3://rag-assets/doc/assets/img-0001.png" in html
    assert "첨부 이미지" in html


def test_render_composite_chunk_evidence_items():
    from rag_document_parser.evidence_html import render_evidence_html

    html = render_evidence_html(
        {
            "items": [
                {
                    "type": "text",
                    "format": "plain",
                    "content": "청크 설명",
                    "source_unit_ids": ["b1"],
                    "metadata": {},
                },
                {
                    "type": "table",
                    "format": "structured_table",
                    "content": {
                        "caption": None,
                        "columns": [{"id": "c1", "text": "항목"}],
                        "rows": [
                            {
                                "index": 1,
                                "cells": [
                                    {
                                        "column_id": "c1",
                                        "text": "급여",
                                        "rowspan": 1,
                                        "colspan": 1,
                                        "children": [],
                                    }
                                ],
                            }
                        ],
                    },
                    "source_unit_ids": ["b2"],
                    "metadata": {},
                },
            ]
        }
    )

    assert "청크 설명" in html
    assert "급여" in html
    assert html.count("<table") == 1


def test_render_evidence_unit_prefers_direct_content_over_legacy_evidence():
    from rag_document_parser.evidence_html import render_evidence_units_html

    html = render_evidence_units_html(
        [
            {
                "id": "b1",
                "type": "text",
                "format": "plain",
                "source": {"kind": "text", "text": "direct source"},
                "content": "direct content",
                "evidence": {
                    "kind": "text",
                    "format": "plain",
                    "content": "legacy content",
                },
                "metadata": {},
            }
        ],
        title="direct precedence",
    )

    assert "direct content" in html
    assert "legacy content" not in html


def test_render_legacy_evidence_when_unit_only_has_top_level_type():
    from rag_document_parser.evidence_html import render_evidence_units_html

    html = render_evidence_units_html(
        [
            {
                "id": "b1",
                "type": "image",
                "source": {"kind": "image", "text": "image: img-0001"},
                "evidence": {
                    "kind": "image",
                    "format": "asset_ref",
                    "content": {
                        "asset_id": "img-0001",
                        "caption": "legacy image",
                    },
                },
                "metadata": {},
            }
        ],
        title="legacy fallback",
        assets=[
            {
                "id": "img-0001",
                "uri": "s3://bucket/doc/img.png",
                "public_url": "http://example.test/img.png",
                "mime": "image/png",
                "ext": "png",
                "sha256": "abc",
                "bytes": 3,
            }
        ],
    )

    assert "legacy image" in html
    assert 'src="http://example.test/img.png"' in html
    assert "None" not in html


def test_render_evidence_image_uses_public_url_while_showing_source_uri():
    from rag_document_parser.evidence_html import render_evidence_units_html

    units = [
        {
            "id": "b1",
            "source": {"kind": "image", "text": "image: img-0001"},
            "evidence": {
                "kind": "image",
                "format": "asset_ref",
                "content": {
                    "asset_id": "img-0001",
                    "caption": "첨부 이미지",
                },
            },
            "metadata": {},
        }
    ]

    html = render_evidence_units_html(
        units,
        title="HWPX evidence",
        assets=[
            {
                "id": "img-0001",
                "uri": "s3://rag-assets/doc/assets/img-0001.png",
                "public_url": "http://203.0.113.10:10190/rag-assets/doc/assets/img-0001.png",
                "mime": "image/png",
                "ext": "png",
                "sha256": "def456",
                "bytes": 10,
            }
        ],
    )

    assert (
        'src="http://203.0.113.10:10190/rag-assets/doc/assets/img-0001.png"'
        in html
    )
    assert "s3://rag-assets/doc/assets/img-0001.png" in html


def test_render_structured_table_uses_header_rows_with_spans():
    from rag_document_parser.evidence_html import render_evidence_html

    html = render_evidence_html(
        {
            "kind": "table",
            "format": "structured_table",
            "content": {
                "caption": None,
                "columns": [
                    {"id": "c1", "text": ""},
                    {"id": "c2", "text": ""},
                    {"id": "c3", "text": "관련 근거"},
                    {"id": "c4", "text": "관련 근거"},
                ],
                "header_rows": [
                    {
                        "index": 1,
                        "cells": [
                            {
                                "column_id": "c1",
                                "text": "",
                                "rowspan": 1,
                                "colspan": 1,
                                "children": [],
                            },
                            {
                                "column_id": "c2",
                                "text": "",
                                "rowspan": 1,
                                "colspan": 1,
                                "children": [],
                            },
                            {
                                "column_id": "c3",
                                "text": "관련 근거",
                                "rowspan": 2,
                                "colspan": 2,
                                "children": [],
                            },
                        ],
                    },
                    {
                        "index": 2,
                        "cells": [
                            {
                                "column_id": "c1",
                                "text": "",
                                "rowspan": 1,
                                "colspan": 1,
                                "children": [],
                            },
                            {
                                "column_id": "c2",
                                "text": "",
                                "rowspan": 1,
                                "colspan": 1,
                                "children": [],
                            },
                        ],
                    },
                ],
                "rows": [
                    {
                        "index": 1,
                        "cells": [
                            {
                                "column_id": "c1",
                                "text": "개정",
                                "rowspan": 1,
                                "colspan": 1,
                                "children": [],
                            },
                            {
                                "column_id": "c2",
                                "text": "고시",
                                "rowspan": 1,
                                "colspan": 1,
                                "children": [],
                            },
                            {
                                "column_id": "c3",
                                "text": "시행",
                                "rowspan": 1,
                                "colspan": 1,
                                "children": [],
                            },
                            {
                                "column_id": "c4",
                                "text": "QA",
                                "rowspan": 1,
                                "colspan": 1,
                                "children": [],
                            },
                        ],
                    }
                ],
            },
        }
    )

    assert '<th rowspan="2" colspan="2">관련 근거</th>' in html
