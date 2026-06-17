from __future__ import annotations


def test_render_evidence_units_as_html_without_llm_enrichment():
    from rag_document_parser.evidence_html import render_evidence_units_html

    units = [
        {
            "id": "b1",
            "type": "table",
            "source": {"kind": "table", "text": "columns: 구분 | 세부"},
            "evidence": {
                "kind": "table",
                "format": "structured_table",
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
                                            "kind": "table",
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
            },
            "metadata": {},
        },
        {
            "id": "b2",
            "type": "image",
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
