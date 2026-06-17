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


def test_render_evidence_image_uses_public_url_while_showing_source_uri():
    from rag_document_parser.evidence_html import render_evidence_units_html

    units = [
        {
            "id": "b1",
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


def test_render_structured_table_fills_column_gaps_from_cell_ids():
    from rag_document_parser.evidence_html import render_evidence_html

    html = render_evidence_html(
        {
            "kind": "table",
            "format": "structured_table",
            "content": {
                "caption": None,
                "columns": [
                    {"id": "c1", "text": "A"},
                    {"id": "c2", "text": ""},
                    {"id": "c3", "text": ""},
                    {"id": "c4", "text": "D"},
                ],
                "header_rows": [],
                "rows": [
                    {
                        "index": 1,
                        "cells": [
                            {
                                "column_id": "c1",
                                "text": "left",
                                "rowspan": 1,
                                "colspan": 1,
                                "children": [],
                            },
                            {
                                "column_id": "c4",
                                "text": "right",
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

    assert "<td>left</td><td>&nbsp;</td><td>&nbsp;</td><td>right</td>" in html


def test_render_structured_diagram_shows_nodes_edges_and_mermaid():
    from rag_document_parser.evidence_html import render_evidence_html

    html = render_evidence_html(
        {
            "kind": "diagram",
            "format": "structured_diagram",
            "content": {
                "caption": None,
                "nodes": [
                    {"id": "n1", "shape_type": "label", "text": "수급권자"},
                    {"id": "n2", "shape_type": "label", "text": "건강보험심사평가원"},
                ],
                "edges": [
                    {
                        "from": "n1",
                        "to": "n2",
                        "type": "arrow",
                        "label": "신청",
                        "confidence": "parsed",
                    }
                ],
                "mermaid": "flowchart TD\n  n1[수급권자] --> n2[건강보험심사평가원]",
            },
        }
    )

    assert 'class="diagram-evidence"' in html
    assert "수급권자" in html
    assert "건강보험심사평가원" in html
    assert "n1 → n2" in html
    assert "신청" in html
    assert "flowchart TD" in html
    assert 'class="diagram-shape">label</span>' not in html


def test_render_structured_diagram_uses_bounding_boxes_for_positioned_layout():
    from rag_document_parser.evidence_html import render_evidence_html

    html = render_evidence_html(
        {
            "kind": "diagram",
            "format": "structured_diagram",
            "content": {
                "caption": None,
                "nodes": [
                    {"id": "n1", "shape_type": "label", "text": "업무처리 흐름도"},
                    {
                        "id": "n2",
                        "shape_type": "label",
                        "text": "수급권자",
                        "bbox": {
                            "x": 100,
                            "y": 200,
                            "width": 300,
                            "height": 100,
                            "unit": "hwp",
                        },
                    },
                    {
                        "id": "n3",
                        "shape_type": "label",
                        "text": "심사평가원",
                        "bbox": {
                            "x": 700,
                            "y": 200,
                            "width": 300,
                            "height": 100,
                            "unit": "hwp",
                        },
                    },
                ],
                "edges": [],
                "mermaid": None,
            },
        }
    )

    assert 'class="diagram-positioned"' in html
    assert 'class="diagram-canvas"' in html
    assert (
        'style="left:0.000%;top:0.000%;width:33.333%;height:100.000%"'
        in html
    )
    assert (
        'style="left:66.667%;top:0.000%;width:33.333%;height:100.000%"'
        in html
    )
    assert "업무처리 흐름도" in html
    assert "수급권자" in html
    assert "심사평가원" in html


def test_render_structured_diagram_draws_positioned_connectors():
    from rag_document_parser.evidence_html import render_evidence_html

    html = render_evidence_html(
        {
            "kind": "diagram",
            "format": "structured_diagram",
            "content": {
                "caption": None,
                "nodes": [
                    {
                        "id": "n1",
                        "shape_type": "label",
                        "text": "수급권자",
                        "bbox": {
                            "x": 100,
                            "y": 200,
                            "width": 300,
                            "height": 100,
                            "unit": "hwp",
                        },
                    },
                    {
                        "id": "n2",
                        "shape_type": "label",
                        "text": "심사평가원",
                        "bbox": {
                            "x": 900,
                            "y": 200,
                            "width": 300,
                            "height": 100,
                            "unit": "hwp",
                        },
                    },
                ],
                "connectors": [
                    {
                        "id": "c1",
                        "type": "line",
                        "bbox": {
                            "x": 400,
                            "y": 250,
                            "width": 500,
                            "height": 0,
                            "unit": "hwp",
                        },
                        "points": [
                            {"x": 400, "y": 250},
                            {"x": 900, "y": 250},
                        ],
                    }
                ],
                "edges": [],
                "mermaid": None,
            },
        }
    )

    assert 'class="diagram-connectors"' in html
    assert (
        '<line x1="27.273%" y1="50.000%" x2="72.727%" y2="50.000%"></line>'
        in html
    )


def test_render_structured_diagram_marks_arrow_connectors():
    from rag_document_parser.evidence_html import render_evidence_html

    html = render_evidence_html(
        {
            "kind": "diagram",
            "format": "structured_diagram",
            "content": {
                "caption": None,
                "nodes": [
                    {
                        "id": "n1",
                        "shape_type": "label",
                        "text": "공단",
                        "bbox": {
                            "x": 100,
                            "y": 100,
                            "width": 200,
                            "height": 100,
                            "unit": "hwp",
                        },
                    },
                    {
                        "id": "n2",
                        "shape_type": "label",
                        "text": "수급권자",
                        "bbox": {
                            "x": 500,
                            "y": 100,
                            "width": 200,
                            "height": 100,
                            "unit": "hwp",
                        },
                    },
                ],
                "connectors": [
                    {
                        "id": "c1",
                        "type": "line",
                        "arrow": True,
                        "points": [
                            {"x": 300, "y": 150},
                            {"x": 500, "y": 150},
                        ],
                    }
                ],
                "edges": [],
                "mermaid": None,
            },
        }
    )

    assert '<marker id="diagram-arrow"' in html
    assert 'marker-end="url(#diagram-arrow)"' in html


def test_render_structured_diagram_places_step_labels_near_connectors():
    from rag_document_parser.evidence_html import render_evidence_html

    html = render_evidence_html(
        {
            "kind": "diagram",
            "format": "structured_diagram",
            "content": {
                "caption": None,
                "nodes": [
                    {"id": "n1", "shape_type": "label", "text": "업무처리 흐름도"},
                    {"id": "n2", "shape_type": "label", "text": "①신청"},
                    {
                        "id": "n3",
                        "shape_type": "label",
                        "text": "수급권자",
                        "bbox": {
                            "x": 100,
                            "y": 100,
                            "width": 200,
                            "height": 100,
                            "unit": "hwp",
                        },
                    },
                    {
                        "id": "n4",
                        "shape_type": "label",
                        "text": "심사평가원",
                        "bbox": {
                            "x": 500,
                            "y": 100,
                            "width": 200,
                            "height": 100,
                            "unit": "hwp",
                        },
                    },
                ],
                "connectors": [
                    {
                        "id": "c1",
                        "type": "line",
                        "points": [
                            {"x": 300, "y": 150},
                            {"x": 500, "y": 150},
                        ],
                    }
                ],
                "edges": [],
                "mermaid": None,
            },
        }
    )

    assert (
        'class="diagram-connector-label" style="left:50.000%;top:50.000%"'
        in html
    )
    assert 'class="diagram-positioned-label">①신청</span>' not in html
    assert "업무처리 흐름도" in html


def test_render_label_only_diagram_as_original_like_flowchart():
    from rag_document_parser.evidence_html import render_evidence_html

    labels = [
        "업무처리 흐름도",
        "< 과다본인부담금 확인절차 >",
        "①급여대상여부확인신청",
        "수급권자",
        "건강보험심사평가원",
        "②심사결정통보",
        "의료급여기관",
        "국민건강보험공단",
        "수급권자",
        "(국민건강보험공단 및 의료급여기관에는 과다 징수금액이 있는 경우에 한해 통보)",
        "< 과다본인부담금 반환절차 >",
        "①과다본인부담금 지체없이반환",
        "의료급여기관",
        "수급권자",
        "⑤차기 지급진료비에서 공제 ②미반환시 신고",
        "⑥공제금 지급 ③과다본인부담금",
        "공제예정 통보",
        "④과다본인부담금 공제요청",
        "국민건강보험공단",
        "건강보험심사평가원",
        "⑦처리결과 통보",
        "⑦처리결과",
        "통보",
        "보장기관",
    ]

    html = render_evidence_html(
        {
            "kind": "diagram",
            "format": "structured_diagram",
            "content": {
                "caption": None,
                "nodes": [
                    {"id": f"n{index}", "shape_type": "label", "text": label}
                    for index, label in enumerate(labels, start=1)
                ],
                "edges": [],
                "mermaid": None,
            },
        }
    )

    assert 'class="diagram-evidence diagram-flowchart"' in html
    assert '<ol class="diagram-nodes">' not in html
    assert "<code>n1</code>" not in html
    assert 'class="diagram-flowchart-title">업무처리 흐름도</div>' in html
    assert (
        'class="diagram-flowchart-section-title">&lt; 과다본인부담금 확인절차 &gt;'
        in html
    )
    assert 'class="diagram-flowchart-step">①급여대상여부확인신청</div>' in html
    assert 'class="diagram-flowchart-box">수급권자</div>' in html
    assert 'class="diagram-flowchart-arrow">→</span>' in html
    assert "⑥공제금 지급 ③과다본인부담금<br>공제예정 통보" in html
    assert "⑦처리결과<br>통보" in html
    assert 'class="diagram-flowchart-note">' in html
