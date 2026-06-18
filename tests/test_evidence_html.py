from __future__ import annotations


def test_render_evidence_units_as_html_without_llm_enrichment():
    from rag_document_parser.renderer.evidence_unit_render import (
        render_evidence_units_html,
    )

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
    from rag_document_parser.renderer.evidence_unit_render import render_evidence_html

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
    from rag_document_parser.renderer.evidence_unit_render import render_evidence_units_html

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
    from rag_document_parser.renderer.evidence_unit_render import render_evidence_units_html

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


def test_render_evidence_image_uses_public_url_for_rendering_and_link_text():
    from rag_document_parser.renderer.evidence_unit_render import render_evidence_units_html

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
    assert ">http://203.0.113.10:10190/rag-assets/doc/assets/img-0001.png</a>" in html
    assert "s3://rag-assets/doc/assets/img-0001.png" not in html


def test_render_structured_table_uses_header_rows_with_spans():
    from rag_document_parser.renderer.evidence_unit_render import render_evidence_html

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


def test_render_structured_table_does_not_fill_rowspanned_columns():
    from rag_document_parser.renderer.evidence_unit_render import render_evidence_html

    html = render_evidence_html(
        {
            "kind": "table",
            "format": "structured_table",
            "content": {
                "caption": None,
                "columns": [
                    {"id": "c1", "text": "최초운영일"},
                    {"id": "c2", "text": "최초운영분기"},
                    {"id": "c3", "text": "최초분기 적용기준일 / 인력"},
                    {"id": "c4", "text": "최초분기 적용기준일 / 병상수"},
                ],
                "header_rows": [
                    {
                        "index": 1,
                        "cells": [
                            {
                                "column_id": "c1",
                                "text": "최초운영일",
                                "rowspan": 2,
                                "colspan": 1,
                                "children": [],
                            },
                            {
                                "column_id": "c2",
                                "text": "최초운영분기",
                                "rowspan": 2,
                                "colspan": 1,
                                "children": [],
                            },
                            {
                                "column_id": "c3",
                                "text": "최초분기 적용기준일",
                                "rowspan": 1,
                                "colspan": 2,
                                "children": [],
                            },
                        ],
                    },
                    {
                        "index": 2,
                        "cells": [
                            {
                                "column_id": "c3",
                                "text": "인력",
                                "rowspan": 1,
                                "colspan": 1,
                                "children": [],
                            },
                            {
                                "column_id": "c4",
                                "text": "병상수",
                                "rowspan": 1,
                                "colspan": 1,
                                "children": [],
                            },
                        ],
                    },
                ],
                "rows": [],
            },
        }
    )

    assert (
        '<tr><th rowspan="2">최초운영일</th><th rowspan="2">최초운영분기</th>'
        '<th colspan="2">최초분기 적용기준일</th></tr>'
        "<tr><th>인력</th><th>병상수</th></tr>"
    ) in html
    assert "<tr><th>&nbsp;</th><th>&nbsp;</th><th>인력</th>" not in html


def test_render_structured_table_does_not_fill_header_gaps_covered_by_rowspan():
    from rag_document_parser.renderer.evidence_unit_render import render_evidence_html

    html = render_evidence_html(
        {
            "kind": "table",
            "format": "structured_table",
            "content": {
                "caption": None,
                "columns": [
                    {"id": "c1", "text": "구 분"},
                    {"id": "c2", "text": "일반식"},
                    {"id": "c3", "text": "치료식"},
                    {"id": "c4", "text": "멸균식"},
                    {"id": "c5", "text": "분 유 / 일반 분유"},
                    {"id": "c6", "text": "분 유 / 특수 분유"},
                ],
                "header_rows": [
                    {
                        "index": 1,
                        "cells": [
                            {
                                "column_id": "c1",
                                "text": "구 분",
                                "rowspan": 2,
                                "colspan": 1,
                                "children": [],
                            },
                            {
                                "column_id": "c2",
                                "text": "일반식",
                                "rowspan": 2,
                                "colspan": 1,
                                "children": [],
                            },
                            {
                                "column_id": "c3",
                                "text": "치료식",
                                "rowspan": 2,
                                "colspan": 1,
                                "children": [],
                            },
                            {
                                "column_id": "c4",
                                "text": "멸균식",
                                "rowspan": 2,
                                "colspan": 1,
                                "children": [],
                            },
                            {
                                "column_id": "c5",
                                "text": "분 유",
                                "rowspan": 1,
                                "colspan": 2,
                                "children": [],
                            },
                        ],
                    },
                    {
                        "index": 2,
                        "cells": [
                            {
                                "column_id": "c5",
                                "text": "일반 분유",
                                "rowspan": 1,
                                "colspan": 1,
                                "children": [],
                            },
                            {
                                "column_id": "c6",
                                "text": "특수 분유",
                                "rowspan": 1,
                                "colspan": 1,
                                "children": [],
                            },
                        ],
                    },
                ],
                "rows": [],
            },
        }
    )

    assert "<tr><th>일반 분유</th><th>특수 분유</th></tr>" in html
    assert "<tr><th>&nbsp;</th><th>&nbsp;</th><th>&nbsp;</th><th>&nbsp;</th>" not in html


def test_render_structured_table_omits_empty_header_rows():
    from rag_document_parser.renderer.evidence_unit_render import render_evidence_html

    html = render_evidence_html(
        {
            "kind": "table",
            "format": "structured_table",
            "content": {
                "caption": None,
                "columns": [
                    {"id": "c1", "text": "[별지29호]"},
                    {"id": "c2", "text": "[별지29호]"},
                ],
                "header_rows": [
                    {
                        "index": 1,
                        "cells": [
                            {
                                "column_id": "c1",
                                "text": "[별지29호]",
                                "rowspan": 1,
                                "colspan": 2,
                                "children": [],
                            }
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
                "rows": [],
            },
        }
    )

    assert '<th colspan="2">[별지29호]</th>' in html
    assert "<tr><th>&nbsp;</th><th>&nbsp;</th></tr>" not in html


def test_render_structured_table_fills_column_gaps_from_cell_ids():
    from rag_document_parser.renderer.evidence_unit_render import render_evidence_html

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


def test_render_structured_table_does_not_fill_cells_covered_by_rowspan():
    from rag_document_parser.renderer.evidence_unit_render import render_evidence_html

    html = render_evidence_html(
        {
            "kind": "table",
            "format": "structured_table",
            "content": {
                "caption": None,
                "columns": [
                    {"id": "c1", "text": "처방명"},
                    {"id": "c2", "text": "한방"},
                    {"id": "c3", "text": "양방"},
                    {"id": "c4", "text": "분류"},
                    {"id": "c5", "text": "적응증"},
                ],
                "header_rows": [],
                "rows": [
                    {
                        "index": 1,
                        "cells": [
                            {
                                "column_id": "c1",
                                "text": "1. 가미소요산",
                                "rowspan": 3,
                                "colspan": 1,
                                "children": [],
                            },
                            {
                                "column_id": "c2",
                                "text": "수 혈",
                                "rowspan": 1,
                                "colspan": 1,
                                "children": [],
                            },
                            {
                                "column_id": "c3",
                                "text": "상세불명",
                                "rowspan": 1,
                                "colspan": 1,
                                "children": [],
                            },
                            {
                                "column_id": "c4",
                                "text": "E13.2",
                                "rowspan": 1,
                                "colspan": 1,
                                "children": [],
                            },
                            {
                                "column_id": "c5",
                                "text": "월경통",
                                "rowspan": 3,
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
                                "text": "산후기울증",
                                "rowspan": 1,
                                "colspan": 1,
                                "children": [],
                            },
                            {
                                "column_id": "c3",
                                "text": "산욕기 장애",
                                "rowspan": 1,
                                "colspan": 1,
                                "children": [],
                            },
                            {
                                "column_id": "c4",
                                "text": "I05.0",
                                "rowspan": 1,
                                "colspan": 1,
                                "children": [],
                            },
                        ],
                    },
                    {
                        "index": 3,
                        "cells": [
                            {
                                "column_id": "c2",
                                "text": "심 화 항 염",
                                "rowspan": 1,
                                "colspan": 1,
                                "children": [],
                            },
                            {
                                "column_id": "c3",
                                "text": "뇌허혈",
                                "rowspan": 1,
                                "colspan": 1,
                                "children": [],
                            },
                            {
                                "column_id": "c4",
                                "text": "C21.1",
                                "rowspan": 1,
                                "colspan": 1,
                                "children": [],
                            },
                        ],
                    },
                ],
            },
        }
    )

    assert (
        "<tr>"
        '<td rowspan="3">1. 가미소요산</td>'
        "<td>수 혈</td><td>상세불명</td><td>E13.2</td>"
        '<td rowspan="3">월경통</td>'
        "</tr>"
        "<tr><td>산후기울증</td><td>산욕기 장애</td><td>I05.0</td></tr>"
        "<tr><td>심 화 항 염</td><td>뇌허혈</td><td>C21.1</td></tr>"
    ) in html
    assert "<tr><td>&nbsp;</td><td>산후기울증</td>" not in html


def test_render_structured_table_preserves_multiline_cell_text():
    from rag_document_parser.renderer.evidence_unit_render import render_evidence_html

    html = render_evidence_html(
        {
            "kind": "table",
            "format": "structured_table",
            "content": {
                "caption": None,
                "columns": [{"id": "c1", "text": "항목"}],
                "header_rows": [],
                "rows": [
                    {
                        "index": 1,
                        "cells": [
                            {
                                "column_id": "c1",
                                "text": "첫 줄\n둘째 줄",
                                "rowspan": 1,
                                "colspan": 1,
                                "children": [],
                            }
                        ],
                    }
                ],
            },
        }
    )

    assert "<td>첫 줄<br>둘째 줄</td>" in html


def test_render_rag_chunks_html_shows_final_evidence_and_chunk_fields():
    from rag_document_parser.renderer.rag_chunk_render import render_rag_chunks_html

    chunks = [
        {
            "id": "chunk-1",
            "type": "mixed",
            "source": {"kind": "mixed", "text": "source text"},
            "evidence": {
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
            },
            "summary": "제왕절개 본인부담률 안내",
            "keywords": ["제왕절개", "본인부담"],
            "questions": ["본인부담률은 어떻게 바뀌나요?"],
            "metadata": {
                "source_unit_ids": ["b1", "b2"],
                "source_units": [
                    {"id": "b1", "type": "text", "format": "plain"},
                    {"id": "b2", "type": "table", "format": "structured_table"},
                ],
                "context_unit_ids": ["b0"],
                "operations": [
                    {"unit_id": "b1", "action": "include"},
                    {"unit_id": "b2", "action": "include"},
                ],
                "_boundary_merges": [
                    {
                        "left_source_unit_ids": ["b1"],
                        "right_source_unit_ids": ["b2"],
                        "reason": "same section",
                    }
                ],
                "_warnings": [
                    {
                        "type": "agentic_chunk_exceeds_max_units",
                        "source_unit_count": 9,
                        "max_units_per_chunk": 8,
                    }
                ],
            },
        }
    ]

    html = render_rag_chunks_html(chunks, title="Agentic chunks")

    assert "<!doctype html>" in html
    assert "Agentic chunks" in html
    assert "chunk-1" in html
    assert "mixed" in html
    assert "제왕절개 본인부담률 안내" in html
    assert "제왕절개" in html
    assert "본인부담률은 어떻게 바뀌나요?" in html
    assert "source units: b1, b2" in html
    assert "context units: b0" in html
    assert "operations" in html
    assert "boundary merges" in html
    assert "same section" in html
    assert "structured_table" in html
    assert "source text" in html
    assert "evidence item 1" in html
    assert "item source units: b1" in html
    assert "evidence item 2" in html
    assert "item source units: b2" in html
    assert "청크 설명" in html
    assert "급여" in html
    assert html.count("<table") == 1
    assert "agentic_chunk_exceeds_max_units" in html


def test_render_rag_chunks_html_accepts_model_objects_and_escapes_diagnostics():
    from rag_document_parser import Evidence, EvidenceItem, RagChunk, SourceEvidence
    from rag_document_parser.renderer.rag_chunk_render import render_rag_chunks_html

    chunk = RagChunk(
        id="chunk-2",
        type="text",
        source=SourceEvidence(kind="text", text="raw <source>"),
        evidence=Evidence(
            items=[
                EvidenceItem(
                    type="text",
                    format="plain",
                    content="safe <content>",
                    source_unit_ids=["b3"],
                    metadata={},
                )
            ]
        ),
        summary="escaped summary",
        metadata={
            "source_unit_ids": ["b3"],
            "_fallback_reason": "bad <script>alert(1)</script>",
            "_rejected_plan": [{"unit_ids": ["b9"]}],
        },
    )

    html = render_rag_chunks_html([chunk])

    assert "chunk-2" in html
    assert "safe &lt;content&gt;" in html
    assert "raw &lt;source&gt;" in html
    assert "bad &lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "&quot;unit_ids&quot;" in html
    assert "<script>" not in html


def test_render_structured_diagram_shows_nodes_edges_and_mermaid():
    from rag_document_parser.renderer.evidence_unit_render import render_evidence_html

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
    from rag_document_parser.renderer.evidence_unit_render import render_evidence_html

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


def test_render_structured_diagram_positions_explicit_shape_nodes():
    from rag_document_parser.renderer.evidence_unit_render import render_evidence_html

    html = render_evidence_html(
        {
            "kind": "diagram",
            "format": "structured_diagram",
            "content": {
                "caption": None,
                "nodes": [
                    {
                        "id": "n1",
                        "shape_type": "rectangle",
                        "text": "접수",
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
                        "shape_type": "ellipse",
                        "text": "완료",
                        "bbox": {
                            "x": 500,
                            "y": 100,
                            "width": 200,
                            "height": 100,
                            "unit": "hwp",
                        },
                    },
                ],
                "edges": [
                    {
                        "from": "n1",
                        "to": "n2",
                        "type": "arrow",
                        "label": "",
                        "confidence": "parsed_subject_ids",
                        "connector_id": "c1",
                    }
                ],
                "connectors": [
                    {
                        "id": "c1",
                        "type": "arrow",
                        "bbox": {
                            "x": 300,
                            "y": 145,
                            "width": 200,
                            "height": 10,
                            "unit": "hwp",
                        },
                        "points": [{"x": 300, "y": 150}, {"x": 500, "y": 150}],
                        "arrow": True,
                    }
                ],
                "mermaid": None,
            },
        }
    )

    assert 'class="diagram-positioned"' in html
    assert 'class="diagram-positioned-node diagram-shape-rectangle"' in html
    assert 'class="diagram-positioned-node diagram-shape-ellipse"' in html
    assert 'class="diagram-connectors"' in html
    assert '<ol class="diagram-nodes">' not in html


def test_render_structured_diagram_draws_positioned_connectors():
    from rag_document_parser.renderer.evidence_unit_render import render_evidence_html

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


def test_render_structured_diagram_keeps_positioned_layout_with_inferred_edges():
    from rag_document_parser.renderer.evidence_unit_render import render_evidence_html

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
                            "y": 100,
                            "width": 200,
                            "height": 100,
                            "unit": "hwp",
                        },
                    },
                    {
                        "id": "n2",
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
                        "arrow": True,
                        "points": [
                            {"x": 300, "y": 150},
                            {"x": 500, "y": 150},
                        ],
                    }
                ],
                "edges": [
                    {
                        "from": "n1",
                        "to": "n2",
                        "type": "arrow",
                        "label": "",
                        "confidence": "inferred_geometry",
                        "connector_id": "c1",
                    }
                ],
                "mermaid": None,
            },
        }
    )

    assert 'class="diagram-positioned"' in html
    assert 'class="diagram-connectors"' in html
    assert '<ul class="diagram-edges">' not in html


def test_render_structured_diagram_marks_arrow_connectors():
    from rag_document_parser.renderer.evidence_unit_render import render_evidence_html

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


def test_render_structured_diagram_marks_narrow_korean_nodes_vertical():
    from rag_document_parser.renderer.evidence_unit_render import render_evidence_html

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
                        "text": "의료기관",
                        "bbox": {
                            "x": 1,
                            "y": 7,
                            "width": 1,
                            "height": 2,
                            "unit": "hwpx_table_grid",
                        },
                    },
                    {
                        "id": "n2",
                        "shape_type": "label",
                        "text": "건강보험심사평가원",
                        "bbox": {
                            "x": 6,
                            "y": 7,
                            "width": 6,
                            "height": 2,
                            "unit": "hwpx_table_grid",
                        },
                    },
                ],
                "connectors": [],
                "edges": [],
                "mermaid": None,
            },
        }
    )

    assert "diagram-node-vertical" in html


def test_render_structured_diagram_places_step_labels_near_connectors():
    from rag_document_parser.renderer.evidence_unit_render import render_evidence_html

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
    from rag_document_parser.renderer.evidence_unit_render import render_evidence_html

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
    assert 'class="diagram-flowchart-step">⑤차기 지급진료비에서 공제</div>' in html
    assert 'class="diagram-flowchart-step">②미반환시 신고</div>' in html
    assert 'class="diagram-flowchart-step">⑥공제금 지급</div>' in html
    assert 'class="diagram-flowchart-step">③과다본인부담금<br>공제예정 통보</div>' in html
    assert "⑥공제금 지급 ③과다본인부담금" not in html
    assert "⑦처리결과<br>통보" in html
    assert 'class="diagram-flowchart-note">' in html


def test_render_hwp5_flowchart_shapes_prefers_original_like_flowchart():
    from rag_document_parser.renderer.evidence_unit_render import render_evidence_html

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
                        "text": "< 과다본인부담금 확인절차 >",
                    },
                    {"id": "n3", "shape_type": "label", "text": "①신청"},
                    {
                        "id": "n4",
                        "shape_type": "rectangle",
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
                        "id": "n5",
                        "shape_type": "rectangle",
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
                        "type": "arrow",
                        "points": [{"x": 300, "y": 150}, {"x": 500, "y": 150}],
                        "arrow": True,
                    }
                ],
                "edges": [
                    {
                        "from": "n4",
                        "to": "n5",
                        "type": "arrow",
                        "label": "①신청",
                        "confidence": "parsed_subject_ids",
                        "connector_id": "c1",
                    }
                ],
                "mermaid": None,
            },
        }
    )

    assert 'class="diagram-evidence diagram-flowchart"' in html
    assert 'class="diagram-flowchart-box">수급권자</div>' in html
    assert 'class="diagram-flowchart-arrow">→</span>' in html
    assert 'class="diagram-positioned"' not in html
