from __future__ import annotations


def _text_unit(id: str, text: str):
    from rag_document_parser import EvidenceUnit, SourceEvidence

    return EvidenceUnit(
        id=id,
        type="text",
        format="plain",
        source=SourceEvidence(kind="text", text=text),
        content=text,
        metadata={"common": {"chunk_kind": "text", "section_path": [], "display_format": "plain"}},
    )


def _table_unit(id: str):
    from rag_document_parser import EvidenceUnit, SourceEvidence

    table = {
        "caption": None,
        "columns": [
            {"id": "c1", "text": "항목"},
            {"id": "c2", "text": "내용"},
        ],
        "rows": [
            {
                "index": 1,
                "cells": [
                    {"column_id": "c1", "text": "A", "rowspan": 1, "colspan": 1, "children": []},
                    {"column_id": "c2", "text": "Alpha", "rowspan": 1, "colspan": 1, "children": []},
                ],
            },
            {
                "index": 2,
                "cells": [
                    {"column_id": "c1", "text": "B", "rowspan": 1, "colspan": 1, "children": []},
                    {"column_id": "c2", "text": "Beta", "rowspan": 1, "colspan": 1, "children": []},
                ],
            },
        ],
    }
    return EvidenceUnit(
        id=id,
        type="table",
        format="structured_table",
        source=SourceEvidence(
            kind="table",
            text="table: 2 columns\nrow 1: 항목=A; 내용=Alpha\nrow 2: 항목=B; 내용=Beta",
        ),
        content=table,
        metadata={
            "common": {"chunk_kind": "table", "section_path": [], "display_format": "structured_table"},
            "table": {"table_id": "t1", "headers": ["항목", "내용"], "row_count": 2},
        },
    )


def _single_row_table_unit(id: str):
    from rag_document_parser import EvidenceUnit, SourceEvidence

    base = _table_unit(id)
    table = dict(base.content)
    table["rows"] = [base.content["rows"][1]]
    metadata = dict(base.metadata)
    metadata["table"] = {**base.metadata["table"], "row_count": 1}
    return EvidenceUnit(
        id=id,
        type=base.type,
        format=base.format,
        source=SourceEvidence(
            kind="table",
            text="table: 2 columns\nrow 2: 항목=B; 내용=Beta",
        ),
        content=table,
        metadata=metadata,
    )


def test_agentic_chunker_uses_llm_prompt_when_no_plan_fn(monkeypatch):
    from rag_document_parser import LlmConfig
    from rag_document_parser.chunk import EvidenceUnitAgenticChunker

    calls = []

    def fake_chat_json(prompt, cfg):
        calls.append((prompt, cfg))
        return [
            {
                "unit_ids": ["b1"],
                "operations": [{"unit_id": "b1", "action": "include"}],
                "title": "첫 문장",
                "summary": "첫 문장 요약",
                "keywords": ["첫"],
                "questions": ["첫 문장은 무엇인가요?"],
            }
        ]

    monkeypatch.setattr("rag_document_parser.chunk.agentic.chat_json", fake_chat_json)
    cfg = LlmConfig(url="http://llm.test/v1", api_key="key", model="model")

    chunks = EvidenceUnitAgenticChunker(llm=cfg).chunk([_text_unit("b1", "첫 문장")])

    assert len(calls) == 1
    assert '"id": "b1"' in calls[0][0]
    assert calls[0][1] is cfg
    assert chunks[0].summary == "첫 문장 요약"


def test_agentic_chunker_records_context_units_without_duplicate_evidence():
    from rag_document_parser.chunk import EvidenceUnitAgenticChunker

    units = [_text_unit("b1", "앞 문맥"), _text_unit("b2", "대상 문장")]

    def plan_fn(window, cfg, max_units):
        return [
            {
                "unit_ids": ["b1"],
                "operations": [{"unit_id": "b1", "action": "include"}],
                "context_unit_ids": [],
                "summary": "앞 문맥",
                "keywords": ["앞"],
                "questions": ["앞 문맥은 무엇인가요?"],
            },
            {
                "unit_ids": ["b2"],
                "operations": [{"unit_id": "b2", "action": "include"}],
                "context_unit_ids": ["b1"],
                "summary": "대상 문장",
                "keywords": ["대상"],
                "questions": ["대상 문장은 무엇인가요?"],
            },
        ]

    chunks = EvidenceUnitAgenticChunker(llm=None, plan_fn=plan_fn).chunk(units)

    assert chunks[1].metadata["source_unit_ids"] == ["b2"]
    assert chunks[1].metadata["context_unit_ids"] == ["b1"]
    assert chunks[1].evidence.items[0].source_unit_ids == ["b2"]


def test_agentic_chunker_merges_adjacent_chunks_across_window_boundary():
    from rag_document_parser.chunk import EvidenceUnitAgenticChunker

    units = [
        _text_unit("b1", "첫 번째 설명"),
        _text_unit("b2", "두 번째 설명"),
        _text_unit("b3", "세 번째 설명"),
        _text_unit("b4", "네 번째 설명"),
    ]
    boundary_calls = []

    def plan_fn(window, cfg, max_units):
        unit_ids = [unit.id for unit in window]
        return [
            {
                "unit_ids": unit_ids,
                "operations": [
                    {"unit_id": unit_id, "action": "include"}
                    for unit_id in unit_ids
                ],
                "summary": f"{unit_ids[0]}부터 {unit_ids[-1]}까지",
                "keywords": [unit_ids[0]],
                "questions": [f"{unit_ids[0]}부터 {unit_ids[-1]}까지 무엇인가요?"],
            }
        ]

    def boundary_merge_fn(left, right, cfg, max_units):
        boundary_calls.append((left.metadata["source_unit_ids"], right.metadata["source_unit_ids"]))
        return {
            "action": "merge",
            "reason": "같은 의미 단위가 window 경계에서 이어진다.",
            "summary": "네 설명을 하나로 제공한다.",
            "keywords": ["설명"],
            "questions": ["네 설명은 무엇인가요?"],
        }

    chunks = EvidenceUnitAgenticChunker(
        llm=None,
        plan_fn=plan_fn,
        boundary_merge_fn=boundary_merge_fn,
        window_size=2,
        max_units_per_chunk=10,
    ).chunk(units)

    assert boundary_calls == [(["b1", "b2"], ["b3", "b4"])]
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.id == "chunk-1"
    assert [item.type for item in chunk.evidence.items] == ["text", "text", "text", "text"]
    assert chunk.metadata["common"]["unit_types"] == ["text"]
    assert chunk.summary == "네 설명을 하나로 제공한다."
    assert chunk.keywords == ["설명"]
    assert chunk.questions == ["네 설명은 무엇인가요?"]
    assert chunk.metadata["source_unit_ids"] == ["b1", "b2", "b3", "b4"]
    assert chunk.metadata["context_unit_ids"] == []
    assert chunk.metadata["_boundary_merges"] == [
        {
            "left_source_unit_ids": ["b1", "b2"],
            "right_source_unit_ids": ["b3", "b4"],
            "reason": "같은 의미 단위가 window 경계에서 이어진다.",
        }
    ]
    assert [item.source_unit_ids for item in chunk.evidence.items] == [["b1"], ["b2"], ["b3"], ["b4"]]
    assert "첫 번째 설명" in chunk.source.text
    assert "네 번째 설명" in chunk.source.text


def test_agentic_chunker_keeps_window_boundary_chunks_when_planner_says_keep():
    from rag_document_parser.chunk import EvidenceUnitAgenticChunker

    units = [
        _text_unit("b1", "첫 번째 설명"),
        _text_unit("b2", "두 번째 설명"),
        _text_unit("b3", "세 번째 설명"),
        _text_unit("b4", "네 번째 설명"),
    ]

    def plan_fn(window, cfg, max_units):
        unit_ids = [unit.id for unit in window]
        return [
            {
                "unit_ids": unit_ids,
                "operations": [
                    {"unit_id": unit_id, "action": "include"}
                    for unit_id in unit_ids
                ],
                "summary": f"{unit_ids[0]} window",
            }
        ]

    def boundary_merge_fn(left, right, cfg, max_units):
        return {"action": "keep", "reason": "서로 다른 의미 단위다."}

    chunks = EvidenceUnitAgenticChunker(
        llm=None,
        plan_fn=plan_fn,
        boundary_merge_fn=boundary_merge_fn,
        window_size=2,
    ).chunk(units)

    assert [chunk.metadata["source_unit_ids"] for chunk in chunks] == [["b1", "b2"], ["b3", "b4"]]
    assert "_boundary_merges" not in chunks[0].metadata
    assert "_boundary_merges" not in chunks[1].metadata


def test_agentic_chunker_keeps_boundary_chunks_when_boundary_planner_fails():
    from rag_document_parser.chunk import EvidenceUnitAgenticChunker

    units = [
        _text_unit("b1", "첫 번째 설명"),
        _text_unit("b2", "두 번째 설명"),
        _text_unit("b3", "세 번째 설명"),
        _text_unit("b4", "네 번째 설명"),
    ]

    def plan_fn(window, cfg, max_units):
        unit_ids = [unit.id for unit in window]
        return [
            {
                "unit_ids": unit_ids,
                "operations": [
                    {"unit_id": unit_id, "action": "include"}
                    for unit_id in unit_ids
                ],
                "summary": f"{unit_ids[0]} window",
            }
        ]

    def boundary_merge_fn(left, right, cfg, max_units):
        raise RuntimeError("boundary planner down")

    chunks = EvidenceUnitAgenticChunker(
        llm=None,
        plan_fn=plan_fn,
        boundary_merge_fn=boundary_merge_fn,
        window_size=2,
    ).chunk(units)

    assert [chunk.metadata["source_unit_ids"] for chunk in chunks] == [["b1", "b2"], ["b3", "b4"]]
    assert chunks[0].metadata["_warnings"] == [
        {
            "type": "agentic_boundary_merge_failed",
            "reason": "boundary planner down",
            "right_source_unit_ids": ["b3", "b4"],
        }
    ]
    assert "_boundary_merges" not in chunks[0].metadata


def test_agentic_chunker_uses_llm_boundary_prompt_between_windows(monkeypatch):
    from rag_document_parser import LlmConfig
    from rag_document_parser.chunk import EvidenceUnitAgenticChunker

    units = [
        _text_unit("b1", "첫 번째 설명"),
        _text_unit("b2", "두 번째 설명"),
        _text_unit("b3", "세 번째 설명"),
        _text_unit("b4", "네 번째 설명"),
    ]
    calls = []

    def plan_fn(window, cfg, max_units):
        unit_ids = [unit.id for unit in window]
        return [
            {
                "unit_ids": unit_ids,
                "operations": [
                    {"unit_id": unit_id, "action": "include"}
                    for unit_id in unit_ids
                ],
                "summary": f"{unit_ids[0]} window",
            }
        ]

    def fake_chat_json(prompt, cfg):
        calls.append((prompt, cfg))
        return {"action": "keep", "reason": "서로 다른 주제다."}

    monkeypatch.setattr("rag_document_parser.chunk.agentic.chat_json", fake_chat_json)
    cfg = LlmConfig(url="http://llm.test/v1", api_key="key", model="model")

    chunks = EvidenceUnitAgenticChunker(
        llm=cfg,
        plan_fn=plan_fn,
        window_size=2,
    ).chunk(units)

    assert [chunk.metadata["source_unit_ids"] for chunk in chunks] == [["b1", "b2"], ["b3", "b4"]]
    assert len(calls) == 1
    assert calls[0][1] is cfg
    assert "window boundary merge planner" in calls[0][0]
    assert '"left_chunk"' in calls[0][0]
    assert '"right_chunk"' in calls[0][0]
    assert '"source_unit_ids": ["b1", "b2"]' in calls[0][0]
    assert '"source_unit_ids": ["b3", "b4"]' in calls[0][0]


def test_agentic_chunker_uses_rich_korean_llm_prompt_contract(monkeypatch):
    from rag_document_parser import LlmConfig
    from rag_document_parser.chunk import EvidenceUnitAgenticChunker

    calls = []

    def fake_chat_json(prompt, cfg):
        calls.append((prompt, cfg))
        return [
            {
                "unit_ids": ["b2"],
                "operations": [{"unit_id": "b2", "action": "include"}],
                "title": "표",
                "summary": "표 전체를 제공한다.",
                "keywords": ["표"],
                "questions": ["표에는 무엇이 있나요?"],
            }
        ]

    monkeypatch.setattr("rag_document_parser.chunk.agentic.chat_json", fake_chat_json)
    cfg = LlmConfig(url="http://llm.test/v1", api_key="key", model="model")

    chunks = EvidenceUnitAgenticChunker(llm=cfg, max_units_per_chunk=7).chunk([_table_unit("b2")])

    assert len(chunks) == 1
    assert len(calls) == 1
    prompt = calls[0][0]
    assert "RAG 인덱싱용 EvidenceUnit chunk planner" in prompt
    assert '"max_units_per_chunk": 7' in prompt
    assert '"source_preview":' in prompt
    assert '"source_text"' not in prompt
    assert '"row_count": 2' in prompt
    assert '"항목"' in prompt
    assert '"내용"' in prompt
    assert "{{" not in prompt
    assert "}}" not in prompt
    assert '    "unit_ids": ["b2"]' in prompt
    assert '      {"unit_id": "b2", "action": "include"}' in prompt
    assert '    "context_unit_ids": []' in prompt
    assert '"row_ranges": [[1, 1]]' in prompt
    assert "양 끝을 포함" in prompt
    assert "inclusive [start, end]" in prompt
    assert "모든 실제 row index" in prompt
    assert 'action "include"로 전체 table을 포함' in prompt
    assert "evidence content는 작성하지 않습니다" in prompt
    assert "evidence content는 unit에서 복사됩니다" in prompt
    assert chunks[0].summary == "표 전체를 제공한다."


def test_agentic_chunker_prompt_uses_table_id_for_include_rows_example(monkeypatch):
    from rag_document_parser import LlmConfig
    from rag_document_parser.chunk import EvidenceUnitAgenticChunker

    calls = []

    def fake_chat_json(prompt, cfg):
        calls.append((prompt, cfg))
        return [
            {
                "unit_ids": ["txt1", "tbl1"],
                "operations": [
                    {"unit_id": "txt1", "action": "include"},
                    {"unit_id": "tbl1", "action": "include"},
                ],
                "summary": "텍스트와 표 요약",
                "keywords": ["텍스트", "표"],
                "questions": ["텍스트와 표에는 무엇이 있나요?"],
            }
        ]

    monkeypatch.setattr("rag_document_parser.chunk.agentic.chat_json", fake_chat_json)
    cfg = LlmConfig(url="http://llm.test/v1", api_key="key", model="model")

    EvidenceUnitAgenticChunker(llm=cfg).chunk([_text_unit("txt1", "텍스트"), _table_unit("tbl1")])

    prompt = calls[0][0]
    assert '      {"unit_id": "txt1", "action": "include"}' in prompt
    assert '{"unit_id": "tbl1", "action": "include_rows", "row_ranges": [[1, 1]]}' in prompt
    assert '"unit_id": "txt1", "action": "include_rows"' not in prompt


def test_agentic_chunker_prompt_omits_include_rows_example_without_table(monkeypatch):
    from rag_document_parser import LlmConfig
    from rag_document_parser.chunk import EvidenceUnitAgenticChunker

    calls = []

    def fake_chat_json(prompt, cfg):
        calls.append((prompt, cfg))
        return [
            {
                "unit_ids": ["txt1"],
                "operations": [{"unit_id": "txt1", "action": "include"}],
                "summary": "텍스트 요약",
                "keywords": ["텍스트"],
                "questions": ["텍스트는 무엇인가요?"],
            }
        ]

    monkeypatch.setattr("rag_document_parser.chunk.agentic.chat_json", fake_chat_json)
    cfg = LlmConfig(url="http://llm.test/v1", api_key="key", model="model")

    EvidenceUnitAgenticChunker(llm=cfg).chunk([_text_unit("txt1", "텍스트")])

    prompt = calls[0][0]
    assert "include_rows operation 예시" not in prompt
    assert '"action": "include_rows"' not in prompt


def test_agentic_chunker_prompt_example_uses_window_unit_id(monkeypatch):
    from rag_document_parser import LlmConfig
    from rag_document_parser.chunk import EvidenceUnitAgenticChunker

    calls = []

    def fake_chat_json(prompt, cfg):
        calls.append((prompt, cfg))
        return [
            {
                "unit_ids": ["u99"],
                "operations": [{"unit_id": "u99", "action": "include"}],
                "summary": "다른 문장 요약",
                "keywords": ["다른"],
                "questions": ["다른 문장은 무엇인가요?"],
            }
        ]

    monkeypatch.setattr("rag_document_parser.chunk.agentic.chat_json", fake_chat_json)
    cfg = LlmConfig(url="http://llm.test/v1", api_key="key", model="model")

    chunks = EvidenceUnitAgenticChunker(llm=cfg).chunk([_text_unit("u99", "다른 문장")])

    prompt = calls[0][0]
    assert '"unit_ids": ["u99"]' in prompt
    assert '"unit_id": "u99"' in prompt
    assert '"unit_ids": ["b1"]' not in prompt
    assert chunks[0].metadata["source_unit_ids"] == ["u99"]


def test_agentic_chunker_prompt_compacts_asset_metadata(monkeypatch):
    from rag_document_parser import EvidenceUnit, LlmConfig, SourceEvidence
    from rag_document_parser.chunk import EvidenceUnitAgenticChunker

    calls = []
    long_alt = "설명" * 180

    unit = EvidenceUnit(
        id="img1",
        type="image",
        format="png",
        source=SourceEvidence(kind="image", text="이미지 설명"),
        content={"asset_id": "asset-1"},
        metadata={
            "common": {"chunk_kind": "image", "section_path": [], "display_format": "png"},
            "asset": {
                "asset_id": "asset-1",
                "kind": "image",
                "mime": "image/png",
                "alt": long_alt,
                "nested": {"ignored": True},
                "extra": "ignored",
            },
        },
    )

    def fake_chat_json(prompt, cfg):
        calls.append((prompt, cfg))
        return [
            {
                "unit_ids": ["img1"],
                "operations": [{"unit_id": "img1", "action": "include"}],
                "summary": "이미지 요약",
                "keywords": ["이미지"],
                "questions": ["이미지는 무엇인가요?"],
            }
        ]

    monkeypatch.setattr("rag_document_parser.chunk.agentic.chat_json", fake_chat_json)
    cfg = LlmConfig(url="http://llm.test/v1", api_key="key", model="model")

    EvidenceUnitAgenticChunker(llm=cfg).chunk([unit])

    prompt = calls[0][0]
    assert '"asset_id": "asset-1"' in prompt
    assert '"kind": "image"' in prompt
    assert '"mime": "image/png"' in prompt
    assert long_alt not in prompt
    assert f'"alt": "{long_alt[:299]}…"' in prompt
    assert '"nested"' not in prompt
    assert '"extra"' not in prompt


def test_agentic_chunker_materializes_cross_kind_chunk_from_plan():
    from rag_document_parser.chunk import EvidenceUnitAgenticChunker

    units = [_text_unit("b1", "기준 설명"), _table_unit("b2")]

    def plan_fn(window, cfg, max_units):
        return [
            {
                "unit_ids": ["b1", "b2"],
                "operations": [
                    {"unit_id": "b1", "action": "include"},
                    {"unit_id": "b2", "action": "include"},
                ],
                "context_unit_ids": [],
                "title": "기준 설명과 표",
                "summary": "기준 설명과 표를 함께 제공한다.",
                "keywords": ["기준", "표"],
                "questions": ["기준 설명과 표에는 무엇이 있나요?"],
            }
        ]

    chunks = EvidenceUnitAgenticChunker(llm=None, plan_fn=plan_fn).chunk(units)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.summary == "기준 설명과 표를 함께 제공한다."
    assert chunk.keywords == ["기준", "표"]
    assert chunk.questions == ["기준 설명과 표에는 무엇이 있나요?"]
    assert chunk.metadata["source_unit_ids"] == ["b1", "b2"]
    assert chunk.metadata["context_unit_ids"] == []
    assert chunk.metadata["common"]["unit_types"] == ["text", "table"]
    assert [item.type for item in chunk.evidence.items] == ["text", "table"]
    assert chunk.evidence.items[0].content == "기준 설명"
    assert chunk.evidence.items[1].content["rows"][1]["index"] == 2


def test_agentic_chunker_materializes_table_row_subset():
    from rag_document_parser.chunk import EvidenceUnitAgenticChunker

    def plan_fn(window, cfg, max_units):
        return [
            {
                "unit_ids": ["b2"],
                "operations": [
                    {"unit_id": "b2", "action": "include_rows", "row_ranges": [[2, 2]]}
                ],
                "title": "B 항목",
                "summary": "B 항목만 제공한다.",
                "keywords": ["B", "Beta"],
                "questions": ["B 항목의 내용은 무엇인가요?"],
            }
        ]

    chunks = EvidenceUnitAgenticChunker(llm=None, plan_fn=plan_fn).chunk([_single_row_table_unit("b2")])

    table_item = chunks[0].evidence.items[0]
    assert table_item.type == "table"
    assert table_item.format == "structured_table"
    assert [row["index"] for row in table_item.content["rows"]] == [2]
    assert "row 2" in chunks[0].source.text
    assert "row 1" not in chunks[0].source.text


def test_agentic_chunker_falls_back_when_table_row_split_omits_rows():
    from rag_document_parser.chunk import EvidenceUnitAgenticChunker

    def plan_fn(window, cfg, max_units):
        return [
            {
                "unit_ids": ["b2"],
                "operations": [
                    {"unit_id": "b2", "action": "include_rows", "row_ranges": [[2, 2]]}
                ],
                "summary": "B 항목만 제공한다.",
            }
        ]

    chunks = EvidenceUnitAgenticChunker(llm=None, plan_fn=plan_fn).chunk([_table_unit("b2")])

    assert len(chunks) == 1
    assert chunks[0].metadata["source_unit_ids"] == ["b2"]
    assert [row["index"] for row in chunks[0].evidence.items[0].content["rows"]] == [1, 2]
    assert "omitted table rows" in chunks[0].metadata["_fallback_reason"]


def test_agentic_chunker_falls_back_without_dropping_units_on_invalid_plan():
    from rag_document_parser.chunk import EvidenceUnitAgenticChunker

    units = [_text_unit("b1", "첫 번째 설명"), _text_unit("b2", "두 번째 설명")]
    rejected_plan = [
        {
            "unit_ids": ["b1"],
            "operations": [{"unit_id": "b1", "action": "include"}],
            "summary": "첫 번째만 포함한다.",
        }
    ]

    def plan_fn(window, cfg, max_units):
        return rejected_plan

    chunks = EvidenceUnitAgenticChunker(llm=None, plan_fn=plan_fn).chunk(units)

    assert [chunk.metadata["source_unit_ids"] for chunk in chunks] == [["b1"], ["b2"]]
    assert chunks[0].metadata["_fallback_reason"].startswith("chunk plan omitted units")
    assert chunks[1].metadata["_fallback_reason"].startswith("chunk plan omitted units")
    assert chunks[0].metadata["_rejected_plan"] == rejected_plan
    assert chunks[1].metadata["_rejected_plan"] == rejected_plan


def test_agentic_chunker_does_not_record_rejected_plan_when_planner_raises_exception():
    from rag_document_parser.chunk import EvidenceUnitAgenticChunker

    def plan_fn(window, cfg, max_units):
        raise RuntimeError("planner down")

    chunks = EvidenceUnitAgenticChunker(llm=None, plan_fn=plan_fn).chunk([_text_unit("b1", "설명")])

    assert "_rejected_plan" not in chunks[0].metadata


def test_agentic_chunker_falls_back_when_planner_raises_exception():
    from rag_document_parser.chunk import EvidenceUnitAgenticChunker

    units = [_text_unit("b1", "첫 번째 설명"), _text_unit("b2", "두 번째 설명")]

    def plan_fn(window, cfg, max_units):
        raise RuntimeError("planner down")

    chunks = EvidenceUnitAgenticChunker(llm=None, plan_fn=plan_fn).chunk(units)

    assert [chunk.metadata["source_unit_ids"] for chunk in chunks] == [["b1"], ["b2"]]
    assert "planner down" in chunks[0].metadata["_fallback_reason"]
    assert "planner down" in chunks[1].metadata["_fallback_reason"]


def test_agentic_chunker_rejects_unit_ids_that_do_not_match_operations():
    from rag_document_parser.chunk import EvidenceUnitAgenticChunker

    units = [_text_unit("b1", "첫 번째 설명"), _text_unit("b2", "두 번째 설명")]

    def plan_fn(window, cfg, max_units):
        return [
            {
                "unit_ids": ["b1"],
                "operations": [
                    {"unit_id": "b1", "action": "include"},
                    {"unit_id": "b2", "action": "include"},
                ],
                "summary": "선언과 작업이 다르다.",
            }
        ]

    chunks = EvidenceUnitAgenticChunker(llm=None, plan_fn=plan_fn).chunk(units)

    assert [chunk.metadata["source_unit_ids"] for chunk in chunks] == [["b1"], ["b2"]]
    assert "unit_ids must match operation unit_ids" in chunks[0].metadata["_fallback_reason"]


def test_agentic_chunker_rejects_present_unit_ids_that_are_not_a_list():
    from rag_document_parser.chunk import EvidenceUnitAgenticChunker

    def plan_fn(window, cfg, max_units):
        return [
            {
                "unit_ids": None,
                "operations": [{"unit_id": "b1", "action": "include"}],
                "summary": "unit_ids가 목록이 아니다.",
            }
        ]

    chunks = EvidenceUnitAgenticChunker(llm=None, plan_fn=plan_fn).chunk([_text_unit("b1", "설명")])

    assert chunks[0].metadata["source_unit_ids"] == ["b1"]
    assert "unit_ids must be a list" in chunks[0].metadata["_fallback_reason"]


def test_agentic_chunker_splits_table_rows_across_chunks():
    from rag_document_parser.chunk import EvidenceUnitAgenticChunker

    def plan_fn(window, cfg, max_units):
        return [
            {
                "unit_ids": ["b2"],
                "operations": [{"unit_id": "b2", "action": "include_rows", "row_ranges": [[1, 1]]}],
                "summary": "A 항목만 제공한다.",
            },
            {
                "unit_ids": ["b2"],
                "operations": [{"unit_id": "b2", "action": "include_rows", "row_ranges": [[2, 2]]}],
                "context_unit_ids": ["b2"],
                "summary": "B 항목만 제공한다.",
            },
        ]

    chunks = EvidenceUnitAgenticChunker(llm=None, plan_fn=plan_fn).chunk([_table_unit("b2")])

    assert len(chunks) == 2
    assert chunks[0].metadata["source_unit_ids"] == ["b2"]
    assert chunks[1].metadata["source_unit_ids"] == ["b2"]
    assert [row["index"] for row in chunks[0].evidence.items[0].content["rows"]] == [1]
    assert [row["index"] for row in chunks[1].evidence.items[0].content["rows"]] == [2]
    assert "_fallback_reason" not in chunks[0].metadata
    assert "_fallback_reason" not in chunks[1].metadata


def test_agentic_chunker_records_warning_when_plan_exceeds_max_unit_hint():
    from rag_document_parser.chunk import EvidenceUnitAgenticChunker

    units = [
        _text_unit("b1", "첫 번째 설명"),
        _text_unit("b2", "두 번째 설명"),
        _text_unit("b3", "세 번째 설명"),
    ]

    def plan_fn(window, cfg, max_units):
        return [
            {
                "unit_ids": ["b1", "b2", "b3"],
                "operations": [
                    {"unit_id": "b1", "action": "include"},
                    {"unit_id": "b2", "action": "include"},
                    {"unit_id": "b3", "action": "include"},
                ],
                "summary": "세 설명을 함께 제공한다.",
            }
        ]

    chunks = EvidenceUnitAgenticChunker(
        llm=None,
        plan_fn=plan_fn,
        max_units_per_chunk=2,
    ).chunk(units)

    assert len(chunks) == 1
    assert chunks[0].metadata["source_unit_ids"] == ["b1", "b2", "b3"]
    assert chunks[0].metadata["_warnings"] == [
        {
            "type": "agentic_chunk_exceeds_max_units",
            "source_unit_count": 3,
            "max_units_per_chunk": 2,
        }
    ]


def test_agentic_chunker_uses_row_subset_text_for_planned_fallback_fields():
    from rag_document_parser.chunk import EvidenceUnitAgenticChunker

    def plan_fn(window, cfg, max_units):
        return [
            {
                "unit_ids": ["b2"],
                "operations": [{"unit_id": "b2", "action": "include_rows", "row_ranges": [[1, 1]]}],
            },
            {
                "unit_ids": ["b2"],
                "operations": [{"unit_id": "b2", "action": "include_rows", "row_ranges": [[2, 2]]}],
                "context_unit_ids": ["b2"],
            },
        ]

    chunks = EvidenceUnitAgenticChunker(llm=None, plan_fn=plan_fn).chunk([_table_unit("b2")])

    assert len(chunks) == 2
    assert "row 2" not in chunks[0].source.text
    assert "row 2" not in chunks[0].summary
    assert "Beta" not in chunks[0].summary
    assert "Beta" not in chunks[0].keywords
    assert "Beta" not in " ".join(chunks[0].questions)
    assert "row 1" not in chunks[1].source.text
    assert "row 1" not in chunks[1].summary
    assert "Alpha" not in chunks[1].summary
    assert "Alpha" not in chunks[1].keywords
    assert "Alpha" not in " ".join(chunks[1].questions)


def test_agentic_chunker_rejects_overlapping_table_row_ranges():
    from rag_document_parser.chunk import EvidenceUnitAgenticChunker

    def plan_fn(window, cfg, max_units):
        return [
            {
                "unit_ids": ["b2"],
                "operations": [{"unit_id": "b2", "action": "include_rows", "row_ranges": [[1, 2]]}],
                "summary": "전체 행을 부분 선택한다.",
            },
            {
                "unit_ids": ["b2"],
                "operations": [{"unit_id": "b2", "action": "include_rows", "row_ranges": [[2, 2]]}],
                "summary": "겹치는 행을 선택한다.",
            },
        ]

    chunks = EvidenceUnitAgenticChunker(llm=None, plan_fn=plan_fn).chunk([_table_unit("b2")])

    assert len(chunks) == 1
    assert chunks[0].metadata["source_unit_ids"] == ["b2"]
    assert "overlap" in chunks[0].metadata["_fallback_reason"]


def test_agentic_chunker_rejects_malformed_row_ranges():
    from rag_document_parser.chunk import EvidenceUnitAgenticChunker

    def plan_fn(window, cfg, max_units):
        return [
            {
                "unit_ids": ["b2"],
                "operations": [
                    {"unit_id": "b2", "action": "include_rows", "row_ranges": [[2, 2], ["bad", 3]]}
                ],
                "summary": "잘못된 행 범위를 포함한다.",
            }
        ]

    chunks = EvidenceUnitAgenticChunker(llm=None, plan_fn=plan_fn).chunk([_table_unit("b2")])

    assert len(chunks) == 1
    assert chunks[0].metadata["source_unit_ids"] == ["b2"]
    assert "row range must be [start, end] ints with start <= end" in chunks[0].metadata["_fallback_reason"]


def test_agentic_chunker_rejects_row_ranges_outside_table_bounds():
    from rag_document_parser.chunk import EvidenceUnitAgenticChunker

    def plan_fn(window, cfg, max_units):
        return [
            {
                "unit_ids": ["b2"],
                "operations": [{"unit_id": "b2", "action": "include_rows", "row_ranges": [[0, 99]]}],
                "summary": "표 범위를 벗어난 행을 선택한다.",
            }
        ]

    chunks = EvidenceUnitAgenticChunker(llm=None, plan_fn=plan_fn).chunk([_table_unit("b2")])

    assert len(chunks) == 1
    assert chunks[0].metadata["source_unit_ids"] == ["b2"]
    assert "row range is outside table rows" in chunks[0].metadata["_fallback_reason"]


def test_agentic_chunker_rejects_full_include_and_row_subset_conflict():
    from rag_document_parser.chunk import EvidenceUnitAgenticChunker

    def plan_fn(window, cfg, max_units):
        return [
            {
                "unit_ids": ["b2"],
                "operations": [{"unit_id": "b2", "action": "include"}],
                "summary": "표 전체를 포함한다.",
            },
            {
                "unit_ids": ["b2"],
                "operations": [{"unit_id": "b2", "action": "include_rows", "row_ranges": [[1, 1]]}],
                "summary": "표 일부도 포함한다.",
            },
        ]

    chunks = EvidenceUnitAgenticChunker(llm=None, plan_fn=plan_fn).chunk([_table_unit("b2")])

    assert len(chunks) == 1
    assert "full include conflicts with include_rows" in chunks[0].metadata["_fallback_reason"]


def test_agentic_chunker_falls_back_on_duplicate_unit_ids_before_planning():
    from rag_document_parser.chunk import EvidenceUnitAgenticChunker

    units = [_text_unit("dup", "첫 번째 설명"), _text_unit("dup", "두 번째 설명")]
    plan_called = False

    def plan_fn(window, cfg, max_units):
        nonlocal plan_called
        plan_called = True
        return []

    chunks = EvidenceUnitAgenticChunker(llm=None, plan_fn=plan_fn).chunk(units)

    assert plan_called is False
    assert [chunk.metadata["source_unit_ids"] for chunk in chunks] == [["dup"], ["dup"]]
    assert chunks[0].source.text == "첫 번째 설명"
    assert chunks[1].source.text == "두 번째 설명"
    assert "duplicate unit id" in chunks[0].metadata["_fallback_reason"]
    assert "duplicate unit id" in chunks[1].metadata["_fallback_reason"]


def test_agentic_chunker_preserves_source_unit_metadata_on_planned_and_fallback_chunks():
    from rag_document_parser.chunk import EvidenceUnitAgenticChunker

    unit = _text_unit("b1", "메타데이터 설명")

    def valid_plan_fn(window, cfg, max_units):
        return [
            {
                "unit_ids": ["b1"],
                "operations": [{"unit_id": "b1", "action": "include"}],
                "summary": "메타데이터를 보존한다.",
            }
        ]

    planned_chunks = EvidenceUnitAgenticChunker(llm=None, plan_fn=valid_plan_fn).chunk([unit])

    assert planned_chunks[0].metadata["source_units"] == [
        {"id": "b1", "type": "text", "format": "plain", "metadata": unit.metadata}
    ]

    def invalid_plan_fn(window, cfg, max_units):
        return []

    fallback_chunks = EvidenceUnitAgenticChunker(llm=None, plan_fn=invalid_plan_fn).chunk([unit])

    assert fallback_chunks[0].metadata["source_units"] == [
        {"id": "b1", "type": "text", "format": "plain", "metadata": unit.metadata}
    ]


def test_agentic_chunker_rejects_context_unit_from_current_chunk():
    from rag_document_parser.chunk import EvidenceUnitAgenticChunker

    def plan_fn(window, cfg, max_units):
        return [
            {
                "unit_ids": ["b1"],
                "operations": [{"unit_id": "b1", "action": "include"}],
                "context_unit_ids": ["b1"],
                "summary": "현재 청크를 컨텍스트로 잘못 참조한다.",
            }
        ]

    chunks = EvidenceUnitAgenticChunker(llm=None, plan_fn=plan_fn).chunk([_text_unit("b1", "현재 설명")])

    assert len(chunks) == 1
    assert "context unit id must refer to a prior assigned unit" in chunks[0].metadata["_fallback_reason"]
