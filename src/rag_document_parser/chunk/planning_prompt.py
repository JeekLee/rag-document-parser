from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ..models import EvidenceUnit, RagChunk

_PLAN_PROMPT = """\
당신은 RAG 인덱싱용 EvidenceUnit chunk planner입니다.
아래 unit 목록을 의미적으로 일관된 chunk plan으로 묶어 주세요.

규칙:
- evidence content는 작성하지 않습니다. unit_id와 operation만 작성합니다.
- evidence content는 unit에서 복사됩니다.
- 모든 unit은 누락 없이 evidence로 포함되어야 합니다.
- structured_table은 include_rows로 여러 chunk에 분해할 수 있습니다.
- include_rows row range는 겹치지 않아야 하며 table row 범위 안에 있어야 합니다.
- include_rows의 row_ranges는 양 끝을 포함하는 inclusive [start, end] 쌍 목록입니다.
- include_rows row range는 table payload의 rows[].index에 표시된 값만 사용합니다.
- 0-based row index를 만들지 않습니다. 첫 row index가 1이면 [1, ...]부터 시작해야 합니다.
- row index는 unit별로 독립적입니다. 이전 table unit의 row 번호를 이어서 계산하지 않습니다.
- 각 include_rows operation은 해당 unit_id의 rows[].index 범위 안에서만 작성합니다.
- include_rows를 사용하면 해당 table의 모든 실제 row index를 빠짐없이, 겹치지 않게 포함해야 합니다.
- table row coverage에 확신이 없으면 action "include"로 전체 table을 포함합니다.
- context_unit_ids는 이미 이전 chunk에서 evidence로 포함된 unit id만 작성하며, 없으면 빈 배열입니다.
- text, table, image를 같은 chunk에 묶을 수 있습니다.
- summary, keywords, questions는 최종 RagChunk 확정 후 별도 enrichment 단계에서 생성합니다.
- max_units_per_chunk는 일반 chunk의 hard limit입니다. 하나의 원자적 unit 자체가 큰 경우를 제외하고 초과하지 않습니다.
- 하나의 plan item에서 서로 다른 unit_id를 max_units_per_chunk보다 많이 포함하지 않습니다.
- unit_ids 길이와 operations의 고유 unit_id 개수는 max_units_per_chunk 이하이어야 합니다.
- 의미상 한 주제가 max_units_per_chunk를 넘으면 같은 title을 유지한 채 인접한 여러 plan item으로 순서대로 분할합니다.
- 같은 장/절/표 아래라도 검색 질문이 달라지는 조문, 번호 항목, 서식 작성 항목은 분리합니다.
- 긴 연속 목록은 의미가 이어져도 max_units_per_chunk 이하의 인접 chunk로 나눕니다.

Unit 목록:
{units}

JSON object만 출력하세요:
{
  "chunks": [
    {
      "unit_ids": [{example_unit_id}],
      "operations": [
        {"unit_id": {example_unit_id}, "action": "include", "row_ranges": null}
      ],
      "context_unit_ids": [],
      "title": "제목"
    }
  ]
}
{include_rows_example}
"""

_BOUNDARY_PROMPT = """\
You are a RAG window boundary merge planner.
Decide whether two adjacent chunks from neighboring EvidenceUnit windows are one continuous semantic unit.

Rules:
- Return only a JSON object.
- Use action "merge" only when the right chunk directly continues the same topic, table, section, or Q&A block.
- Use action "keep" when the chunks are merely related but answer different retrieval questions.
- Do not invent evidence content. Evidence content will be copied from the existing chunks.
- Summary, keywords, and questions are generated after final chunks are fixed.
- Keep merged chunks reasonably close to {max_units} source units when possible.

Boundary payload:
{boundary}

JSON object only:
{{
  "action": "merge",
  "reason": "short reason",
  "title": "optional merged title"
}}
"""


def build_plan_prompt(window: list[EvidenceUnit], max_units: int) -> str:
    payload = {
        "max_units_per_chunk": max_units,
        "units": [_unit_payload(index, unit) for index, unit in enumerate(window)],
    }
    example_unit_id = json.dumps(window[0].id if window else "unit", ensure_ascii=False)
    table_unit = next((unit for unit in window if unit.format == "structured_table"), None)
    include_rows_example = ""
    row_range = _example_row_range(table_unit) if table_unit is not None else None
    if table_unit is not None and row_range is not None:
        table_unit_id = json.dumps(table_unit.id, ensure_ascii=False)
        include_rows_example = (
            "\ninclude_rows operation 예시:\n"
            f'{{"unit_id": {table_unit_id}, "action": "include_rows", "row_ranges": [{row_range}]}}\n'
        )
    return _PLAN_PROMPT.replace("{max_units}", str(max_units)).replace(
        "{example_unit_id}", example_unit_id
    ).replace(
        "{include_rows_example}", include_rows_example
    ).replace(
        "{units}", json.dumps(payload, ensure_ascii=False, indent=2)
    )


def build_boundary_prompt(left: RagChunk, right: RagChunk, max_units: int) -> str:
    payload = {
        "max_units_per_chunk": max_units,
        "left_chunk": _boundary_chunk_payload(left),
        "right_chunk": _boundary_chunk_payload(right),
    }
    return _BOUNDARY_PROMPT.replace("{max_units}", str(max_units)).replace(
        "{boundary}",
        json.dumps(payload, ensure_ascii=False),
    )


def _boundary_chunk_payload(chunk: RagChunk) -> dict[str, Any]:
    return {
        "id": chunk.id,
        "source_unit_ids": _strings(chunk.metadata.get("source_unit_ids")),
        "context_unit_ids": _strings(chunk.metadata.get("context_unit_ids")),
        "unit_types": _strings(
            chunk.metadata.get("common", {}).get("unit_types")
            if isinstance(chunk.metadata.get("common"), Mapping)
            else None
        ),
        "summary": chunk.summary,
        "keywords": list(chunk.keywords),
        "questions": list(chunk.questions),
        "source_preview": _truncate(chunk.source.text, 900),
        "evidence_items": [
            {
                "type": item.type,
                "format": item.format,
                "source_unit_ids": list(item.source_unit_ids),
            }
            for item in chunk.evidence.items
        ],
    }


def _example_row_range(unit: EvidenceUnit) -> list[int] | None:
    if not isinstance(unit.content, Mapping):
        return None
    rows = unit.content.get("rows")
    if not isinstance(rows, list):
        return None
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        index = row.get("index")
        if type(index) is int:
            return [index, index]
    return None


def _unit_payload(index: int, unit: EvidenceUnit) -> dict[str, Any]:
    common = unit.metadata.get("common", {})
    table = unit.metadata.get("table", {})
    asset = unit.metadata.get("asset", {})
    return {
        "id": unit.id,
        "index": index,
        "type": unit.type,
        "format": unit.format,
        "section_path": common.get("section_path", []) if isinstance(common, Mapping) else [],
        "source_preview": _truncate(unit.source.text, 900),
        "table": _compact_table(table),
        "asset": _compact_asset(asset),
    }


def _compact_table(table: Any) -> dict[str, Any]:
    if not isinstance(table, Mapping):
        return {}
    result: dict[str, Any] = {}
    if "table_id" in table:
        result["table_id"] = table["table_id"]
    if "headers" in table and isinstance(table["headers"], list):
        result["headers"] = [str(header)[:80] for header in table["headers"][:12]]
    if "row_count" in table:
        result["row_count"] = table["row_count"]
    return result


def _compact_asset(asset: Any) -> dict[str, Any]:
    if not isinstance(asset, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key in ("asset_id", "kind", "mime", "ext", "uri", "public_url", "alt", "caption"):
        if key not in asset:
            continue
        value = asset[key]
        if value is None:
            result[key] = None
        elif isinstance(value, str):
            result[key] = _truncate(value, 300)
        elif type(value) in (int, float, bool):
            result[key] = value
    return result


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


def _truncate(value: str, limit: int) -> str:
    text = value.strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"
