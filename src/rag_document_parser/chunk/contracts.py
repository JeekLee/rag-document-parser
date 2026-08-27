from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class LlmContract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ChunkPlanOperation(LlmContract):
    unit_id: str = Field(min_length=1)
    action: Literal["include", "include_rows"]
    row_ranges: list[list[int]] | None


class ChunkPlanItem(LlmContract):
    unit_ids: list[str]
    operations: list[ChunkPlanOperation] = Field(min_length=1)
    context_unit_ids: list[str]
    title: str


class ChunkPlanResponse(LlmContract):
    chunks: list[ChunkPlanItem] = Field(min_length=1)


class BoundaryDecision(LlmContract):
    action: Literal["merge", "keep"]
    reason: str
    title: str


class ChunkEnrichmentResponse(LlmContract):
    summary: str
    keywords: list[str]
    questions: list[str]


class BatchChunkEnrichmentItem(ChunkEnrichmentResponse):
    id: str = Field(min_length=1)


class BatchChunkEnrichmentResponse(LlmContract):
    chunks: list[BatchChunkEnrichmentItem]
