from __future__ import annotations

from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

LlmResponse = TypeVar("LlmResponse", bound=BaseModel)
ChatMessage = dict[str, Any]


class LlmGateway(Protocol):
    """Application port for JSON and schema-constrained LLM responses."""

    def complete_json(self, prompt: str) -> Any:
        ...

    def complete_model(
        self,
        messages: list[ChatMessage],
        response_model: type[LlmResponse],
        *,
        schema_name: str,
    ) -> LlmResponse:
        ...
