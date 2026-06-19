from __future__ import annotations

import json
from typing import Any, Literal
from urllib import request

from pydantic import BaseModel, ConfigDict

ChatMessage = dict[str, Any]
GeminiThinkingMode = Literal["default", "disabled", "minimal", "low", "medium", "high"]
GemmaThinkingMode = Literal["default"]
QwenThinkingMode = Literal["default", "enabled", "disabled"]
QwenThinkingParameter = Literal["chat_template_kwargs", "enable_thinking"]


class LlmConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str
    api_key: str
    model: str
    temperature: float = 0.0
    timeout: float = 120.0

    def prepare_messages(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        return messages

    def request_body_options(self) -> dict[str, Any]:
        return {}


class GeminiLlmConfig(LlmConfig):
    provider: Literal["gemini"] = "gemini"
    thinking: GeminiThinkingMode = "default"

    def request_body_options(self) -> dict[str, Any]:
        if self.thinking == "default":
            return {}
        if self.thinking == "disabled":
            return {"reasoning_effort": "none"}
        return {"reasoning_effort": self.thinking}


class QwenLlmConfig(LlmConfig):
    provider: Literal["qwen"] = "qwen"
    thinking: QwenThinkingMode = "default"
    thinking_parameter: QwenThinkingParameter = "chat_template_kwargs"

    def request_body_options(self) -> dict[str, Any]:
        if self.thinking == "default":
            return {}
        enabled = self.thinking == "enabled"
        if self.thinking_parameter == "enable_thinking":
            return {"enable_thinking": enabled}
        return {"chat_template_kwargs": {"enable_thinking": enabled}}


class GemmaLlmConfig(LlmConfig):
    provider: Literal["gemma"] = "gemma"
    thinking: GemmaThinkingMode = "default"


def chat_json(prompt: str, cfg: LlmConfig) -> Any:
    endpoint = chat_completions_url(cfg.url)
    messages = cfg.prepare_messages(
        [
            {
                "role": "system",
                "content": "Return only valid JSON. Do not wrap it in Markdown.",
            },
            {"role": "user", "content": prompt},
        ]
    )
    body = {
        "model": cfg.model,
        "temperature": cfg.temperature,
        "messages": messages,
    }
    apply_llm_request_options(body, cfg)
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        endpoint,
        data=data,
        headers={
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=cfg.timeout) as response:
        response_body = response.read().decode("utf-8")
    payload = json.loads(response_body)
    content = payload["choices"][0]["message"]["content"]
    return _loads_json_object(content)


def apply_llm_request_options(body: dict[str, Any], cfg: LlmConfig) -> None:
    body.update(cfg.request_body_options())


def chat_completions_url(url: str) -> str:
    normalized = url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


def _loads_json_object(content: str) -> Any:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(content[start : end + 1])
