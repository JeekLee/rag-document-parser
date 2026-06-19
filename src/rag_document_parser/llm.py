from __future__ import annotations

import json
import time
from typing import Any, Literal
from urllib import error, request

from pydantic import BaseModel, ConfigDict, Field

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
    max_retries: int = Field(default=2, ge=0)
    retry_backoff_seconds: float = Field(default=0.5, ge=0.0)
    retry_status_codes: tuple[int, ...] = (429, 500, 502, 503, 504)

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
    response_body = _read_response_with_retries(req, cfg)
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


def _read_response_with_retries(req: request.Request, cfg: LlmConfig) -> str:
    attempt = 0
    while True:
        try:
            with request.urlopen(req, timeout=cfg.timeout) as response:
                return response.read().decode("utf-8")
        except Exception as exc:
            if not _should_retry_llm_error(exc, cfg) or attempt >= cfg.max_retries:
                raise
            delay = cfg.retry_backoff_seconds * (2**attempt)
            if delay > 0:
                time.sleep(delay)
            attempt += 1


def _should_retry_llm_error(exc: Exception, cfg: LlmConfig) -> bool:
    if isinstance(exc, error.HTTPError):
        return exc.code in cfg.retry_status_codes
    return isinstance(exc, (error.URLError, TimeoutError, ConnectionError))


def _loads_json_object(content: str) -> Any:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        object_start = content.find("{")
        array_start = content.find("[")
        starts = [index for index in (object_start, array_start) if index != -1]
        if not starts:
            raise
        start = min(starts)
        end_char = "}" if content[start] == "{" else "]"
        end = content.rfind(end_char)
        if end == -1 or end <= start:
            raise
        return json.loads(content[start : end + 1])
