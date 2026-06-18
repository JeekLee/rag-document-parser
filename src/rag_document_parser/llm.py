from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib import request


@dataclass(frozen=True)
class LlmConfig:
    url: str
    api_key: str
    model: str
    temperature: float = 0.0
    timeout: float = 120.0


def chat_json(prompt: str, cfg: LlmConfig) -> Any:
    endpoint = chat_completions_url(cfg.url)
    body = {
        "model": cfg.model,
        "temperature": cfg.temperature,
        "messages": [
            {
                "role": "system",
                "content": "Return only valid JSON. Do not wrap it in Markdown.",
            },
            {"role": "user", "content": prompt},
        ],
    }
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
