from __future__ import annotations

import re
from typing import Any

_TOKEN_ENCODER: Any | None = None
_TOKEN_ENCODER_UNAVAILABLE = False


def count_tokens(text: str) -> int:
    encoder = _token_encoder()
    if encoder is not None:
        return len(encoder.encode(text or ""))
    return _estimated_token_count(text)


def _token_encoder() -> Any | None:
    global _TOKEN_ENCODER, _TOKEN_ENCODER_UNAVAILABLE
    if _TOKEN_ENCODER is not None:
        return _TOKEN_ENCODER
    if _TOKEN_ENCODER_UNAVAILABLE:
        return None

    try:
        import tiktoken  # type: ignore[import-not-found]
    except Exception:
        _TOKEN_ENCODER_UNAVAILABLE = True
        return None

    try:
        _TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _TOKEN_ENCODER_UNAVAILABLE = True
        return None
    return _TOKEN_ENCODER


def _estimated_token_count(text: str) -> int:
    count = 0
    for token in re.findall(r"[A-Za-z0-9_]+|[가-힣]|[^\s]", text or ""):
        if re.fullmatch(r"[A-Za-z0-9_]+", token):
            count += max(1, (len(token) + 3) // 4)
        else:
            count += 1
    return count
