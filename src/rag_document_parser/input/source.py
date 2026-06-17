from __future__ import annotations


def normalize_source(source: bytes | str) -> bytes:
    return source.encode() if isinstance(source, str) else bytes(source)


def normalize_suffix(suffix: str) -> str:
    normalized = suffix.lower()
    if normalized.startswith("."):
        return normalized
    return f".{normalized}"
