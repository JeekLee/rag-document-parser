from __future__ import annotations

import re


def clean_text(text: str) -> str:
    text = "".join(char for char in text if char == "\n" or char > "\x1f")
    return re.sub(r"[ \t\r\f\v]+", " ", text).strip()
