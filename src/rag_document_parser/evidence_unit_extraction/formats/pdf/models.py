from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PdfImage:
    data: bytes
    mime: str
    ext: str
    is_diagram: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Segment:
    top: float
    bottom: float
    kind: str
    payload: Any
    page: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NestedResolution:
    suppressed: set[int]
    children: dict[int, dict[tuple[int, int], list[int]]]


@dataclass(frozen=True)
class TableCellSpans:
    spans: dict[tuple[int, int], tuple[int, int]]
    covered: set[tuple[int, int]]
