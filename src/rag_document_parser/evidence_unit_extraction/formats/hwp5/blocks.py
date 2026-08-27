from __future__ import annotations

from dataclasses import dataclass, field

from .table_extraction import Cell


@dataclass
class TextBlock:
    text: str
    origin: str = "body"
    bbox: dict[str, int | str] | None = None
    shape_type: str = "label"
    instance_id: int | None = None
    ctrl_id: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class DiagramBlock:
    text: str
    nodes: list[dict[str, object]] = field(default_factory=list)
    bboxes: list[dict[str, int | str] | None] = field(default_factory=list)
    connectors: list[dict[str, object]] = field(default_factory=list)


@dataclass
class DrawingLineBlock:
    bbox: dict[str, int | str] | None
    points: list[dict[str, int]]
    arrow: bool = False
    ctrl_id: str | None = None
    instance_id: int | None = None
    payload: bytes | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class TableBlock:
    rows: list[list[Cell]]
    row_count: int | None = None
    column_count: int | None = None


@dataclass
class ImageBlock:
    asset_id: str
    metadata: dict[str, object] = field(default_factory=dict)
