from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ....models import PendingAsset
from ...backend import ParsedDocument
from ...ocr import OcrFn
from .blocks import DiagramBlock, DrawingLineBlock, ImageBlock, TableBlock, TextBlock


@dataclass
class ParsedBlocks:
    blocks: list[
        TextBlock | DiagramBlock | DrawingLineBlock | TableBlock | ImageBlock
    ] = field(default_factory=list)
    assets: list[PendingAsset] = field(default_factory=list)
    saw_drawing: bool = False
    missing_image_count: int = 0
    quality_warnings: list[dict[str, Any]] = field(default_factory=list)

    def extend(self, other: ParsedBlocks) -> None:
        self.blocks.extend(other.blocks)
        self.assets.extend(other.assets)
        self.saw_drawing = self.saw_drawing or other.saw_drawing
        self.missing_image_count += other.missing_image_count
        self.quality_warnings.extend(other.quality_warnings)

    def to_document(
        self,
        ocr_fn: OcrFn | None = None,
    ) -> ParsedDocument:
        from .document_assembly import Hwp5DocumentAssembler

        return Hwp5DocumentAssembler(ocr_fn=ocr_fn).assemble(self)
