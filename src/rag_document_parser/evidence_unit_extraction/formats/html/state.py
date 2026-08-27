from __future__ import annotations

from ....models import PendingAsset


class HtmlParseState:
    def __init__(self) -> None:
        self._block_index = 1
        self._table_index = 1
        self._image_index = 1
        self._headings: list[tuple[int, str]] = []
        self.assets: list[PendingAsset] = []
        self.quality_warnings: list[dict[str, object]] = []

    @property
    def section_path(self) -> list[str]:
        return [text for _, text in self._headings]

    def next_block_id(self) -> str:
        block_id = f"b{self._block_index}"
        self._block_index += 1
        return block_id

    def next_table_id(self) -> str:
        table_id = f"t{self._table_index}"
        self._table_index += 1
        return table_id

    def next_image_id(self) -> str:
        image_id = f"img-{self._image_index:04d}"
        self._image_index += 1
        return image_id

    def set_heading(self, level: int, text: str) -> None:
        if not text:
            return
        while self._headings and self._headings[-1][0] >= level:
            self._headings.pop()
        self._headings.append((level, text))
