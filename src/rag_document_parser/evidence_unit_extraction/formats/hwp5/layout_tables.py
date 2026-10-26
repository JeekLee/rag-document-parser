from __future__ import annotations

from collections.abc import Iterable, Iterator

from .blocks import Block, TableBlock, TextBlock
from .table_extraction import Hwp5TableBuilder


def expand_layout_tables(blocks: Iterable[Block]) -> Iterator[Block]:
    """Promote ordered cell blocks only through genuine 1x1 table wrappers.

    A real data table is a boundary: its cells and nested tables stay intact.
    Image/diagram-only cells also retain their existing representation.
    """
    for block in blocks:
        if not isinstance(block, TableBlock):
            yield block
            continue
        if _is_layout_wrapper(block):
            yield from expand_layout_tables(block.rows[0][0].blocks)
            continue
        # A nested text box promoted out of a wrapper follows the same path as
        # a top-level text box. Do not introduce a one-cell data table for it.
        text = Hwp5TableBuilder().single_cell_text(
            block.rows, row_count=block.row_count, column_count=block.column_count,
        )
        yield TextBlock(text) if text is not None else block


def _is_layout_wrapper(block: TableBlock) -> bool:
    if (
        block.row_count not in (None, 1) or block.column_count not in (None, 1)
        or len(block.rows) != 1 or len(block.rows[0]) != 1
    ):
        return False
    cell = block.rows[0][0]
    return (
        cell.row_addr in (None, 0) and cell.col_addr in (None, 0)
        and cell.rowspan == cell.colspan == 1
        and any(isinstance(child, TableBlock) for child in cell.blocks)
    )
