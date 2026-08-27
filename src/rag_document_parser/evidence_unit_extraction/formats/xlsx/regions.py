from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class XlsxRegionDetector:
    """Finds logical worksheet regions while respecting declared-table barriers."""

    def detect(
        self,
        occupied: set[tuple[int, int]],
        *,
        barriers: list[Region] | None = None,
    ) -> list[Region]:
        return _logical_regions(occupied, barriers=barriers)


@dataclass(frozen=True)
class Region:
    min_row: int
    max_row: int
    min_col: int
    max_col: int

    @property
    def width(self) -> int:
        return self.max_col - self.min_col + 1

    @property
    def height(self) -> int:
        return self.max_row - self.min_row + 1

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class RegionIntervalNode:
    center: int
    by_start: list[Region]
    by_end: list[Region]
    starts: list[int]
    negative_ends: list[int]
    left: RegionIntervalNode | None = None
    right: RegionIntervalNode | None = None


@dataclass(frozen=True)
class RegionSpatialIndex:
    rows: RegionIntervalNode | None
    columns: RegionIntervalNode | None
    row_starts: tuple[int, ...]
    row_ends: tuple[int, ...]
    column_starts: tuple[int, ...]
    column_ends: tuple[int, ...]


def _regions_overlap(left: Region, right: Region) -> bool:
    return not (
        left.max_row < right.min_row
        or right.max_row < left.min_row
        or left.max_col < right.min_col
        or right.max_col < left.min_col
    )


def _logical_regions(
    occupied: set[tuple[int, int]],
    *,
    barriers: list[Region] | None = None,
) -> list[Region]:
    groups = _logical_region_groups(occupied)
    if not barriers:
        return [region for region, _ in groups]

    barrier_index = RegionSpatialIndex(
        rows=_build_region_interval_index(barriers, axis="row"),
        columns=_build_region_interval_index(barriers, axis="column"),
        row_starts=tuple(sorted(region.min_row for region in barriers)),
        row_ends=tuple(sorted(region.max_row for region in barriers)),
        column_starts=tuple(sorted(region.min_col for region in barriers)),
        column_ends=tuple(sorted(region.max_col for region in barriers)),
    )
    regions: list[Region] = []
    for region, positions in groups:
        overlapping = _overlapping_indexed_regions(barrier_index, region)
        if not overlapping:
            regions.append(region)
            continue

        row_cuts = {region.min_row, region.max_row + 1}
        for barrier in overlapping:
            row_cuts.add(max(region.min_row, barrier.min_row))
            row_cuts.add(min(region.max_row + 1, barrier.max_row + 1))
        sorted_cuts = sorted(row_cuts)
        positions_by_band: defaultdict[int, set[tuple[int, int]]] = defaultdict(set)
        for position in positions:
            band = bisect_right(sorted_cuts, position[0]) - 1
            positions_by_band[band].add(position)
        for band in sorted(positions_by_band):
            regions.extend(
                fragment
                for fragment, _ in _logical_region_groups(positions_by_band[band])
            )
    return sorted(regions, key=lambda item: (item.min_row, item.min_col))


def _logical_region_groups(
    occupied: set[tuple[int, int]],
) -> list[tuple[Region, set[tuple[int, int]]]]:
    columns_by_row: defaultdict[int, set[int]] = defaultdict(set)
    for row, column in occupied:
        columns_by_row[row].add(column)
    rows = sorted(columns_by_row)
    groups: list[tuple[Region, set[tuple[int, int]]]] = []
    for min_row, max_row in _contiguous_ranges(rows):
        band_positions = [
            (row, column)
            for row in range(min_row, max_row + 1)
            for column in columns_by_row[row]
        ]
        band_columns = sorted({column for _, column in band_positions})
        column_ranges = _contiguous_ranges(band_columns)
        range_by_column = {
            column: range_index
            for range_index, (min_col, max_col) in enumerate(column_ranges)
            for column in range(min_col, max_col + 1)
        }
        grouped_positions: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
        for position in band_positions:
            grouped_positions[range_by_column[position[1]]].append(position)
        for range_index, (min_col, max_col) in enumerate(column_ranges):
            positions = grouped_positions[range_index]
            groups.append(
                (
                    Region(
                        min_row=min(row for row, _ in positions),
                        max_row=max(row for row, _ in positions),
                        min_col=min(column for _, column in positions),
                        max_col=max(column for _, column in positions),
                    ),
                    set(positions),
                )
            )
    return sorted(
        groups,
        key=lambda item: (item[0].min_row, item[0].min_col),
    )


def _build_region_interval_index(
    regions: list[Region],
    *,
    axis: str,
) -> RegionIntervalNode | None:
    if not regions:
        return None
    centers = sorted(sum(_region_axis_bounds(region, axis)) // 2 for region in regions)
    center = centers[len(centers) // 2]
    left: list[Region] = []
    right: list[Region] = []
    spanning: list[Region] = []
    for region in regions:
        start, end = _region_axis_bounds(region, axis)
        if end < center:
            left.append(region)
        elif start > center:
            right.append(region)
        else:
            spanning.append(region)
    by_start = sorted(
        spanning,
        key=lambda region: _region_axis_bounds(region, axis)[0],
    )
    by_end = sorted(
        spanning,
        key=lambda region: _region_axis_bounds(region, axis)[1],
        reverse=True,
    )
    return RegionIntervalNode(
        center=center,
        by_start=by_start,
        by_end=by_end,
        starts=[_region_axis_bounds(region, axis)[0] for region in by_start],
        negative_ends=[-_region_axis_bounds(region, axis)[1] for region in by_end],
        left=_build_region_interval_index(left, axis=axis),
        right=_build_region_interval_index(right, axis=axis),
    )


def _overlapping_indexed_regions(
    index: RegionSpatialIndex,
    query: Region,
) -> list[Region]:
    row_count = _interval_overlap_count(
        index.row_starts,
        index.row_ends,
        query.min_row,
        query.max_row,
    )
    column_count = _interval_overlap_count(
        index.column_starts,
        index.column_ends,
        query.min_col,
        query.max_col,
    )
    if row_count <= column_count:
        candidates = _query_interval_index(
            index.rows,
            query.min_row,
            query.max_row,
        )
        return [
            region
            for region in candidates
            if not (region.max_col < query.min_col or region.min_col > query.max_col)
        ]
    candidates = _query_interval_index(
        index.columns,
        query.min_col,
        query.max_col,
    )
    return [
        region
        for region in candidates
        if not (region.max_row < query.min_row or region.min_row > query.max_row)
    ]


def _query_interval_index(
    node: RegionIntervalNode | None,
    query_start: int,
    query_end: int,
) -> list[Region]:
    if node is None:
        return []
    candidates: list[Region] = []
    if query_end < node.center:
        stop = bisect_right(node.starts, query_end)
        candidates.extend(node.by_start[:stop])
        candidates.extend(_query_interval_index(node.left, query_start, query_end))
    elif query_start > node.center:
        stop = bisect_right(node.negative_ends, -query_start)
        candidates.extend(node.by_end[:stop])
        candidates.extend(_query_interval_index(node.right, query_start, query_end))
    else:
        candidates.extend(node.by_start)
        candidates.extend(_query_interval_index(node.left, query_start, query_end))
        candidates.extend(_query_interval_index(node.right, query_start, query_end))
    return candidates


def _interval_overlap_count(
    starts: tuple[int, ...],
    ends: tuple[int, ...],
    query_start: int,
    query_end: int,
) -> int:
    ending_before = bisect_right(ends, query_start - 1)
    starting_after = len(starts) - bisect_right(starts, query_end)
    return len(starts) - ending_before - starting_after


def _region_axis_bounds(region: Region, axis: str) -> tuple[int, int]:
    if axis == "row":
        return region.min_row, region.max_row
    return region.min_col, region.max_col


def _contiguous_ranges(values: list[int]) -> list[tuple[int, int]]:
    if not values:
        return []
    ranges: list[tuple[int, int]] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append((start, previous))
        start = previous = value
    ranges.append((start, previous))
    return ranges
