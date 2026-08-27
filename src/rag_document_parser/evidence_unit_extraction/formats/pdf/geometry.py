from __future__ import annotations


def _coerce_bbox(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        return None
    try:
        return tuple(float(part) for part in value)
    except (TypeError, ValueError):
        return None


def _bbox_overlap_ratio(
    first: tuple[float, float, float, float] | None,
    second: tuple[float, float, float, float] | None,
) -> float:
    if first is None or second is None:
        return 0.0
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    smaller_area = min(_bbox_area(first), _bbox_area(second))
    return intersection / smaller_area if smaller_area else 0.0


def _bbox_in_cell(
    sub_bbox: tuple[float, float, float, float],
    cell_bbox: tuple[float, float, float, float],
    tol: float = 2.0,
) -> bool:
    sub_x0, sub_top, sub_x1, sub_bottom = sub_bbox
    cell_x0, cell_top, cell_x1, cell_bottom = cell_bbox
    return (
        sub_x0 >= cell_x0 - tol
        and sub_x1 <= cell_x1 + tol
        and sub_top >= cell_top - tol
        and sub_bottom <= cell_bottom + tol
    )


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _bbox_near_equal(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
    margin: float = 3.0,
) -> bool:
    return all(abs(first[index] - second[index]) <= margin for index in range(4))
