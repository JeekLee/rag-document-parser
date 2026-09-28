from __future__ import annotations

import re
from dataclasses import dataclass

from ....models import EvidenceUnit, SourceEvidence
from .diagram import PdfDiagramExtractor
from .models import Segment as _Segment
from .source_projection import PdfTableSourceProjector
from .table_extraction import PdfTableExtractor
from .table_extraction import (
    _is_empty_single_cell_table_artifact,
    _join_lines,
    _single_line_header_only_table_text,
    _title_table_text,
)
from .table_normalization import PdfTableNormalizer

_REVISION_LINE_RE = re.compile(
    r"^(개정\s+[’']?\d{2,4}\.\d{1,2}\.\d{1,2}\.)"
    r"(?:\s+(고시\s+제\d{4}-\d+호))?"
    r"(?:\s+\(([^)]*시행)\))?"
    r"\s*(.*)$"
)
_DIAGRAM_EXTRACTOR = PdfDiagramExtractor()
_TABLE_EXTRACTOR = PdfTableExtractor()
_PDF_TABLE_NORMALIZER = PdfTableNormalizer()
_SOURCE_PROJECTOR = PdfTableSourceProjector()
_crop_text = _TABLE_EXTRACTOR.crop_text
_diagram_source_text = _DIAGRAM_EXTRACTOR.source_text
_is_pdf_artifact_text = _TABLE_EXTRACTOR.is_artifact_text
_is_table_continuation = _TABLE_EXTRACTOR.is_continuation
_resolve_nested_tables = _TABLE_EXTRACTOR.resolve_nested
_simple_cell = _PDF_TABLE_NORMALIZER.simple_cell
_structured_table_from_pdf_table = _TABLE_EXTRACTOR.build
_table_source_text = _SOURCE_PROJECTOR.project


@dataclass(frozen=True)
class PdfSegmentProcessor:
    """Orders page segments and materializes canonical PDF evidence units."""

    def order_page(
        self,
        page: object,
        page_number: int,
        tables: list[object],
        image_segments: list[_Segment],
        diagram_segments: list[_Segment],
        cell_image_children: dict[int, dict[tuple[int, int], list[dict[str, object]]]],
        table_text_fallbacks: dict[int, list[list[str]]] | None = None,
    ) -> list[_Segment]:
        return _page_segments_ordered(
            page,
            page_number,
            tables,
            image_segments,
            diagram_segments,
            cell_image_children,
            table_text_fallbacks,
        )

    def to_units(self, page_segments: list[list[_Segment]]) -> list[EvidenceUnit]:
        return _segments_to_units(page_segments)

    def merge_continuation_tables(
        self,
        page_segments: list[list[_Segment]],
    ) -> list[list[_Segment]]:
        return _merge_continuation_tables(page_segments)

    def expand_text(
        self,
        page_segments: list[list[_Segment]],
    ) -> list[list[_Segment]]:
        return _expand_text_segments(page_segments)


def _page_segments_ordered(
    page: object,
    page_number: int,
    tables: list[object],
    image_segments: list[_Segment],
    diagram_segments: list[_Segment],
    cell_image_children: dict[int, dict[tuple[int, int], list[dict[str, object]]]],
    table_text_fallbacks: dict[int, list[list[str]]] | None = None,
) -> list[_Segment]:
    segments: list[_Segment] = []
    nested = _resolve_nested_tables(tables)
    for table_idx, table in enumerate(tables):
        if table_idx in nested.suppressed:
            continue
        structured = _structured_table_from_pdf_table(
            page,
            tables,
            table_idx,
            nested,
            seen=set(),
            cell_image_children=cell_image_children,
            fallback_rows=(table_text_fallbacks or {}).get(table_idx),
        )
        if structured is None:
            continue
        if _is_empty_single_cell_table_artifact(structured):
            continue
        text_box = _title_table_text(structured)
        if text_box is None:
            text_box = _single_line_header_only_table_text(table, structured)
        if text_box is not None:
            segments.append(
                _Segment(
                    top=float(table.bbox[1]),
                    bottom=float(table.bbox[3]),
                    kind="text",
                    payload=text_box,
                    page=page_number,
                )
            )
            continue
        if not structured["columns"] and not structured["rows"]:
            continue
        segments.append(
            _Segment(
                top=float(table.bbox[1]),
                bottom=float(table.bbox[3]),
                kind="table",
                payload=structured,
                page=page_number,
            )
        )

    segments.extend(image_segments)
    segments.extend(diagram_segments)
    obstacle_bands = sorted(
        [(segment.top, segment.bottom) for segment in segments],
        key=lambda band: band[0],
    )
    prev_bottom = 0.0
    page_height = float(getattr(page, "height", 0.0))
    page_width = float(getattr(page, "width", 0.0))
    for top, bottom in obstacle_bands:
        if top > prev_bottom + 2:
            text = _crop_text(page, 0.0, prev_bottom, page_width, top)
            if text:
                segments.append(
                    _Segment(
                        top=prev_bottom,
                        bottom=top,
                        kind="text",
                        payload=text,
                        page=page_number,
                    )
                )
        prev_bottom = max(prev_bottom, bottom)
    if page_height and prev_bottom < page_height - 2:
        text = _crop_text(page, 0.0, prev_bottom, page_width, page_height)
        if text:
            segments.append(
                _Segment(
                    top=prev_bottom,
                    bottom=page_height,
                    kind="text",
                    payload=text,
                    page=page_number,
                )
            )

    return sorted(segments, key=lambda segment: segment.top)


def _segments_to_units(page_segments: list[list[_Segment]]) -> list[EvidenceUnit]:
    units: list[EvidenceUnit] = []
    block_index = 1
    table_index = 1

    for segments in page_segments:
        for segment in sorted(segments, key=lambda item: item.top):
            if segment.kind == "text":
                text = str(segment.payload)
                if not segment.metadata.get("table_fallback"):
                    text = text.strip()
                if not text.strip() or _is_pdf_artifact_text(text):
                    continue
                pdf_metadata = {"page": segment.page}
                pdf_metadata.update(segment.metadata)
                units.append(
                    EvidenceUnit(
                        id=f"b{block_index}",
                        type="text",
                        format="plain",
                        source=SourceEvidence(kind="text", text=text),
                        content=text,
                        metadata={
                            "common": {
                                "chunk_kind": "text",
                                "section_path": [],
                                "display_format": "plain",
                            },
                            "pdf": pdf_metadata,
                        },
                    )
                )
                block_index += 1
                continue
            if segment.kind == "table":
                table = segment.payload
                table_id = f"t{table_index}"
                headers = [str(column["text"]) for column in table["columns"]]
                pdf_metadata = {
                    "page": segment.page,
                    "confidence": segment.metadata.get("confidence", "high"),
                }
                if segment.metadata.get("ocr"):
                    pdf_metadata["ocr"] = True
                units.append(
                    EvidenceUnit(
                        id=f"b{block_index}",
                        type="table",
                        format="structured_table",
                        source=SourceEvidence(
                            kind="table",
                            text=_table_source_text(table),
                        ),
                        content=table,
                        metadata={
                            "common": {
                                "chunk_kind": "table",
                                "section_path": [],
                                "display_format": "structured_table",
                            },
                            "table": {
                                "table_id": table_id,
                                "headers": headers,
                                "row_count": len(table["rows"]),
                            },
                            "pdf": pdf_metadata,
                        },
                    )
                )
                block_index += 1
                table_index += 1
                continue
            if segment.kind == "image":
                asset_id = str(segment.payload["asset_id"])
                units.append(
                    EvidenceUnit(
                        id=f"b{block_index}",
                        type="image",
                        format="asset_ref",
                        source=SourceEvidence(
                            kind="image",
                            text=f"image: {asset_id}",
                        ),
                        content=dict(segment.payload),
                        metadata={
                            "common": {
                                "chunk_kind": "image",
                                "section_path": [],
                                "display_format": "image",
                            },
                            "asset": {"asset_id": asset_id},
                            "pdf": {
                                "page": segment.page,
                                "confidence": segment.metadata.get(
                                    "confidence",
                                    "medium",
                                ),
                            },
                        },
                    )
                )
                block_index += 1
                continue
            if segment.kind == "diagram":
                diagram = segment.payload
                source_text = _diagram_source_text(diagram)
                if not source_text:
                    asset_id = str(diagram.get("asset_id", "")).strip()
                    source_text = f"diagram image: {asset_id}" if asset_id else ""
                if not source_text:
                    continue
                confidence = str(
                    segment.metadata.get(
                        "confidence",
                        diagram.get("confidence", "low"),
                    )
                )
                diagram_metadata: dict[str, object] = {
                    "node_count": len(diagram.get("nodes", [])),
                    "edge_count": len(diagram.get("edges", [])),
                    "confidence": confidence,
                }
                if diagram.get("asset_id"):
                    diagram_metadata["fallback_asset_id"] = diagram["asset_id"]
                units.append(
                    EvidenceUnit(
                        id=f"b{block_index}",
                        type="diagram",
                        format="structured_diagram",
                        source=SourceEvidence(kind="diagram", text=source_text),
                        content=diagram,
                        metadata={
                            "common": {
                                "chunk_kind": "diagram",
                                "section_path": [],
                                "display_format": "structured_diagram",
                            },
                            "diagram": diagram_metadata,
                            "pdf": {
                                "page": segment.page,
                                "confidence": confidence,
                            },
                        },
                    )
                )
                block_index += 1

    return units


def _merge_continuation_tables(page_segments: list[list[_Segment]]) -> list[list[_Segment]]:
    merged_pages: list[list[_Segment]] = [[] for _ in page_segments]
    previous_table: _Segment | None = None
    previous_table_end_page: int | None = None

    for page_idx, segments in enumerate(page_segments):
        page_has_content_before_table = False
        for segment in sorted(segments, key=lambda item: item.top):
            if segment.kind != "table":
                merged_pages[page_idx].append(segment)
                is_artifact_text = (
                    segment.kind == "text"
                    and _is_pdf_artifact_text(str(segment.payload).strip())
                )
                if segment.kind in {"text", "image", "diagram"} and not is_artifact_text:
                    page_has_content_before_table = True
                    previous_table = None
                continue

            if (
                previous_table is not None
                and not page_has_content_before_table
                and not (
                    previous_table_end_page == segment.page
                    and previous_table.metadata.get("ocr")
                    and segment.metadata.get("ocr")
                )
                and _is_table_continuation(previous_table.payload, segment.payload)
            ):
                _PDF_TABLE_NORMALIZER.append_rows(
                    previous_table.payload,
                    segment.payload,
                )
                previous_table_end_page = segment.page
                continue

            merged_pages[page_idx].append(segment)
            previous_table = segment
            previous_table_end_page = segment.page

    return merged_pages


def _expand_text_segments(
    page_segments: list[list[_Segment]],
) -> list[list[_Segment]]:
    expanded_pages: list[list[_Segment]] = []
    for segments in page_segments:
        expanded: list[_Segment] = []
        for segment in segments:
            if segment.kind != "text" or segment.metadata.get("table_fallback"):
                expanded.append(segment)
                continue
            parts = _revision_history_parts(str(segment.payload))
            if parts is None:
                text_parts = _structured_text_parts(str(segment.payload))
                if text_parts is None:
                    expanded.append(segment)
                    continue
                parts = [("text", part) for part in text_parts]
            for part_index, (kind, payload) in enumerate(parts):
                expanded.append(
                    _Segment(
                        top=segment.top + (part_index * 0.001),
                        bottom=segment.bottom,
                        kind=kind,
                        payload=payload,
                        page=segment.page,
                    )
                )
                continue
        expanded_pages.append(
            _drop_duplicate_short_title_segments(
                sorted(expanded, key=lambda item: item.top)
            )
        )
    return expanded_pages


def _structured_text_parts(text: str) -> list[str] | None:
    return (
        _scanned_official_letter_text_parts(text)
        or _scanned_official_letter_continuation_parts(text)
        or _official_notice_text_parts(text)
        or _related_basis_text_parts(text)
        or _sectioned_text_parts(text)
        or _short_heading_text_parts(text)
    )


def _drop_duplicate_short_title_segments(segments: list[_Segment]) -> list[_Segment]:
    result: list[_Segment] = []
    index = 0
    while index < len(segments):
        segment = segments[index]
        next_segment = segments[index + 1] if index + 1 < len(segments) else None
        if _is_duplicate_short_title_segment(segment, next_segment):
            index += 1
            continue
        result.append(segment)
        index += 1
    return result


def _is_duplicate_short_title_segment(
    segment: _Segment,
    next_segment: _Segment | None,
) -> bool:
    if segment.kind != "text" or next_segment is None or next_segment.kind != "text":
        return False
    if segment.page != next_segment.page:
        return False
    title = str(segment.payload).strip()
    next_text = str(next_segment.payload).strip()
    if (
        0 < len(title) <= 30
        and "\n" not in title
        and title.endswith("대상")
        and title in next_text
        and bool(re.match(r"^[가-힣]\.", next_text))
    ):
        return True
    if not _is_short_pdf_text_box_segment(segment, title):
        return False
    normalized_title = _normalize_duplicate_text(title)
    normalized_next = _normalize_duplicate_text(next_text)
    return (
        len(normalized_title) >= 6
        and normalized_title in normalized_next
    )


def _is_short_pdf_text_box_segment(segment: _Segment, text: str) -> bool:
    return (
        0 < len(text) <= 160
        and "\n" not in text
        and segment.bottom - segment.top <= 24
    )


def _normalize_duplicate_text(text: str) -> str:
    return re.sub(r"\s+", "", text.strip())


def _official_notice_text_parts(text: str) -> list[str] | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or not lines[0].startswith("보건복지부 고시 "):
        return None

    parts = [lines[0]]
    paragraph: list[str] = []
    for line in lines[1:]:
        if _is_standalone_notice_line(line):
            if paragraph:
                parts.append(_join_pdf_text_lines(paragraph))
                paragraph = []
            parts.append(line)
            continue
        paragraph.append(line)
        if line.endswith("다."):
            parts.append(_join_pdf_text_lines(paragraph))
            paragraph = []
    if paragraph:
        parts.append(_join_pdf_text_lines(paragraph))
    return parts if len(parts) > 1 else None


def _scanned_official_letter_text_parts(text: str) -> list[str] | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not _looks_like_scanned_official_letter(lines):
        return None

    parts: list[str] = []
    paragraph: list[str] = []
    for line in lines:
        if _is_official_letter_boundary_line(line):
            if paragraph:
                parts.append(_join_pdf_text_lines(paragraph))
                paragraph = []
            if _is_numbered_paragraph_line(line):
                paragraph = [line]
            else:
                parts.append(line)
            continue
        if paragraph:
            paragraph.append(line)
            continue
        parts.append(line)

    if paragraph:
        parts.append(_join_pdf_text_lines(paragraph))
    return parts if len(parts) > 1 else None


def _looks_like_scanned_official_letter(lines: list[str]) -> bool:
    return (
        any(line.startswith("수신자") for line in lines)
        and any(line.startswith("제목") for line in lines)
        and any(_is_numbered_paragraph_line(line) for line in lines)
    )


def _is_official_letter_boundary_line(line: str) -> bool:
    return (
        _is_numbered_paragraph_line(line)
        or line.startswith("붙임")
        or line.startswith('"긴급지원')
        or line == "보건복지부"
        or line.startswith("수신자")
        or line == "(경유)"
        or line.startswith("제목")
    )


def _is_numbered_paragraph_line(line: str) -> bool:
    return bool(re.match(r"^\d+\.\s+", line))


def _scanned_official_letter_continuation_parts(text: str) -> list[str] | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1 and lines[0] == "보건복지부" and lines[1].startswith("수신자"):
        return _split_official_letter_continuation_lines(lines)
    if len(lines) != 1:
        return None
    line = lines[0]
    if not (line.startswith("보건복지부 수신자 ") and " 시행 " in line and " 전화 " in line):
        return None
    parts = _split_official_letter_continuation_line(line)
    return parts if len(parts) > 1 else None


def _split_official_letter_continuation_lines(lines: list[str]) -> list[str]:
    parts: list[str] = []
    for line in lines:
        split = _split_official_letter_continuation_line(line)
        parts.extend(split if len(split) > 1 else [line])
    return parts


def _split_official_letter_continuation_line(line: str) -> list[str]:
    markers = (
        " 수신자 ",
        " 주무관 ",
        " 주주관 ",
        " 주관 ",
        " 시행 ",
        " 접수 ",
        " 우 ",
        " 전화 ",
    )
    split_points = sorted(
        index
        for marker in markers
        for index in [line.find(marker)]
        if index > 0
    )
    parts: list[str] = []
    start = 0
    for index in split_points:
        part = line[start:index].strip()
        if part:
            parts.append(part)
        start = index + 1
    tail = line[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _related_basis_text_parts(text: str) -> list[str] | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 3 or lines[1] != "1. 관련 근거":
        return None
    if not all(line.startswith("○") for line in lines[2:]):
        return None
    return lines


def _sectioned_text_parts(text: str) -> list[str] | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2 or not any(_is_section_heading_line(line) for line in lines):
        return None

    parts: list[str] = []
    paragraph: list[str] = []
    for line in lines:
        if _is_section_heading_line(line):
            if paragraph:
                parts.append(_join_section_text_lines(paragraph))
                paragraph = []
            parts.append(line)
            continue
        paragraph.append(line)
    if paragraph:
        parts.append(_join_section_text_lines(paragraph))
    return parts if len(parts) > 1 else None


def _is_section_heading_line(line: str) -> bool:
    stripped = line.strip()
    return (
        stripped.startswith("□")
        or stripped == "일반사항"
        or stripped == "⋮"
        or stripped.startswith("개정 ")
        or (stripped.startswith("「") and stripped.endswith("Q&A"))
        or (stripped.endswith("Q&A") and len(stripped) <= 60)
        or (stripped.startswith("<") and stripped.endswith(">"))
    )


def _is_standalone_notice_line(line: str) -> bool:
    return (
        bool(re.match(r"^\d{4}년\s+\d{1,2}월\s+\d{1,2}일$", line))
        or line == "보건복지부 장관"
        or line.endswith("일부개정")
    )


def _short_heading_text_parts(text: str) -> list[str] | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2 or len(lines) > 3:
        return None
    if all(len(line) <= 40 and not line.endswith(("다.", ".")) for line in lines):
        return lines
    return None


def _join_pdf_text_lines(lines: list[str]) -> str:
    text = " ".join(line.strip() for line in lines if line.strip())
    text = text.replace("세부 사항", "세부사항")
    return text.strip()


def _join_section_text_lines(lines: list[str]) -> str:
    groups: list[list[str]] = []
    current: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if current and _is_section_text_boundary_line(line):
            groups.append(current)
            current = [line]
            continue
        current.append(line)
    if current:
        groups.append(current)
    return "\n".join(_join_pdf_text_lines(group) for group in groups).strip()


def _is_section_text_boundary_line(line: str) -> bool:
    stripped = line.strip()
    return bool(
        stripped.startswith(("■", "□", "○", "〇", "-", "ㆍ", "․", "*", "※"))
        or re.match(r"^\d+\.\s+", stripped)
        or re.match(r"^\(?\d+\)\s*", stripped)
        or re.match(r"^[가-힣]\.\s+", stripped)
        or re.match(r"^제\d+조(?:의\d+)?\(", stripped)
    )


def _revision_history_parts(text: str) -> list[tuple[str, object]] | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    try:
        marker_index = lines.index("관련 근거")
    except ValueError:
        return None

    rows: list[dict[str, object]] = []
    after_index = marker_index + 1
    for line in lines[marker_index + 1:]:
        row = _revision_history_row(line)
        if row is None:
            break
        row["index"] = len(rows) + 1
        rows.append(row)
        after_index += 1
    if len(rows) < 2:
        return None

    parts: list[tuple[str, object]] = [
        ("text", line)
        for line in lines[:marker_index]
        if line
    ]
    parts.append(
        (
            "table",
            {
                "caption": None,
                "columns": [
                    {"id": "c1", "text": "개정일"},
                    {"id": "c2", "text": "고시"},
                    {"id": "c3", "text": "시행일"},
                    {"id": "c4", "text": "관련 근거"},
                ],
                "header_rows": [
                    {
                        "index": 1,
                        "cells": [
                            _simple_cell("c1", "개정일"),
                            _simple_cell("c2", "고시"),
                            _simple_cell("c3", "시행일"),
                            _simple_cell("c4", "관련 근거"),
                        ],
                    }
                ],
                "rows": rows,
            },
        )
    )
    parts.extend(("text", part) for part in _trailing_revision_texts(lines[after_index:]))
    return parts


def _revision_history_row(line: str) -> dict[str, object] | None:
    match = _REVISION_LINE_RE.match(line)
    if match is None:
        return None
    values = [(part or "").strip() for part in match.groups()]
    return {
        "index": 0,
        "cells": [
            _simple_cell("c1", values[0]),
            _simple_cell("c2", values[1]),
            _simple_cell("c3", values[2]),
            _simple_cell("c4", values[3]),
        ],
    }


def _trailing_revision_texts(lines: list[str]) -> list[str]:
    result: list[str] = []
    paragraph: list[str] = []
    for line in lines:
        if line.startswith("*"):
            if paragraph:
                result.append(_join_lines("\n".join(paragraph)))
                paragraph = []
            result.append(line)
            continue
        if line.startswith("※") and paragraph:
            result.append(_join_lines("\n".join(paragraph)))
            paragraph = [line]
            continue
        paragraph.append(line)
    if paragraph:
        result.append(_join_lines("\n".join(paragraph)))
    return result
