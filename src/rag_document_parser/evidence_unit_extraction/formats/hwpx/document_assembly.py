from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ....models import EvidenceUnit, PendingAsset, SourceEvidence
from ...ocr import OcrFn, OcrResult, coerce_ocr_result
from .diagram import HwpxDiagramBuilder
from .xml_utils import clean_text as _clean_text

_DIAGRAM_BUILDER = HwpxDiagramBuilder()
_diagram_source_text = _DIAGRAM_BUILDER.source_text


@dataclass(frozen=True)
class HwpxDocumentAssembler:
    """Creates HWPX evidence units and applies optional image OCR fallback."""

    ocr_fn: OcrFn | None = None

    def text_unit(self, unit_id: str, text: str) -> EvidenceUnit:
        return _text_unit(unit_id, text)

    def diagram_unit(
        self,
        unit_id: str,
        structured: dict[str, object],
    ) -> EvidenceUnit:
        return _diagram_unit(unit_id, structured)

    def append_ocr_units(
        self,
        units: list[EvidenceUnit],
        assets: list[PendingAsset],
        warnings: list[dict[str, Any]],
        block_index: int,
    ) -> int:
        return _append_ocr_fallback_units(
            units,
            assets,
            warnings,
            self.ocr_fn,
            block_index,
        )


def _text_unit(unit_id: str, text: str) -> EvidenceUnit:
    return EvidenceUnit(
        id=unit_id,
        type="text",
        format="plain",
        source=SourceEvidence(kind="text", text=text),
        content=text,
        metadata={
            "common": {
                "chunk_kind": "text",
                "section_path": [],
                "display_format": "plain",
            }
        },
    )


def _diagram_unit(unit_id: str, structured: dict[str, object]) -> EvidenceUnit:
    return EvidenceUnit(
        id=unit_id,
        type="diagram",
        format="structured_diagram",
        source=SourceEvidence(kind="diagram", text=_diagram_source_text(structured)),
        content=structured,
        metadata={
            "common": {
                "chunk_kind": "diagram",
                "section_path": [],
                "display_format": "structured_diagram",
            },
            "diagram": {
                "node_count": len(structured["nodes"]),
                "edge_count": len(structured["edges"]),
            },
        },
    )


def _append_ocr_fallback_units(
    units: list[EvidenceUnit],
    assets: list[PendingAsset],
    warnings: list[dict[str, Any]],
    ocr_fn: OcrFn | None,
    block_index: int,
) -> int:
    if ocr_fn is None or not assets or _has_native_source_text(units):
        return block_index

    for image_index, asset in enumerate(assets):
        try:
            raw_result = ocr_fn(asset.data, image_index)
            result = coerce_ocr_result(raw_result)
        except Exception as exc:
            warnings.append(
                _hwpx_ocr_failed_warning(image_index + 1, asset.id, str(exc))
            )
            continue
        if result.status == "no_text":
            warning = _hwpx_ocr_empty_warning(image_index + 1, asset.id)
            if isinstance(raw_result, OcrResult):
                warning["reason"] = result.reason
            warnings.append(warning)
            continue
        if result.status == "uncertain":
            warnings.append(
                _hwpx_ocr_uncertain_warning(
                    image_index + 1,
                    asset.id,
                    result.reason,
                )
            )
            continue
        text = _clean_text(result.text)
        if not text:
            warnings.append(_hwpx_ocr_empty_warning(image_index + 1, asset.id))
            continue
        units.append(_text_unit(f"b{block_index}", text))
        block_index += 1
    return block_index


def _hwpx_ocr_failed_warning(
    image_number: int,
    asset_id: str,
    message: str,
) -> dict[str, Any]:
    return {
        "type": "hwpx_ocr_failed",
        "severity": "medium",
        "image_index": image_number,
        "asset_id": asset_id,
        "message": message,
    }


def _hwpx_ocr_empty_warning(
    image_number: int,
    asset_id: str,
) -> dict[str, Any]:
    return {
        "type": "hwpx_ocr_empty",
        "severity": "low",
        "image_index": image_number,
        "asset_id": asset_id,
        "message": "empty OCR result",
    }


def _hwpx_ocr_uncertain_warning(
    image_number: int,
    asset_id: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "type": "hwpx_ocr_uncertain",
        "severity": "medium",
        "image_index": image_number,
        "asset_id": asset_id,
        "reason": reason,
        "message": "OCR could not determine whether the image contains readable text.",
    }


def _has_native_source_text(units: list[EvidenceUnit]) -> bool:
    return any(
        unit.type in {"text", "table", "diagram"} and unit.source.text.strip()
        for unit in units
    )
