from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path


def test_hwp5_validation_run_id_gets_datetime_prefix():
    validate_hwp5_clic_minio = _load_validation_script()

    run_id = validate_hwp5_clic_minio._timestamped_run_id(
        "hwp5-renderer",
        now=datetime(2026, 6, 18, 13, 40, 3),
    )

    assert run_id == "20260618-134003-hwp5-renderer"


def test_hwp5_validation_metrics_include_tables_diagrams_and_warnings():
    from rag_document_parser.evidence_unit_extraction.backend import ParsedDocument
    from rag_document_parser.models import EvidenceUnit, SourceEvidence

    validate_hwp5_clic_minio = _load_validation_script()
    parsed = ParsedDocument(
        units=[
            EvidenceUnit(
                id="b1",
                type="diagram",
                format="structured_diagram",
                source=SourceEvidence(kind="diagram", text="업무처리 흐름도"),
                content={
                    "nodes": [
                        {"id": "n1", "shape_type": "rectangle", "text": "A"},
                        {"id": "n2", "shape_type": "rectangle", "text": "B"},
                    ],
                    "edges": [
                        {
                            "from": "n1",
                            "to": "n2",
                            "type": "arrow",
                            "label": "①신청",
                            "confidence": "inferred_geometry",
                            "connector_id": "c1",
                        }
                    ],
                    "connectors": [{"id": "c1", "type": "arrow"}],
                    "mermaid": None,
                },
            ),
            EvidenceUnit(
                id="b2",
                type="table",
                format="structured_table",
                source=SourceEvidence(kind="table", text="table: 1 columns"),
                content={
                    "columns": [{"id": "c1", "text": "항목"}],
                    "header_rows": [],
                    "rows": [{"index": 1, "cells": []}],
                },
            ),
        ],
        quality_warnings=[
            {
                "type": "hwp5_drawing_structure_partial",
                "severity": "medium",
                "message": "partial",
            }
        ],
    )

    metrics = validate_hwp5_clic_minio._metrics(
        parsed=parsed,
        raw_bytes=123,
        document_sha256="abc",
        evidence_elapsed=0.25,
        uploads={"evidence_html": "s3://bucket/run/evidence-units.html"},
        uploaded_assets=[],
    )

    assert metrics["source"] == {"sha256": "abc", "bytes": 123}
    assert metrics["evidence_units"]["by_type"] == {"diagram": 1, "table": 1}
    assert metrics["tables"] == {"total": 1, "max_columns": 1, "total_rows": 1}
    assert metrics["diagrams"] == {
        "total": 1,
        "nodes": 2,
        "connectors": 1,
        "edges": 1,
        "labeled_edges": 1,
    }
    assert metrics["quality_warnings"]["by_type"] == {
        "hwp5_drawing_structure_partial": 1
    }


def test_hwp5_validation_skips_non_hwp5_signature_without_parsing(monkeypatch):
    validate_hwp5_clic_minio = _load_validation_script()

    class FailingBackend:
        def parse(self, data, suffix):
            raise AssertionError("parser should not be called")

    monkeypatch.setattr(validate_hwp5_clic_minio, "Hwp5Backend", FailingBackend)

    parsed, skip_reason = validate_hwp5_clic_minio._parse_hwp5_or_skip(
        b"PK\x03\x04not-hwp5",
    )

    assert parsed.units == []
    assert parsed.assets == []
    assert skip_reason == "non_hwp5_signature"
    assert parsed.quality_warnings == [
        {
            "type": "non_hwp5_skipped",
            "severity": "low",
            "stage": "hwp5_validation",
            "message": "Input does not have an HWP5/OLE container signature.",
        }
    ]

    metrics = validate_hwp5_clic_minio._metrics(
        parsed=parsed,
        raw_bytes=12,
        document_sha256="abc",
        evidence_elapsed=0.0,
        uploads={},
        uploaded_assets=[],
        skip_reason=skip_reason,
    )

    assert metrics["skipped"] is True
    assert metrics["skip_reason"] == "non_hwp5_signature"
    assert metrics["quality_warnings"]["by_type"] == {"non_hwp5_skipped": 1}


def _load_validation_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "validate_hwp5_clic_minio.py"
    spec = importlib.util.spec_from_file_location("validate_hwp5_clic_minio", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
