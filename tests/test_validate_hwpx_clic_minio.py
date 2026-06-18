from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path


def test_upload_assets_adds_public_url_when_public_endpoint_is_configured(monkeypatch):
    from rag_document_parser.evidence_unit_extraction.backend import ParsedDocument
    from rag_document_parser.models import EvidenceUnit, PendingAsset, SourceEvidence
    from rag_document_parser.storage import S3Config

    validate_hwpx_clic_minio = _load_validation_script()

    def fake_put_object(cfg, key, data, content_type):
        assert key == "doc-sha/assets/img-0001.png"
        assert content_type == "image/png"
        return "s3://rag-assets/validation/run/doc-sha/assets/img-0001.png"

    monkeypatch.setattr(validate_hwpx_clic_minio, "put_object", fake_put_object)

    uploaded = validate_hwpx_clic_minio._upload_assets(
        S3Config(
            endpoint="http://localhost:10190",
            bucket="rag-assets",
            access_key="minioadmin",
            secret_key="minioadmin",
            prefix="validation/run",
        ),
        ParsedDocument(
            units=[
                EvidenceUnit(
                    id="u1",
                    type="image",
                    format="asset_ref",
                    source=SourceEvidence(kind="image", text="image: img-0001"),
                    content={"asset_id": "img-0001"},
                )
            ],
            assets=[
                PendingAsset(
                    id="img-0001",
                    kind="image",
                    data=b"png",
                    mime="image/png",
                    ext="png",
                )
            ],
        ),
        "doc-sha",
        public_endpoint="http://203.0.113.10:10190",
    )

    assert uploaded[0]["uri"] == "s3://rag-assets/validation/run/doc-sha/assets/img-0001.png"
    assert (
        uploaded[0]["public_url"]
        == "http://203.0.113.10:10190/rag-assets/validation/run/doc-sha/assets/img-0001.png"
    )


def test_validation_run_id_gets_datetime_prefix():
    validate_hwpx_clic_minio = _load_validation_script()

    run_id = validate_hwpx_clic_minio._timestamped_run_id(
        "grid-fidelity",
        now=datetime(2026, 6, 17, 16, 42, 3),
    )

    assert run_id == "20260617-164203-grid-fidelity"


def test_validation_run_id_does_not_double_prefix():
    validate_hwpx_clic_minio = _load_validation_script()

    run_id = validate_hwpx_clic_minio._timestamped_run_id(
        "20260617-164203-grid-fidelity",
        now=datetime(2026, 6, 17, 17, 0, 0),
    )

    assert run_id == "20260617-164203-grid-fidelity"


def test_validation_run_id_default_is_timestamped():
    validate_hwpx_clic_minio = _load_validation_script()

    run_id = validate_hwpx_clic_minio._timestamped_run_id(
        None,
        now=datetime(2026, 6, 17, 16, 42, 3),
    )

    assert run_id == "20260617-164203-validation"


def _load_validation_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "validate_hwpx_clic_minio.py"
    spec = importlib.util.spec_from_file_location("validate_hwpx_clic_minio", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
