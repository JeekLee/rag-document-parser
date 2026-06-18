from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from rag_document_parser import Hwp5Backend, S3Config
from rag_document_parser.backends import ParsedDocument
from rag_document_parser.renderer.evidence_unit_render import render_evidence_units_html
from rag_document_parser.storage import public_url_for_s3_uri, put_object

_OLE_COMPOUND_HEADER = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


def main() -> None:
    args = _parse_args()
    raw = args.input.read_bytes()
    document_sha256 = hashlib.sha256(raw).hexdigest()
    run_id = _timestamped_run_id(args.run_id)
    prefix = f"{args.s3_prefix.strip('/')}/{run_id}".strip("/")
    storage = S3Config(
        endpoint=args.s3_endpoint,
        bucket=args.s3_bucket,
        access_key=args.s3_access_key,
        secret_key=args.s3_secret_key,
        prefix=prefix,
        region=args.s3_region,
    )

    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    evidence_started = time.perf_counter()
    parsed, skip_reason = _parse_hwp5_or_skip(raw)
    evidence_elapsed = time.perf_counter() - evidence_started
    public_asset_endpoint = None
    if args.html_asset_url_mode == "public":
        public_asset_endpoint = args.public_asset_endpoint or args.s3_endpoint
    uploaded_assets = _upload_assets(
        storage,
        parsed,
        document_sha256,
        public_endpoint=public_asset_endpoint,
    )

    unit_dicts = [unit.to_dict() for unit in parsed.units]
    evidence_payload = {
        "source": {
            "name": args.source_name or args.input.name,
            "suffix": ".hwp",
            "sha256": document_sha256,
            "bytes": len(raw),
        },
        "units": unit_dicts,
        "assets": uploaded_assets,
        "quality_warnings": parsed.quality_warnings,
    }
    evidence_json = _json_bytes(evidence_payload)
    evidence_html = render_evidence_units_html(
        unit_dicts,
        title=args.source_name or args.input.name,
        assets=uploaded_assets,
    ).encode("utf-8")

    evidence_json_path = output_dir / "evidence-units.json"
    evidence_html_path = output_dir / "evidence-units.html"
    evidence_json_path.write_bytes(evidence_json)
    evidence_html_path.write_bytes(evidence_html)

    uploads = {
        "evidence_json": put_object(
            storage,
            "evidence-units.json",
            evidence_json,
            "application/json; charset=utf-8",
        ),
        "evidence_html": put_object(
            storage,
            "evidence-units.html",
            evidence_html,
            "text/html; charset=utf-8",
        ),
    }

    metrics = _metrics(
        parsed=parsed,
        raw_bytes=len(raw),
        document_sha256=document_sha256,
        evidence_elapsed=evidence_elapsed,
        uploads=uploads,
        uploaded_assets=uploaded_assets,
        skip_reason=skip_reason,
    )
    metrics_json = _json_bytes(metrics)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_bytes(metrics_json)
    uploads["metrics_json"] = put_object(
        storage,
        "metrics.json",
        metrics_json,
        "application/json; charset=utf-8",
    )
    metrics["uploads"] = uploads
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def _metrics(
    *,
    parsed: ParsedDocument,
    raw_bytes: int,
    document_sha256: str,
    evidence_elapsed: float,
    uploads: dict[str, str],
    uploaded_assets: list[dict[str, Any]],
    skip_reason: str | None = None,
) -> dict[str, Any]:
    unit_dicts = [unit.to_dict() for unit in parsed.units]
    counts = Counter(unit.type for unit in parsed.units)
    warning_counts = Counter(
        str(warning.get("type"))
        for warning in parsed.quality_warnings
        if warning.get("type")
    )
    metrics = {
        "source": {
            "sha256": document_sha256,
            "bytes": raw_bytes,
        },
        "evidence_units": {
            "total": len(parsed.units),
            "by_type": dict(sorted(counts.items())),
            "asset_refs": _count_asset_refs(unit_dicts),
            "elapsed_seconds": round(evidence_elapsed, 3),
        },
        "tables": _table_metrics(unit_dicts),
        "diagrams": _diagram_metrics(unit_dicts),
        "assets": {
            "total": len(parsed.assets),
            "bytes": sum(len(asset.data) for asset in parsed.assets),
            "by_mime": dict(sorted(Counter(asset.mime for asset in parsed.assets).items())),
            "uris": [asset["uri"] for asset in uploaded_assets],
            "public_urls": [
                asset["public_url"]
                for asset in uploaded_assets
                if isinstance(asset.get("public_url"), str)
            ],
        },
        "quality_warnings": {
            "total": len(parsed.quality_warnings),
            "by_type": dict(sorted(warning_counts.items())),
        },
        "uploads": dict(uploads),
    }
    if skip_reason is not None:
        metrics["skipped"] = True
        metrics["skip_reason"] = skip_reason
    return metrics


def _parse_hwp5_or_skip(raw: bytes) -> tuple[ParsedDocument, str | None]:
    if not _has_hwp5_container_signature(raw):
        return (
            ParsedDocument(
                units=[],
                quality_warnings=[
                    {
                        "type": "non_hwp5_skipped",
                        "severity": "low",
                        "stage": "hwp5_validation",
                        "message": "Input does not have an HWP5/OLE container signature.",
                    }
                ],
            ),
            "non_hwp5_signature",
        )
    return Hwp5Backend().parse(raw, ".hwp"), None


def _has_hwp5_container_signature(raw: bytes) -> bool:
    return raw.startswith(_OLE_COMPOUND_HEADER)


def _table_metrics(units: list[dict[str, Any]]) -> dict[str, int]:
    tables = [
        _unit_content(unit)
        for unit in units
        if unit.get("type") == "table"
    ]
    return {
        "total": len(tables),
        "max_columns": max(
            (len(table.get("columns", [])) for table in tables),
            default=0,
        ),
        "total_rows": sum(
            len(table.get("header_rows", [])) + len(table.get("rows", []))
            for table in tables
        ),
    }


def _diagram_metrics(units: list[dict[str, Any]]) -> dict[str, int]:
    diagrams = [
        _unit_content(unit)
        for unit in units
        if unit.get("type") == "diagram"
    ]
    return {
        "total": len(diagrams),
        "nodes": sum(len(diagram.get("nodes", [])) for diagram in diagrams),
        "connectors": sum(
            len(diagram.get("connectors", []))
            for diagram in diagrams
        ),
        "edges": sum(len(diagram.get("edges", [])) for diagram in diagrams),
        "labeled_edges": sum(
            1
            for diagram in diagrams
            for edge in diagram.get("edges", [])
            if isinstance(edge, dict) and str(edge.get("label", "")).strip()
        ),
    }


def _unit_content(unit: dict[str, Any]) -> dict[str, Any]:
    content = unit.get("content")
    if isinstance(content, dict):
        return content
    legacy = unit.get("evidence", {})
    if isinstance(legacy, dict):
        legacy_content = legacy.get("content")
        if isinstance(legacy_content, dict):
            return legacy_content
    return {}


def _upload_assets(
    storage: S3Config,
    parsed: ParsedDocument,
    document_sha256: str,
    *,
    public_endpoint: str | None = None,
) -> list[dict[str, Any]]:
    uploaded = []
    for asset in parsed.assets:
        ext = asset.ext.lstrip(".")
        key = f"{document_sha256}/assets/{asset.id}.{ext}"
        uri = put_object(storage, key, asset.data, asset.mime)
        asset_payload: dict[str, Any] = {
            "id": asset.id,
            "kind": asset.kind,
            "uri": uri,
            "mime": asset.mime,
            "ext": ext,
            "sha256": hashlib.sha256(asset.data).hexdigest(),
            "bytes": len(asset.data),
            "metadata": dict(asset.metadata),
        }
        if public_endpoint:
            asset_payload["public_url"] = public_url_for_s3_uri(uri, public_endpoint)
        uploaded.append(asset_payload)
    return uploaded


def _count_asset_refs(value: Any) -> int:
    if isinstance(value, list):
        return sum(_count_asset_refs(item) for item in value)
    if not isinstance(value, dict):
        return 0
    count = 1 if value.get("format") == "asset_ref" else 0
    return count + sum(_count_asset_refs(item) for item in value.values())


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def _timestamped_run_id(run_id: str | None, *, now: datetime | None = None) -> str:
    name = (run_id or "hwp5-validation").strip()
    if re.match(r"^\d{8}-\d{6}-", name):
        return name
    timestamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{name}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate HWP5 EvidenceUnit extraction against clic MinIO.",
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--source-name")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/rag-document-parser-validation"),
    )
    parser.add_argument(
        "--s3-endpoint",
        default=os.getenv("RDP_S3_ENDPOINT", "http://localhost:10190"),
    )
    parser.add_argument(
        "--s3-bucket",
        default=os.getenv("RDP_S3_BUCKET", "rag-document-parser-test"),
    )
    parser.add_argument(
        "--s3-access-key",
        default=os.getenv("RDP_S3_ACCESS_KEY", "minioadmin"),
    )
    parser.add_argument(
        "--s3-secret-key",
        default=os.getenv("RDP_S3_SECRET_KEY", "minioadmin"),
    )
    parser.add_argument("--s3-region", default=os.getenv("RDP_S3_REGION", "us-east-1"))
    parser.add_argument(
        "--s3-prefix",
        default=os.getenv("RDP_S3_PREFIX", "rag-document-parser-results"),
    )
    parser.add_argument(
        "--html-asset-url-mode",
        choices=("public", "s3"),
        default=os.getenv("RDP_HTML_ASSET_URL_MODE", "public"),
        help="Use public HTTP asset URLs in generated HTML, or keep raw s3:// URIs.",
    )
    parser.add_argument(
        "--public-asset-endpoint",
        default=os.getenv("RDP_PUBLIC_ASSET_ENDPOINT"),
        help="Browser-reachable MinIO/S3 API endpoint used for public HTML asset URLs.",
    )
    args = parser.parse_args()
    if not args.input.exists():
        parser.error(f"input does not exist: {args.input}")
    return args


if __name__ == "__main__":
    main()
