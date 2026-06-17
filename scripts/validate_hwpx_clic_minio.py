from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_document_parser import HwpxBackend, LlmConfig, RagDocumentParser, S3Config
from rag_document_parser.backends import ParsedDocument
from rag_document_parser.evidence_html import render_evidence_units_html
from rag_document_parser.storage import put_object


@dataclass
class LimitedBackend:
    max_units: int | None = None

    def parse(self, data: bytes, suffix: str) -> ParsedDocument:
        parsed = HwpxBackend().parse(data, suffix)
        if self.max_units is None:
            return parsed
        return ParsedDocument(
            units=parsed.units[: self.max_units],
            assets=parsed.assets,
            quality_warnings=parsed.quality_warnings,
        )


def main() -> None:
    args = _parse_args()
    raw = args.input.read_bytes()
    document_sha256 = hashlib.sha256(raw).hexdigest()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
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
    parsed = HwpxBackend().parse(raw, ".hwpx")
    evidence_elapsed = time.perf_counter() - evidence_started
    uploaded_assets = _upload_assets(storage, parsed, document_sha256)

    unit_dicts = [unit.to_dict() for unit in parsed.units]
    evidence_payload = {
        "source": {
            "name": args.source_name or args.input.name,
            "suffix": ".hwpx",
            "sha256": document_sha256,
            "bytes": len(raw),
        },
        "units": unit_dicts,
        "assets": uploaded_assets,
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

    llm_payload: dict[str, Any] | None = None
    llm_elapsed = None
    if not args.skip_llm:
        llm_started = time.perf_counter()
        result = RagDocumentParser(
            llm=LlmConfig(
                url=args.llm_url,
                api_key=args.llm_api_key,
                model=args.llm_model,
                timeout=args.llm_timeout,
            ),
            object_storage=storage,
            backends={".hwpx": LimitedBackend(args.llm_max_units)},
        ).parse(
            raw,
            suffix=".hwpx",
            source_name=args.source_name or args.input.name,
        )
        llm_elapsed = time.perf_counter() - llm_started
        llm_payload = result.to_dict()
        llm_json = _json_bytes(llm_payload)
        llm_json_path = output_dir / "parse-result.llm.json"
        llm_json_path.write_bytes(llm_json)
        uploads["llm_parse_result_json"] = put_object(
            storage,
            "parse-result.llm.json",
            llm_json,
            "application/json; charset=utf-8",
        )

    metrics = _metrics(
        parsed=parsed,
        raw_bytes=len(raw),
        document_sha256=document_sha256,
        evidence_elapsed=evidence_elapsed,
        llm_elapsed=llm_elapsed,
        llm_payload=llm_payload,
        uploads=uploads,
        llm_max_units=args.llm_max_units,
        uploaded_assets=uploaded_assets,
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
    llm_elapsed: float | None,
    llm_payload: dict[str, Any] | None,
    uploads: dict[str, str],
    llm_max_units: int | None,
    uploaded_assets: list[dict[str, Any]],
) -> dict[str, Any]:
    counts = Counter(unit.type for unit in parsed.units)
    return {
        "source": {
            "sha256": document_sha256,
            "bytes": raw_bytes,
        },
        "evidence_units": {
            "total": len(parsed.units),
            "by_type": dict(sorted(counts.items())),
            "asset_refs": _count_asset_refs([unit.to_dict() for unit in parsed.units]),
            "elapsed_seconds": round(evidence_elapsed, 3),
        },
        "assets": {
            "total": len(parsed.assets),
            "bytes": sum(len(asset.data) for asset in parsed.assets),
            "by_mime": dict(sorted(Counter(asset.mime for asset in parsed.assets).items())),
            "uris": [asset["uri"] for asset in uploaded_assets],
        },
        "llm": None
        if llm_payload is None
        else {
            "max_units": llm_max_units,
            "chunks": len(llm_payload["chunks"]),
            "elapsed_seconds": round(llm_elapsed or 0.0, 3),
            "sample_chunk": llm_payload["chunks"][0] if llm_payload["chunks"] else None,
        },
        "uploads": dict(uploads),
    }


def _upload_assets(
    storage: S3Config,
    parsed: ParsedDocument,
    document_sha256: str,
) -> list[dict[str, Any]]:
    uploaded = []
    for asset in parsed.assets:
        ext = asset.ext.lstrip(".")
        key = f"{document_sha256}/assets/{asset.id}.{ext}"
        uri = put_object(storage, key, asset.data, asset.mime)
        uploaded.append(
            {
                "id": asset.id,
                "kind": asset.kind,
                "uri": uri,
                "mime": asset.mime,
                "ext": ext,
                "sha256": hashlib.sha256(asset.data).hexdigest(),
                "bytes": len(asset.data),
                "metadata": dict(asset.metadata),
            }
        )
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate HWPX EvidenceUnit extraction against clic MinIO.",
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--source-name")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/rag-document-parser-validation"),
    )
    parser.add_argument("--s3-endpoint", default=os.getenv("RDP_S3_ENDPOINT", "http://localhost:10190"))
    parser.add_argument("--s3-bucket", default=os.getenv("RDP_S3_BUCKET", "rag-document-parser-test"))
    parser.add_argument("--s3-access-key", default=os.getenv("RDP_S3_ACCESS_KEY", "minioadmin"))
    parser.add_argument("--s3-secret-key", default=os.getenv("RDP_S3_SECRET_KEY", "minioadmin"))
    parser.add_argument("--s3-region", default=os.getenv("RDP_S3_REGION", "us-east-1"))
    parser.add_argument("--s3-prefix", default=os.getenv("RDP_S3_PREFIX", "hwpx-validation"))
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--llm-url", default=os.getenv("RDP_LLM_URL", "http://localhost:10080/v1"))
    parser.add_argument("--llm-api-key", default=os.getenv("RDP_LLM_API_KEY", ""))
    parser.add_argument("--llm-model", default=os.getenv("RDP_LLM_MODEL", "qwen3-vl-30b-a3b"))
    parser.add_argument("--llm-timeout", type=float, default=float(os.getenv("RDP_LLM_TIMEOUT", "120")))
    parser.add_argument("--llm-max-units", type=int)
    args = parser.parse_args()
    if not args.input.exists():
        parser.error(f"input does not exist: {args.input}")
    if not args.skip_llm and not args.llm_api_key:
        parser.error("--llm-api-key or RDP_LLM_API_KEY is required unless --skip-llm is set")
    return args


if __name__ == "__main__":
    main()
