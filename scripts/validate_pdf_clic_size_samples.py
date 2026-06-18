from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from rag_document_parser import LlmConfig, PdfBackend, S3Config
from rag_document_parser.evidence_unit_extraction.backend import ParsedDocument
from rag_document_parser.renderer.evidence_unit_render import render_evidence_units_html
from rag_document_parser.storage import public_url_for_s3_uri, put_object


_BANDS: tuple[tuple[str, int, int], ...] = (
    ("small", 0, 100 * 1024),
    ("medium", 100 * 1024, 512 * 1024),
    ("large", 512 * 1024, 2 * 1024 * 1024),
    ("xlarge", 2 * 1024 * 1024, 20 * 1024 * 1024),
)
_QUANTILES = (0.1, 0.3, 0.5, 0.7, 0.9)
_GENERIC_PDF_NAMES = {"본문내용.pdf", "본문.pdf"}


@dataclass(frozen=True)
class MinioObject:
    key: str
    size: int
    last_modified: str | None = None
    etag: str | None = None


@dataclass(frozen=True)
class Sample:
    id: str
    band: str
    band_index: int
    object: MinioObject
    fixture_path: Path


def main() -> None:
    args = _parse_args()
    if args.worker_input is not None:
        _worker_main(args)
        return

    run_id = _timestamped_run_id(args.run_id)
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    args.fixture_dir.mkdir(parents=True, exist_ok=True)

    pdfs = _list_pdfs(args)
    samples, outliers = _select_samples(pdfs, args.fixture_dir)
    _download_samples(args, samples)
    _write_manifest(args, samples, outliers, pdf_count=len(pdfs))

    storage = S3Config(
        endpoint=args.s3_endpoint,
        bucket=args.s3_bucket,
        access_key=args.s3_access_key,
        secret_key=args.s3_secret_key,
        prefix=f"{args.s3_prefix.strip('/')}/{run_id}".strip("/"),
        region=args.s3_region,
    )
    results: list[dict[str, Any]] = []
    for sample in samples:
        result = _validate_sample_with_subprocess(
            sample,
            storage,
            output_dir,
            args,
            public_asset_endpoint=args.public_asset_endpoint or args.s3_endpoint,
        )
        results.append(result)

    summary = _summary_payload(
        run_id=run_id,
        pdf_count=len(pdfs),
        samples=samples,
        outliers=outliers,
        results=results,
        storage=storage,
        public_endpoint=args.public_asset_endpoint or args.s3_endpoint,
    )
    summary_json = _json_bytes(summary)
    summary_html = _summary_html(summary).encode("utf-8")
    (output_dir / "index.json").write_bytes(summary_json)
    (output_dir / "index.html").write_bytes(summary_html)
    summary["uploads"] = {
        "index_json": put_object(
            storage,
            "index.json",
            summary_json,
            "application/json; charset=utf-8",
        ),
        "index_html": put_object(
            storage,
            "index.html",
            summary_html,
            "text/html; charset=utf-8",
        ),
    }
    (output_dir / "index.json").write_bytes(_json_bytes(summary))
    put_object(storage, "index.json", _json_bytes(summary), "application/json; charset=utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _worker_main(args: argparse.Namespace) -> None:
    if args.worker_input is None or args.worker_output is None:
        raise RuntimeError("worker mode requires --worker-input and --worker-output")
    payload = json.loads(args.worker_input.read_text(encoding="utf-8"))
    obj = MinioObject(**payload["object"])
    sample = Sample(
        id=payload["id"],
        band=payload["band"],
        band_index=int(payload["band_index"]),
        object=obj,
        fixture_path=Path(payload["fixture_path"]),
    )
    storage = S3Config(**payload["storage"])
    backend = PdfBackend(
        ocr_llm=_ocr_config(args),
        max_ocr_workers=args.max_ocr_workers,
    )
    result = _validate_sample(
        sample,
        backend,
        storage,
        args.output_dir,
        public_asset_endpoint=payload["public_asset_endpoint"],
        per_document_timeout=0,
    )
    args.worker_output.write_bytes(_json_bytes(result))


def _list_pdfs(args: argparse.Namespace) -> list[MinioObject]:
    proc = subprocess.run(
        [
            "docker",
            "exec",
            args.minio_container,
            args.mc_binary,
            "ls",
            "--json",
            "--recursive",
            args.minio_source,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    pdfs: list[MinioObject] = []
    for line in proc.stdout.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = str(item.get("key", ""))
        if item.get("type") != "file" or not key.lower().endswith(".pdf"):
            continue
        pdfs.append(
            MinioObject(
                key=key,
                size=int(item.get("size") or 0),
                last_modified=item.get("lastModified"),
                etag=item.get("etag"),
            )
        )
    return sorted(pdfs, key=lambda obj: obj.size)


def _select_samples(
    pdfs: list[MinioObject],
    fixture_dir: Path,
) -> tuple[list[Sample], list[dict[str, Any]]]:
    samples: list[Sample] = []
    for band, lower, upper in _BANDS:
        band_objects = [
            obj for obj in pdfs if lower < obj.size <= upper
        ]
        if len(band_objects) < len(_QUANTILES):
            raise RuntimeError(
                f"not enough PDFs for {band}: {len(band_objects)}"
            )
        used: set[str] = set()
        for index, quantile in enumerate(_QUANTILES, start=1):
            selected = _select_near_quantile(band_objects, quantile, used)
            used.add(selected.key)
            sample_id = f"{band}-{index:02d}-{selected.key.split('/', 1)[0]}"
            samples.append(
                Sample(
                    id=_safe_id(sample_id),
                    band=band,
                    band_index=index,
                    object=selected,
                    fixture_path=fixture_dir / band / f"{_safe_id(sample_id)}.pdf",
                )
            )
    outliers = [
        {
            "key": obj.key,
            "bytes": obj.size,
        }
        for obj in pdfs
        if obj.size > _BANDS[-1][2]
    ]
    return samples, outliers


def _select_near_quantile(
    objects: list[MinioObject],
    quantile: float,
    used: set[str],
) -> MinioObject:
    index = round((len(objects) - 1) * quantile)
    window = range(max(0, index - 15), min(len(objects), index + 16))

    def score(candidate_index: int) -> tuple[bool, int, str]:
        obj = objects[candidate_index]
        name = Path(obj.key).name
        generic = name in _GENERIC_PDF_NAMES or name.startswith("본문")
        return (generic, abs(candidate_index - index), obj.key)

    for candidate_index in sorted(window, key=score):
        candidate = objects[candidate_index]
        if candidate.key not in used:
            return candidate
    for obj in objects:
        if obj.key not in used:
            return obj
    raise RuntimeError("no unused candidate")


def _download_samples(args: argparse.Namespace, samples: list[Sample]) -> None:
    for sample in samples:
        sample.fixture_path.parent.mkdir(parents=True, exist_ok=True)
        if sample.fixture_path.is_file():
            existing = sample.fixture_path.read_bytes()
            if len(existing) == sample.object.size:
                continue
        source = f"{args.minio_source.rstrip('/')}/{sample.object.key}"
        proc = subprocess.run(
            [
                "docker",
                "exec",
                args.minio_container,
                args.mc_binary,
                "cat",
                source,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        sample.fixture_path.write_bytes(proc.stdout)
        if sample.fixture_path.stat().st_size != sample.object.size:
            raise RuntimeError(
                f"download size mismatch for {sample.id}: "
                f"{sample.fixture_path.stat().st_size} != {sample.object.size}"
            )


def _write_manifest(
    args: argparse.Namespace,
    samples: list[Sample],
    outliers: list[dict[str, Any]],
    *,
    pdf_count: int,
) -> None:
    payload = {
        "source": {
            "minio_source": args.minio_source,
            "pdf_count": pdf_count,
            "size_bands": [
                {"id": band, "min_exclusive": lower, "max_inclusive": upper}
                for band, lower, upper in _BANDS
            ],
            "selection": {
                "strategy": "size-sorted quantiles with non-generic filename preference",
                "quantiles": list(_QUANTILES),
            },
            "outliers_above_largest_band": outliers,
        },
        "documents": [
            {
                "id": sample.id,
                "band": sample.band,
                "band_index": sample.band_index,
                "path": str(sample.fixture_path.relative_to(args.fixture_dir)),
                "minio_key": sample.object.key,
                "bytes": sample.fixture_path.stat().st_size,
                "sha256": hashlib.sha256(sample.fixture_path.read_bytes()).hexdigest(),
                "etag": sample.object.etag,
                "last_modified": sample.object.last_modified,
            }
            for sample in samples
        ],
    }
    (args.fixture_dir / "manifest.json").write_bytes(_json_bytes(payload))


def _validate_sample(
    sample: Sample,
    backend: PdfBackend,
    storage: S3Config,
    output_dir: Path,
    *,
    public_asset_endpoint: str,
    per_document_timeout: int,
) -> dict[str, Any]:
    doc_output_dir = output_dir / sample.id
    doc_output_dir.mkdir(parents=True, exist_ok=True)
    raw = sample.fixture_path.read_bytes()
    started = time.perf_counter()
    error: str | None = None
    parsed: ParsedDocument | None = None
    try:
        with _document_timeout(per_document_timeout):
            parsed = backend.parse(raw, ".pdf")
    except TimeoutError as exc:
        error = str(exc)
    except Exception as exc:
        error = str(exc)
    elapsed = time.perf_counter() - started

    source_uri = put_object(storage, f"{sample.id}/source.pdf", raw, "application/pdf")
    source_url = public_url_for_s3_uri(source_uri, public_asset_endpoint)
    if parsed is None:
        result = {
            "id": sample.id,
            "band": sample.band,
            "minio_key": sample.object.key,
            "bytes": len(raw),
            "elapsed_seconds": round(elapsed, 3),
            "status": "timeout" if isinstance(error, str) and "timed out" in error else "parse_error",
            "error": error,
            "uploads": {"source": source_uri, "source_url": source_url},
        }
        (doc_output_dir / "metrics.json").write_bytes(_json_bytes(result))
        put_object(
            storage,
            f"{sample.id}/metrics.json",
            _json_bytes(result),
            "application/json; charset=utf-8",
        )
        return result

    uploaded_assets = _upload_assets(
        storage,
        sample.id,
        parsed,
        public_endpoint=public_asset_endpoint,
    )
    unit_dicts = [unit.to_dict() for unit in parsed.units]
    evidence_payload = {
        "source": {
            "name": Path(sample.object.key).name,
            "minio_key": sample.object.key,
            "suffix": ".pdf",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "size_band": sample.band,
        },
        "units": unit_dicts,
        "assets": uploaded_assets,
        "quality_warnings": parsed.quality_warnings,
    }
    evidence_json = _json_bytes(evidence_payload)
    evidence_html = render_evidence_units_html(
        unit_dicts,
        title=f"{sample.id} - {Path(sample.object.key).name}",
        assets=uploaded_assets,
    ).encode("utf-8")
    (doc_output_dir / "evidence-units.json").write_bytes(evidence_json)
    (doc_output_dir / "evidence-units.html").write_bytes(evidence_html)
    uploads = {
        "source": source_uri,
        "source_url": source_url,
        "evidence_json": put_object(
            storage,
            f"{sample.id}/evidence-units.json",
            evidence_json,
            "application/json; charset=utf-8",
        ),
        "evidence_html": put_object(
            storage,
            f"{sample.id}/evidence-units.html",
            evidence_html,
            "text/html; charset=utf-8",
        ),
    }
    result = {
        "id": sample.id,
        "band": sample.band,
        "minio_key": sample.object.key,
        "bytes": len(raw),
        "elapsed_seconds": round(elapsed, 3),
        "status": "ok",
        "quality": _quality_metrics(parsed, unit_dicts),
        "uploads": {
            **uploads,
            "evidence_html_url": public_url_for_s3_uri(
                uploads["evidence_html"],
                public_asset_endpoint,
            ),
            "evidence_json_url": public_url_for_s3_uri(
                uploads["evidence_json"],
                public_asset_endpoint,
            ),
        },
    }
    result["quality"]["review_flags"] = _review_flags(result["quality"])
    metrics_json = _json_bytes(result)
    (doc_output_dir / "metrics.json").write_bytes(metrics_json)
    result["uploads"]["metrics_json"] = put_object(
        storage,
        f"{sample.id}/metrics.json",
        metrics_json,
        "application/json; charset=utf-8",
    )
    result["uploads"]["metrics_json_url"] = public_url_for_s3_uri(
        result["uploads"]["metrics_json"],
        public_asset_endpoint,
    )
    return result


def _validate_sample_with_subprocess(
    sample: Sample,
    storage: S3Config,
    output_dir: Path,
    args: argparse.Namespace,
    *,
    public_asset_endpoint: str,
) -> dict[str, Any]:
    doc_output_dir = output_dir / sample.id
    doc_output_dir.mkdir(parents=True, exist_ok=True)
    worker_input = doc_output_dir / "worker-input.json"
    worker_output = doc_output_dir / "worker-output.json"
    worker_input.write_bytes(
        _json_bytes(
            {
                "id": sample.id,
                "band": sample.band,
                "band_index": sample.band_index,
                "object": {
                    "key": sample.object.key,
                    "size": sample.object.size,
                    "last_modified": sample.object.last_modified,
                    "etag": sample.object.etag,
                },
                "fixture_path": str(sample.fixture_path),
                "storage": {
                    "endpoint": storage.endpoint,
                    "bucket": storage.bucket,
                    "access_key": storage.access_key,
                    "secret_key": storage.secret_key,
                    "prefix": storage.prefix,
                    "region": storage.region,
                },
                "public_asset_endpoint": public_asset_endpoint,
            }
        )
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-input",
        str(worker_input),
        "--worker-output",
        str(worker_output),
        "--output-dir",
        str(output_dir),
        "--max-ocr-workers",
        str(args.max_ocr_workers),
        "--ocr-base-url",
        args.ocr_base_url,
        "--ocr-model",
        args.ocr_model,
        "--ocr-timeout",
        str(args.ocr_timeout),
    ]
    env = os.environ.copy()
    if args.ocr_api_key:
        env["RDP_PDF_OCR_API_KEY"] = args.ocr_api_key
    try:
        subprocess.run(
            command,
            check=True,
            timeout=args.per_document_timeout if args.per_document_timeout > 0 else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return _timeout_result(
            sample,
            storage,
            doc_output_dir,
            public_asset_endpoint=public_asset_endpoint,
            seconds=args.per_document_timeout,
        )
    except subprocess.CalledProcessError as exc:
        return _worker_error_result(
            sample,
            storage,
            doc_output_dir,
            public_asset_endpoint=public_asset_endpoint,
            message=exc.stderr.decode("utf-8", errors="replace") if exc.stderr else str(exc),
        )
    if not worker_output.is_file():
        return _worker_error_result(
            sample,
            storage,
            doc_output_dir,
            public_asset_endpoint=public_asset_endpoint,
            message="worker completed without output",
        )
    return json.loads(worker_output.read_text(encoding="utf-8"))


def _timeout_result(
    sample: Sample,
    storage: S3Config,
    doc_output_dir: Path,
    *,
    public_asset_endpoint: str,
    seconds: int,
) -> dict[str, Any]:
    return _failed_worker_result(
        sample,
        storage,
        doc_output_dir,
        public_asset_endpoint=public_asset_endpoint,
        status="timeout",
        message=f"document parse timed out after {seconds}s",
    )


def _worker_error_result(
    sample: Sample,
    storage: S3Config,
    doc_output_dir: Path,
    *,
    public_asset_endpoint: str,
    message: str,
) -> dict[str, Any]:
    return _failed_worker_result(
        sample,
        storage,
        doc_output_dir,
        public_asset_endpoint=public_asset_endpoint,
        status="parse_error",
        message=message,
    )


def _failed_worker_result(
    sample: Sample,
    storage: S3Config,
    doc_output_dir: Path,
    *,
    public_asset_endpoint: str,
    status: str,
    message: str,
) -> dict[str, Any]:
    raw = sample.fixture_path.read_bytes()
    source_uri = put_object(storage, f"{sample.id}/source.pdf", raw, "application/pdf")
    source_url = public_url_for_s3_uri(source_uri, public_asset_endpoint)
    result = {
        "id": sample.id,
        "band": sample.band,
        "minio_key": sample.object.key,
        "bytes": len(raw),
        "elapsed_seconds": None,
        "status": status,
        "error": message,
        "uploads": {"source": source_uri, "source_url": source_url},
    }
    metrics_json = _json_bytes(result)
    doc_output_dir.mkdir(parents=True, exist_ok=True)
    (doc_output_dir / "metrics.json").write_bytes(metrics_json)
    result["uploads"]["metrics_json"] = put_object(
        storage,
        f"{sample.id}/metrics.json",
        metrics_json,
        "application/json; charset=utf-8",
    )
    result["uploads"]["metrics_json_url"] = public_url_for_s3_uri(
        result["uploads"]["metrics_json"],
        public_asset_endpoint,
    )
    return result


def _upload_assets(
    storage: S3Config,
    sample_id: str,
    parsed: ParsedDocument,
    *,
    public_endpoint: str,
) -> list[dict[str, Any]]:
    uploaded = []
    for asset in parsed.assets:
        ext = asset.ext.lstrip(".") or "bin"
        key = f"{sample_id}/assets/{asset.id}.{ext}"
        uri = put_object(storage, key, asset.data, asset.mime)
        payload: dict[str, Any] = {
            "id": asset.id,
            "kind": asset.kind,
            "uri": uri,
            "mime": asset.mime,
            "ext": ext,
            "sha256": hashlib.sha256(asset.data).hexdigest(),
            "bytes": len(asset.data),
            "metadata": dict(asset.metadata),
            "public_url": public_url_for_s3_uri(uri, public_endpoint),
        }
        uploaded.append(payload)
    return uploaded


def _quality_metrics(
    parsed: ParsedDocument,
    unit_dicts: list[dict[str, Any]],
) -> dict[str, Any]:
    all_evidence = _collect_evidence(unit_dicts)
    top_counts = Counter(unit.type for unit in parsed.units)
    all_counts = Counter(str(item.get("type", item.get("kind", ""))) for item in all_evidence)
    warning_counts = Counter(
        str(warning.get("type"))
        for warning in parsed.quality_warnings
        if warning.get("type")
    )
    table_metrics = _table_metrics(all_evidence)
    source_text_chars = sum(len(unit.source.text) for unit in parsed.units)
    ocr_units = [
        unit for unit in parsed.units
        if isinstance(unit.metadata, dict) and unit.metadata.get("ocr")
    ]
    return {
        "units": {
            "top_level_total": len(parsed.units),
            "top_level_by_type": dict(sorted(top_counts.items())),
            "all_evidence_total": len(all_evidence),
            "all_evidence_by_type": dict(sorted(all_counts.items())),
            "source_text_chars": source_text_chars,
            "ocr_top_level_units": len(ocr_units),
        },
        "tables": table_metrics,
        "assets": {
            "total": len(parsed.assets),
            "bytes": sum(len(asset.data) for asset in parsed.assets),
            "by_mime": dict(sorted(Counter(asset.mime for asset in parsed.assets).items())),
            "nested_asset_refs": _count_asset_refs(unit_dicts),
        },
        "quality_warnings": {
            "total": len(parsed.quality_warnings),
            "by_type": dict(sorted(warning_counts.items())),
            "items": parsed.quality_warnings[:20],
        },
    }


def _collect_evidence(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, list):
        for item in value:
            found.extend(_collect_evidence(item))
        return found
    if not isinstance(value, dict):
        return found
    if "type" in value or "kind" in value:
        found.append(value)
    for child in value.get("children", []) if isinstance(value.get("children"), list) else []:
        found.extend(_collect_evidence(child))
    content = value.get("content")
    if isinstance(content, dict):
        for row_key in ("header_rows", "rows"):
            rows = content.get(row_key)
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    for cell in row.get("cells", []):
                        if isinstance(cell, dict):
                            found.extend(_collect_evidence(cell.get("children", [])))
    return found


def _table_metrics(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    tables = [
        item.get("content", {})
        for item in evidence
        if item.get("type", item.get("kind")) == "table"
        and isinstance(item.get("content"), dict)
    ]
    blank_cells = 0
    total_cells = 0
    rowspans = 0
    colspans = 0
    nested_children = 0
    max_columns = 0
    total_rows = 0
    for table in tables:
        max_columns = max(max_columns, len(table.get("columns", [])))
        rows = []
        for row_key in ("header_rows", "rows"):
            values = table.get(row_key)
            if isinstance(values, list):
                rows.extend(row for row in values if isinstance(row, dict))
        total_rows += len(rows)
        for row in rows:
            for cell in row.get("cells", []):
                if not isinstance(cell, dict):
                    continue
                total_cells += 1
                if not str(cell.get("text", "")).strip() and not cell.get("children"):
                    blank_cells += 1
                if int(cell.get("rowspan") or 1) > 1:
                    rowspans += 1
                if int(cell.get("colspan") or 1) > 1:
                    colspans += 1
                children = cell.get("children")
                if isinstance(children, list):
                    nested_children += len(children)
    return {
        "total": len(tables),
        "max_columns": max_columns,
        "total_rows": total_rows,
        "total_cells": total_cells,
        "blank_cells": blank_cells,
        "blank_cell_ratio": round(blank_cells / total_cells, 3) if total_cells else 0.0,
        "rowspan_cells": rowspans,
        "colspan_cells": colspans,
        "nested_cell_children": nested_children,
    }


def _review_flags(quality: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    units = quality["units"]
    tables = quality["tables"]
    warnings = quality["quality_warnings"]
    if units["top_level_total"] == 0:
        flags.append("no_units")
    if units["source_text_chars"] < 200 and quality["assets"]["total"] > 0:
        flags.append("low_text_with_assets")
    if tables["total"] and tables["blank_cell_ratio"] >= 0.45:
        flags.append("high_table_blank_ratio")
    if warnings["by_type"].get("pdf_ocr_failed"):
        flags.append("ocr_failed")
    if warnings["total"] >= 10:
        flags.append("many_quality_warnings")
    return flags


def _count_asset_refs(value: Any) -> int:
    if isinstance(value, list):
        return sum(_count_asset_refs(item) for item in value)
    if not isinstance(value, dict):
        return 0
    count = 1 if value.get("format") == "asset_ref" else 0
    return count + sum(_count_asset_refs(item) for item in value.values())


def _summary_payload(
    *,
    run_id: str,
    pdf_count: int,
    samples: list[Sample],
    outliers: list[dict[str, Any]],
    results: list[dict[str, Any]],
    storage: S3Config,
    public_endpoint: str,
) -> dict[str, Any]:
    statuses = Counter(str(result.get("status")) for result in results)
    flags = Counter(
        flag
        for result in results
        for flag in result.get("quality", {}).get("review_flags", [])
    )
    by_band: dict[str, dict[str, Any]] = {}
    for result in results:
        band = str(result["band"])
        by_band.setdefault(
            band,
            {"total": 0, "parse_errors": 0, "review_flags": Counter()},
        )
        by_band[band]["total"] += 1
        if result.get("status") != "ok":
            by_band[band]["parse_errors"] += 1
        for flag in result.get("quality", {}).get("review_flags", []):
            by_band[band]["review_flags"][flag] += 1
    for band in by_band.values():
        band["review_flags"] = dict(sorted(band["review_flags"].items()))
    return {
        "run_id": run_id,
        "source_pdf_count": pdf_count,
        "size_bands": [
            {"id": band, "min_exclusive": lower, "max_inclusive": upper}
            for band, lower, upper in _BANDS
        ],
        "outliers_above_largest_band": outliers,
        "status_counts": dict(sorted(statuses.items())),
        "review_flag_counts": dict(sorted(flags.items())),
        "by_band": by_band,
        "documents": results,
        "fixture_documents": [
            {
                "id": sample.id,
                "band": sample.band,
                "minio_key": sample.object.key,
                "fixture_path": str(sample.fixture_path),
                "bytes": sample.object.size,
            }
            for sample in samples
        ],
        "result_prefix": f"s3://{storage.bucket}/{storage.prefix}",
        "public_index_url": public_url_for_s3_uri(
            f"s3://{storage.bucket}/{storage.prefix}/index.html",
            public_endpoint,
        ),
    }


def _summary_html(summary: dict[str, Any]) -> str:
    rows = []
    for doc in summary["documents"]:
        quality = doc.get("quality", {})
        units = quality.get("units", {})
        tables = quality.get("tables", {})
        warnings = quality.get("quality_warnings", {})
        uploads = doc.get("uploads", {})
        flags = ", ".join(quality.get("review_flags", []))
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(doc.get('band')))}</td>"
            f"<td>{html.escape(str(doc.get('id')))}</td>"
            f"<td>{html.escape(str(doc.get('bytes')))}</td>"
            f"<td>{html.escape(str(doc.get('elapsed_seconds')))}</td>"
            f"<td>{html.escape(str(units.get('top_level_total', '')))}</td>"
            f"<td>{html.escape(str(tables.get('total', '')))}</td>"
            f"<td>{html.escape(str(warnings.get('total', '')))}</td>"
            f"<td>{html.escape(flags)}</td>"
            f"<td><a href=\"{html.escape(str(uploads.get('evidence_html_url', '')))}\">html</a></td>"
            f"<td>{html.escape(str(doc.get('minio_key')))}</td>"
            "</tr>"
        )
    return "\n".join(
        [
            "<!doctype html>",
            "<html lang=\"ko\"><head><meta charset=\"utf-8\">",
            "<title>PDF size validation</title>",
            "<style>body{font-family:system-ui,sans-serif;margin:24px}"
            "table{border-collapse:collapse;width:100%;font-size:13px}"
            "td,th{border:1px solid #ddd;padding:6px;vertical-align:top}"
            "th{background:#f5f5f5;text-align:left}</style>",
            "</head><body>",
            f"<h1>{html.escape(str(summary['run_id']))}</h1>",
            f"<p>source PDFs: {summary['source_pdf_count']}</p>",
            "<table><thead><tr>"
            "<th>band</th><th>id</th><th>bytes</th><th>sec</th>"
            "<th>units</th><th>tables</th><th>warnings</th><th>flags</th>"
            "<th>html</th><th>minio key</th></tr></thead><tbody>",
            *rows,
            "</tbody></table>",
            "</body></html>",
        ]
    )


def _ocr_config(args: argparse.Namespace) -> LlmConfig | None:
    api_key = args.ocr_api_key
    if not api_key:
        return None
    return LlmConfig(
        url=args.ocr_base_url,
        api_key=api_key,
        model=args.ocr_model,
        timeout=args.ocr_timeout,
    )


class _document_timeout:
    def __init__(self, seconds: int) -> None:
        self.seconds = seconds
        self.previous_handler: Any = None

    def __enter__(self) -> None:
        if self.seconds <= 0:
            return
        self.previous_handler = signal.signal(signal.SIGALRM, self._raise_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.seconds <= 0:
            return
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self.previous_handler)

    def _raise_timeout(self, signum: int, frame: Any) -> None:
        raise TimeoutError(f"document parse timed out after {self.seconds}s")


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return safe[:120]


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def _timestamped_run_id(run_id: str | None, *, now: datetime | None = None) -> str:
    name = (run_id or "pdf-size-validation").strip()
    if re.match(r"^\d{8}-\d{6}-", name):
        return name
    timestamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{name}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate PDF extraction quality across clic MinIO size bands.",
    )
    parser.add_argument("--minio-container", default="clic-minio")
    parser.add_argument("--mc-binary", default="/usr/bin/mc")
    parser.add_argument("--minio-source", default="local/clic/raw")
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=Path("tests/fixtures/corpus/pdf-size-validation"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/rag-document-parser-pdf-size-validation"),
    )
    parser.add_argument("--worker-input", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--run-id")
    parser.add_argument("--max-ocr-workers", type=int, default=2)
    parser.add_argument(
        "--per-document-timeout",
        type=int,
        default=int(os.getenv("RDP_PDF_VALIDATION_DOC_TIMEOUT", "240")),
        help="Seconds before one PDF is marked as timeout; 0 disables timeout.",
    )
    parser.add_argument(
        "--ocr-base-url",
        default=os.getenv("RDP_PDF_OCR_BASE_URL", "http://localhost:10080/v1"),
    )
    parser.add_argument(
        "--ocr-api-key",
        default=os.getenv("RDP_PDF_OCR_API_KEY", ""),
    )
    parser.add_argument(
        "--ocr-model",
        default=os.getenv("RDP_PDF_OCR_MODEL", "qwen3-vl-30b-a3b"),
    )
    parser.add_argument(
        "--ocr-timeout",
        type=float,
        default=float(os.getenv("RDP_PDF_OCR_TIMEOUT", "240")),
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
        "--public-asset-endpoint",
        default=os.getenv("RDP_PUBLIC_ASSET_ENDPOINT", "http://192.168.21.62:10190"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
