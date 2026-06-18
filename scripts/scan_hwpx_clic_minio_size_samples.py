from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import shlex
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

from rag_document_parser import HwpxBackend, S3Config
from rag_document_parser.backends import PendingAsset
from rag_document_parser.renderer.evidence_unit_render import render_evidence_units_html
from rag_document_parser.storage import public_url_for_s3_uri, put_object


def main() -> None:
    args = _parse_args()
    documents = _list_hwpx_documents(
        args.source_prefix,
        mc_command=args.mc_command,
    )
    buckets = _select_size_candidate_buckets(documents, per_bucket=args.per_bucket)
    run_id = _timestamped_run_id(args.run_id)
    save_dir = args.save_dir
    validation_dir = args.validation_output_dir / run_id
    save_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)

    storage = S3Config(
        endpoint=args.s3_endpoint,
        bucket=args.s3_bucket,
        access_key=args.s3_access_key,
        secret_key=args.s3_secret_key,
        prefix=f"{args.s3_prefix.strip('/')}/{run_id}".strip("/"),
        region=args.s3_region,
    )
    public_endpoint = args.public_endpoint or args.s3_endpoint

    sampled: list[dict[str, Any]] = []
    skipped_documents: list[dict[str, Any]] = []
    seen_sha256: set[str] = set()
    target_total = _target_sample_total(buckets, per_bucket=args.per_bucket)
    for bucket, bucket_documents in buckets.items():
        selected_samples, skipped = _select_unique_raw_samples(
            bucket_documents,
            per_bucket=args.per_bucket,
            read_raw=lambda document: _read_mc_path(
                str(document["mc_path"]),
                mc_command=args.mc_command,
            ),
            seen_sha256=seen_sha256,
        )
        skipped_documents.extend(
            {"bucket": bucket, **skipped_document}
            for skipped_document in skipped
        )
        for index, sample in enumerate(selected_samples, start=1):
            document = sample["document"]
            sampled.append(
                _save_and_validate_sample(
                    bucket=bucket,
                    index=index,
                    document=document,
                    raw=sample["raw"],
                    document_sha256=sample["sha256"],
                    save_dir=save_dir,
                    validation_dir=validation_dir,
                    storage=storage,
                    public_endpoint=public_endpoint,
                )
            )
            if args.progress:
                print(
                    json.dumps(
                        {
                            "validated": len(sampled),
                            "total": target_total,
                            "bucket": bucket,
                            "key": document["key"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    report = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "source_prefix": args.source_prefix,
        "selection": {
            "per_bucket": args.per_bucket,
            "total_hwpx_documents": len(documents),
            "size_bytes": _size_distribution(documents),
            "skipped_documents": skipped_documents,
        },
        "documents": sampled,
        "summary": _quality_summary(sampled),
    }
    report["report_uri"] = _s3_uri_for_key(storage, "quality-report.json")
    report["report_url"] = public_url_for_s3_uri(report["report_uri"], public_endpoint)
    report["index_uri"] = _s3_uri_for_key(storage, "index.html")
    report["index_url"] = public_url_for_s3_uri(report["index_uri"], public_endpoint)
    report["screenshot_warnings"] = []
    if args.screenshot_mode != "off":
        report["screenshot_warnings"] = _capture_and_upload_screenshots(
            report,
            validation_dir=validation_dir,
            storage=storage,
            public_endpoint=public_endpoint,
            geckodriver_command=args.geckodriver_command,
        )

    report_json = _json_bytes(report)
    manifest_json = _json_bytes(_stable_manifest_from_report(report))
    index_html = _render_index_html(report).encode("utf-8")
    (save_dir / "manifest.json").write_bytes(manifest_json)
    (validation_dir / "quality-report.json").write_bytes(report_json)
    (validation_dir / "index.html").write_bytes(index_html)
    put_object(
        storage,
        "quality-report.json",
        report_json,
        "application/json; charset=utf-8",
    )
    put_object(
        storage,
        "index.html",
        index_html,
        "text/html; charset=utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _list_hwpx_documents(
    source_prefix: str,
    *,
    mc_command: str,
) -> list[dict[str, Any]]:
    output = _run_mc(
        mc_command,
        ["ls", source_prefix, "--recursive", "--json"],
    )
    documents = []
    for line in output.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("type") != "file":
            continue
        key = str(item.get("key", ""))
        if not key.lower().endswith(".hwpx"):
            continue
        documents.append(
            {
                "key": key,
                "size": int(item.get("size") or 0),
                "last_modified": item.get("lastModified"),
                "source_uri": f"s3://clic/raw/{key}",
                "mc_path": f"{source_prefix.rstrip('/')}/{key}",
            }
        )
    return sorted(documents, key=lambda item: (int(item["size"]), str(item["key"])))


def _select_size_samples(
    documents: list[dict[str, Any]],
    *,
    per_bucket: int,
) -> dict[str, list[dict[str, Any]]]:
    return {
        bucket: candidates[:per_bucket]
        for bucket, candidates in _select_size_candidate_buckets(
            documents,
            per_bucket=per_bucket,
        ).items()
    }


def _select_size_candidate_buckets(
    documents: list[dict[str, Any]],
    *,
    per_bucket: int,
) -> dict[str, list[dict[str, Any]]]:
    if per_bucket <= 0:
        raise ValueError("per_bucket must be positive")
    sorted_documents = sorted(
        documents,
        key=lambda item: (int(item["size"]), str(item["key"])),
    )
    if len(sorted_documents) < per_bucket:
        return {"small": sorted_documents, "medium": [], "large": []}
    medium_start = max(0, (len(sorted_documents) - per_bucket) // 2)
    return {
        "small": _diversify_documents(sorted_documents),
        "medium": _diversify_documents(
            sorted_documents[medium_start:] + sorted_documents[:medium_start],
        ),
        "large": _diversify_documents(list(reversed(sorted_documents))),
    }


def _diversify_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    seen_groups: set[str] = set()
    seen_families: set[str] = set()

    def add(document: dict[str, Any]) -> None:
        selected.append(document)
        selected_ids.add(id(document))
        seen_groups.add(_document_group(document))
        seen_families.add(_document_family(document))

    for document in documents:
        if (
            _document_group(document) not in seen_groups
            and _document_family(document) not in seen_families
        ):
            add(document)

    for document in documents:
        if id(document) not in selected_ids and _document_family(document) not in seen_families:
            add(document)

    for document in documents:
        if id(document) not in selected_ids:
            add(document)

    return selected


def _document_group(document: dict[str, Any]) -> str:
    key = str(document.get("key", ""))
    return key.split("/", 1)[0]


def _document_family(document: dict[str, Any]) -> str:
    name = str(document.get("key", "")).rsplit("/", 1)[-1].removesuffix(".hwpx")
    name = name.replace("+", " ").replace("_", " ")
    name = re.sub(r"\([^)]*\)", " ", name)
    name = re.sub(r"\[[^\]]*\]", " ", name)
    name = re.sub(r"제\s*\d{4}\s*-\s*\d+\s*호", " ", name)
    name = re.sub(r"\d{4}[.-]\d{1,2}[.-]\d{1,2}", " ", name)
    name = re.sub(r"\d+", " ", name)
    name = re.sub(r"[-¸,.;:]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip().casefold()
    return name or str(document.get("key", ""))


def _select_unique_raw_samples(
    documents: list[dict[str, Any]],
    *,
    per_bucket: int,
    read_raw: Any,
    seen_sha256: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seen = seen_sha256 if seen_sha256 is not None else set()
    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for document in documents:
        raw = read_raw(document)
        digest = hashlib.sha256(raw).hexdigest()
        if digest in seen:
            skipped.append(
                {
                    "key": str(document.get("key", "")),
                    "reason": "duplicate_sha256",
                    "sha256": digest,
                }
            )
            continue
        seen.add(digest)
        selected.append({"document": document, "raw": raw, "sha256": digest})
        if len(selected) >= per_bucket:
            break
    return selected, skipped


def _target_sample_total(
    buckets: dict[str, list[Any]],
    *,
    per_bucket: int,
) -> int:
    return len(buckets) * per_bucket


def _save_and_validate_sample(
    *,
    bucket: str,
    index: int,
    document: dict[str, Any],
    raw: bytes,
    document_sha256: str,
    save_dir: Path,
    validation_dir: Path,
    storage: S3Config,
    public_endpoint: str,
) -> dict[str, Any]:
    filename = _sample_filename(bucket, index, document)
    bucket_dir = save_dir / bucket
    bucket_dir.mkdir(parents=True, exist_ok=True)
    sample_path = bucket_dir / filename
    sample_path.write_bytes(raw)

    started = time.perf_counter()
    validation: dict[str, Any]
    try:
        parsed = HwpxBackend().parse(raw, ".hwpx")
        elapsed = time.perf_counter() - started
        unit_dicts = [unit.to_dict() for unit in parsed.units]
        uploaded_assets = _upload_assets(
            storage,
            parsed.assets,
            bucket=bucket,
            document=document,
            public_endpoint=public_endpoint,
        )
        html = render_evidence_units_html(
            unit_dicts,
            title=f"{bucket} {index}: {document['key']}",
            assets=uploaded_assets,
        ).encode("utf-8")
        html_key = _html_object_key(bucket, document)
        html_uri = put_object(
            storage,
            html_key,
            html,
            "text/html; charset=utf-8",
        )
        local_html = validation_dir / bucket / f"{filename.removesuffix('.hwpx')}.html"
        local_html.parent.mkdir(parents=True, exist_ok=True)
        local_html.write_bytes(html)
        validation = {
            "ok": True,
            "elapsed_seconds": round(elapsed, 3),
            "units": _unit_counts(unit_dicts),
            "assets": len(parsed.assets),
            "uploaded_assets": uploaded_assets,
            "quality_warnings": parsed.quality_warnings,
            "html_uri": html_uri,
            "html_url": public_url_for_s3_uri(html_uri, public_endpoint),
            "local_html": str(local_html),
        }
    except Exception as exc:  # pragma: no cover - exercised by real scan.
        validation = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "bucket": bucket,
        "index": index,
        "key": document["key"],
        "source_uri": document["source_uri"],
        "filename": filename,
        "path": str(sample_path),
        "bytes": len(raw),
        "sha256": document_sha256,
        "validation": validation,
    }


def _sample_filename(bucket: str, index: int, document: dict[str, Any]) -> str:
    key = str(document["key"])
    size = int(document["size"])
    stem = key.removesuffix(".hwpx")
    slug = re.sub(r"[^\w가-힣.-]+", "-", stem, flags=re.UNICODE).strip("-")
    slug = re.sub(r"-+", "-", slug)[:120].strip("-")
    return f"{bucket}-{index:02d}-{size}-{slug}.hwpx"


def _html_object_key(bucket: str, document: dict[str, Any]) -> str:
    return f"{_document_object_prefix(bucket, document)}/evidence-units.html"


def _document_object_prefix(bucket: str, document: dict[str, Any]) -> str:
    digest = hashlib.sha1(str(document["key"]).encode("utf-8")).hexdigest()
    return f"{bucket}/{digest}"


def _asset_object_key(
    bucket: str,
    document: dict[str, Any],
    asset_id: str,
    ext: str,
) -> str:
    safe_ext = ext.lstrip(".") or "bin"
    return f"{_document_object_prefix(bucket, document)}/assets/{asset_id}.{safe_ext}"


def _s3_uri_for_key(storage: S3Config, key: str) -> str:
    full_key = f"{storage.prefix.strip('/')}/{key}" if storage.prefix else key
    return f"s3://{storage.bucket}/{full_key}"


def _upload_assets(
    storage: S3Config,
    assets: list[PendingAsset],
    *,
    bucket: str,
    document: dict[str, Any],
    public_endpoint: str | None = None,
) -> list[dict[str, Any]]:
    uploaded = []
    for asset in assets:
        ext = asset.ext.lstrip(".") or "bin"
        uri = put_object(
            storage,
            _asset_object_key(bucket, document, asset.id, ext),
            asset.data,
            asset.mime,
        )
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


def _stable_manifest_from_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_prefix": report.get("source_prefix"),
        "selection": {
            "per_bucket": report.get("selection", {}).get("per_bucket"),
            "total_hwpx_documents": report.get("selection", {}).get(
                "total_hwpx_documents"
            ),
            "size_bytes": report.get("selection", {}).get("size_bytes", {}),
            "skipped_documents": report.get("selection", {}).get(
                "skipped_documents",
                [],
            ),
        },
        "documents": [
            _stable_manifest_document(document)
            for document in report.get("documents", [])
            if isinstance(document, dict)
        ],
    }


def _stable_manifest_document(document: dict[str, Any]) -> dict[str, Any]:
    validation = document.get("validation", {})
    if not isinstance(validation, dict):
        validation = {}
    return {
        "bucket": document.get("bucket"),
        "index": document.get("index"),
        "key": document.get("key"),
        "source_uri": document.get("source_uri"),
        "filename": document.get("filename"),
        "path": document.get("path"),
        "bytes": document.get("bytes"),
        "sha256": document.get("sha256"),
        "expected": {
            "units": validation.get("units", {"total": 0, "by_type": {}}),
            "assets": validation.get("assets", 0),
            "quality_warnings": validation.get("quality_warnings", []),
        },
    }


def _screenshot_targets(report: dict[str, Any]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    seen_labels: set[tuple[str, str]] = set()
    for document in report.get("documents", []):
        if not isinstance(document, dict):
            continue
        validation = document.get("validation", {})
        if not isinstance(validation, dict) or not validation.get("ok"):
            continue
        url = validation.get("html_url")
        if not isinstance(url, str) or not url:
            continue
        units = validation.get("units", {})
        by_type = units.get("by_type", {}) if isinstance(units, dict) else {}
        candidates = []
        if isinstance(by_type, dict) and by_type.get("diagram", 0):
            candidates.append(("diagram", ".diagram-positioned"))
        if validation.get("assets", 0):
            candidates.append(("asset", ".nested-evidence, figure"))
        if isinstance(by_type, dict) and by_type.get("table", 0):
            candidates.append(("table", ".evidence-table"))

        for label, selector in candidates:
            bucket = str(document.get("bucket", ""))
            target_key = (bucket, label)
            if label in {"diagram", "asset"}:
                target_key = ("*", label)
            if target_key in seen_labels:
                continue
            seen_labels.add(target_key)
            targets.append(
                {
                    "bucket": document.get("bucket"),
                    "index": document.get("index"),
                    "label": label,
                    "selector": selector,
                    "url": url,
                }
            )
    return targets


def _capture_and_upload_screenshots(
    report: dict[str, Any],
    *,
    validation_dir: Path,
    storage: S3Config,
    public_endpoint: str,
    geckodriver_command: str,
) -> list[dict[str, Any]]:
    driver = shutil.which(shlex.split(geckodriver_command)[0])
    if driver is None:
        return [
            {
                "type": "screenshot_unavailable",
                "message": f"geckodriver not found: {geckodriver_command}",
            }
        ]

    targets = _screenshot_targets(report)
    if not targets:
        return []

    screenshot_dir = validation_dir / "screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    port = _free_local_port()
    proc = subprocess.Popen(
        [*shlex.split(geckodriver_command), "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_webdriver(base_url)
        session_id = _webdriver_request(
            base_url,
            "POST",
            "/session",
            {
                "capabilities": {
                    "alwaysMatch": {
                        "browserName": "firefox",
                        "moz:firefoxOptions": {"args": ["-headless"]},
                    }
                }
            },
        )["value"]["sessionId"]
        try:
            _webdriver_request(
                base_url,
                "POST",
                f"/session/{session_id}/window/rect",
                {"x": 0, "y": 0, "width": 1360, "height": 900},
            )
            for target in targets:
                screenshot = _capture_target_screenshot(base_url, session_id, target)
                filename = (
                    f"{target['bucket']}-{int(target['index']):02d}-"
                    f"{target['label']}.png"
                )
                local_path = screenshot_dir / filename
                local_path.write_bytes(screenshot)
                object_key = f"screenshots/{filename}"
                uri = put_object(storage, object_key, screenshot, "image/png")
                payload = {
                    "label": target["label"],
                    "selector": target["selector"],
                    "uri": uri,
                    "url": public_url_for_s3_uri(uri, public_endpoint),
                    "local_path": str(local_path),
                }
                _attach_screenshot(report, target, payload)
        finally:
            _webdriver_request(base_url, "DELETE", f"/session/{session_id}")
    except Exception as exc:  # pragma: no cover - depends on local browser setup.
        return [
            {
                "type": "screenshot_failed",
                "message": f"{type(exc).__name__}: {exc}",
            }
        ]
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover
            proc.kill()
    return []


def _capture_target_screenshot(
    base_url: str,
    session_id: str,
    target: dict[str, Any],
) -> bytes:
    _webdriver_request(
        base_url,
        "POST",
        f"/session/{session_id}/url",
        {"url": target["url"]},
    )
    time.sleep(1.5)
    element_key = "element-6066-11e4-a52e-4f735466cecf"
    element = _find_screenshot_element(base_url, session_id, target)
    element_id = element[element_key]
    _webdriver_request(
        base_url,
        "POST",
        f"/session/{session_id}/execute/sync",
        {
            "script": "arguments[0].scrollIntoView({block:'center'}); return true;",
            "args": [{element_key: element_id}],
        },
    )
    _webdriver_request(
        base_url,
        "POST",
        f"/session/{session_id}/execute/sync",
        {
            "script": (
                "return Array.from(document.images)"
                ".every(img => img.complete && img.naturalWidth > 0);"
            ),
            "args": [],
        },
    )
    time.sleep(0.8)
    encoded = _webdriver_request(
        base_url,
        "GET",
        f"/session/{session_id}/element/{element_id}/screenshot",
    )["value"]
    return base64.b64decode(encoded)


def _find_screenshot_element(
    base_url: str,
    session_id: str,
    target: dict[str, Any],
) -> dict[str, Any]:
    lookup = _element_lookup(target)
    if lookup["mode"] == "css":
        return _webdriver_request(
            base_url,
            "POST",
            f"/session/{session_id}/element",
            {"using": "css selector", "value": lookup["selector"]},
        )["value"]
    return _webdriver_request(
        base_url,
        "POST",
        f"/session/{session_id}/execute/sync",
        {"script": lookup["script"], "args": []},
    )["value"]


def _element_lookup(target: dict[str, Any]) -> dict[str, str]:
    if target.get("label") != "asset":
        return {"mode": "css", "selector": str(target.get("selector", ""))}
    return {
        "mode": "script",
        "script": """
const candidates = Array.from(document.querySelectorAll('.nested-evidence, figure'));
const nested = candidates.filter((element) => element.matches('.nested-evidence'));
const pool = nested.length ? nested : candidates;
if (!pool.length) {
  return null;
}
pool.sort((left, right) => {
  const leftRect = left.getBoundingClientRect();
  const rightRect = right.getBoundingClientRect();
  return (rightRect.width * rightRect.height) - (leftRect.width * leftRect.height);
});
return pool[0];
""".strip(),
    }


def _attach_screenshot(
    report: dict[str, Any],
    target: dict[str, Any],
    screenshot: dict[str, Any],
) -> None:
    for document in report.get("documents", []):
        if not isinstance(document, dict):
            continue
        if (
            document.get("bucket") == target.get("bucket")
            and document.get("index") == target.get("index")
        ):
            validation = document.setdefault("validation", {})
            if isinstance(validation, dict):
                validation.setdefault("screenshots", []).append(screenshot)
            return


def _wait_for_webdriver(base_url: str) -> None:
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{base_url}/status", timeout=0.5).read()
            return
        except Exception:
            time.sleep(0.2)
    raise RuntimeError("geckodriver did not start")


def _webdriver_request(
    base_url: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        body = urllib.request.urlopen(request, timeout=60).read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"{method} {path} failed: {exc.code} {detail[:1000]}") from exc
    return json.loads(body.decode("utf-8")) if body else {}


def _free_local_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _render_index_html(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    selection = report.get("selection", {})
    rows = "\n".join(
        _render_index_row(document)
        for document in report.get("documents", [])
        if isinstance(document, dict)
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>HWPX size sample validation</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #1f2933; }}
    a {{ color: #0b5cad; }}
    table {{ border-collapse: collapse; width: 100%; table-layout: fixed; }}
    th, td {{ border: 1px solid #d9dee5; padding: 8px; vertical-align: top; }}
    th {{ background: #eef2f7; text-align: left; }}
    code {{ word-break: break-all; }}
    .summary {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0 24px; }}
    .summary div {{ border: 1px solid #d9dee5; padding: 10px 12px; }}
    .ok {{ color: #126b37; font-weight: 600; }}
    .failed {{ color: #a61b1b; font-weight: 600; }}
  </style>
</head>
<body>
  <h1>HWPX size sample validation</h1>
  <div class="summary">
    <div>Started: {escape(str(report.get("started_at", "")))}</div>
    <div>Source docs: {escape(str(selection.get("total_hwpx_documents", "")))}</div>
    <div>Validated: {escape(str(summary.get("documents", "")))}</div>
    <div>Failed: {escape(str(summary.get("failed", "")))}</div>
    <div>Warnings: {escape(str(summary.get("with_warnings", "")))}</div>
  </div>
  <p>Size bytes: {escape(_format_size_distribution(selection.get("size_bytes", {})))}</p>
  <p>JSON report: {_render_link(report.get("report_url"), "quality-report.json")}</p>
  <table>
    <thead>
      <tr>
        <th style="width: 8%">Bucket</th>
        <th style="width: 8%">Bytes</th>
        <th style="width: 28%">Source key</th>
        <th style="width: 18%">Units</th>
        <th style="width: 8%">Assets</th>
        <th style="width: 10%">Warnings</th>
        <th style="width: 10%">Status</th>
        <th style="width: 10%">HTML</th>
        <th style="width: 10%">Screenshots</th>
      </tr>
    </thead>
    <tbody>
{rows}
    </tbody>
  </table>
</body>
</html>
"""


def _render_index_row(document: dict[str, Any]) -> str:
    validation = document.get("validation", {})
    if not isinstance(validation, dict):
        validation = {}
    ok = bool(validation.get("ok"))
    status_class = "ok" if ok else "failed"
    status = "ok" if ok else str(validation.get("error", "failed"))
    warnings = validation.get("quality_warnings", [])
    warning_count = len(warnings) if isinstance(warnings, list) else 0
    return (
        "      <tr>"
        f"<td>{escape(str(document.get('bucket', '')))}-{escape(str(document.get('index', '')))}</td>"
        f"<td>{escape(str(document.get('bytes', '')))}</td>"
        f"<td><code>{escape(str(document.get('key', '')))}</code></td>"
        f"<td>{escape(_format_unit_counts(validation.get('units', {})))}</td>"
        f"<td>{escape(str(validation.get('assets', '')))}</td>"
        f"<td>{warning_count}</td>"
        f'<td class="{status_class}">{escape(status)}</td>'
        f"<td>{_render_link(validation.get('html_url'), 'open')}</td>"
        f"<td>{_render_screenshot_links(validation.get('screenshots', []))}</td>"
        "</tr>"
    )


def _render_screenshot_links(screenshots: Any) -> str:
    if not isinstance(screenshots, list):
        return ""
    links = []
    for screenshot in screenshots:
        if not isinstance(screenshot, dict):
            continue
        label = str(screenshot.get("label") or "screenshot")
        link = _render_link(screenshot.get("url"), label)
        if link:
            links.append(link)
    return ", ".join(links)


def _render_link(url: Any, label: str) -> str:
    if not isinstance(url, str) or not url:
        return ""
    return f'<a href="{escape(url, quote=True)}">{escape(label)}</a>'


def _format_unit_counts(units: Any) -> str:
    if not isinstance(units, dict):
        return ""
    by_type = units.get("by_type", {})
    if not isinstance(by_type, dict):
        return ""
    return ", ".join(f"{key}: {value}" for key, value in sorted(by_type.items()))


def _format_size_distribution(size_bytes: Any) -> str:
    if not isinstance(size_bytes, dict):
        return ""
    return ", ".join(f"{key}: {value}" for key, value in size_bytes.items())


def _read_mc_path(mc_path: str, *, mc_command: str) -> bytes:
    command = [*shlex.split(mc_command), "cat", mc_path]
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def _run_mc(mc_command: str, args: list[str]) -> str:
    completed = subprocess.run(
        [*shlex.split(mc_command), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout


def _unit_counts(units: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(unit.get("type", "")) for unit in units)
    return {
        "total": len(units),
        "by_type": dict(sorted(counts.items())),
    }


def _quality_summary(documents: list[dict[str, Any]]) -> dict[str, Any]:
    warning_counts: Counter[str] = Counter()
    failures = 0
    for document in documents:
        validation = document.get("validation", {})
        if not validation.get("ok"):
            failures += 1
            continue
        for warning in validation.get("quality_warnings", []):
            if isinstance(warning, dict):
                warning_counts[str(warning.get("type", "unknown"))] += 1
    return {
        "documents": len(documents),
        "failed": failures,
        "with_warnings": sum(
            1
            for document in documents
            if document.get("validation", {}).get("quality_warnings")
        ),
        "warning_types": dict(sorted(warning_counts.items())),
    }


def _size_distribution(documents: list[dict[str, Any]]) -> dict[str, int]:
    sizes = sorted(int(document["size"]) for document in documents)
    if not sizes:
        return {}
    return {
        "min": sizes[0],
        "p25": sizes[round((len(sizes) - 1) * 0.25)],
        "p50": sizes[round((len(sizes) - 1) * 0.5)],
        "p75": sizes[round((len(sizes) - 1) * 0.75)],
        "p90": sizes[round((len(sizes) - 1) * 0.9)],
        "max": sizes[-1],
    }


def _timestamped_run_id(run_id: str | None, *, now: datetime | None = None) -> str:
    name = (run_id or "validation").strip()
    if re.match(r"^\d{8}-\d{6}-", name):
        return name
    timestamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{name}"


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample HWPX clic MinIO documents by file size and validate output.",
    )
    parser.add_argument(
        "--source-prefix",
        default=os.getenv("RDP_SCAN_SOURCE_PREFIX", "local/clic/raw"),
    )
    parser.add_argument(
        "--mc-command",
        default=os.getenv("RDP_MC_COMMAND", "docker exec clic-minio mc"),
    )
    parser.add_argument("--per-bucket", type=int, default=5)
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("tests/fixtures/corpus/hwpx_size_samples"),
    )
    parser.add_argument(
        "--validation-output-dir",
        type=Path,
        default=Path("/tmp/rag-document-parser-validation/hwpx-size-samples"),
    )
    parser.add_argument("--run-id")
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
        default=os.getenv("RDP_S3_PREFIX", "hwpx-size-validation"),
    )
    parser.add_argument(
        "--public-endpoint",
        default=os.getenv("RDP_PUBLIC_ASSET_ENDPOINT", "http://192.168.21.62:10190"),
    )
    parser.add_argument(
        "--screenshot-mode",
        choices=["auto", "off"],
        default=os.getenv("RDP_SCREENSHOT_MODE", "auto"),
    )
    parser.add_argument(
        "--geckodriver-command",
        default=os.getenv("RDP_GECKODRIVER_COMMAND", "geckodriver"),
    )
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
