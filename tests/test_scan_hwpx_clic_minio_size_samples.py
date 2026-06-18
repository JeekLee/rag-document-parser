from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


def test_select_size_samples_picks_small_medium_and_large_buckets():
    scan_hwpx = _load_scan_script()
    documents = [
        {"key": f"raw/doc-{index:02d}.hwpx", "size": size}
        for index, size in enumerate(range(10, 260, 10), start=1)
    ]

    buckets = scan_hwpx._select_size_samples(documents, per_bucket=5)

    assert [item["size"] for item in buckets["small"]] == [10, 20, 30, 40, 50]
    assert [item["size"] for item in buckets["medium"]] == [110, 120, 130, 140, 150]
    assert [item["size"] for item in buckets["large"]] == [250, 240, 230, 220, 210]


def test_select_size_samples_prefers_distinct_document_families():
    scan_hwpx = _load_scan_script()
    documents = [
        {
            "key": "20260101-1-0001/의료급여수가의 기준 및 일반기준(고시 A)_전문.hwpx",
            "size": 1000,
        },
        {
            "key": "20250701-1-0001/의료급여수가의 기준 및 일반기준(고시 B)_전문.hwpx",
            "size": 990,
        },
        {
            "key": "20260101-1-0005/질의응답(고가의약품 급여관리에 관한 기준).hwpx",
            "size": 980,
        },
        {"key": "20260401-1-0014/치료재료 급여기준 질의응답.hwpx", "size": 970},
        {"key": "20240501-1-0003/기타 문서.hwpx", "size": 960},
        {"key": "20240201-1-0001/소형 문서.hwpx", "size": 10},
        {"key": "20240201-1-0002/중간 문서.hwpx", "size": 500},
    ]

    buckets = scan_hwpx._select_size_samples(documents, per_bucket=3)

    assert [item["key"] for item in buckets["large"]] == [
        "20260101-1-0001/의료급여수가의 기준 및 일반기준(고시 A)_전문.hwpx",
        "20260101-1-0005/질의응답(고가의약품 급여관리에 관한 기준).hwpx",
        "20260401-1-0014/치료재료 급여기준 질의응답.hwpx",
    ]


def test_select_unique_raw_samples_skips_duplicate_sha():
    scan_hwpx = _load_scan_script()
    documents = [
        {"key": "a.hwpx"},
        {"key": "duplicate-a.hwpx"},
        {"key": "b.hwpx"},
    ]
    raw_by_key = {
        "a.hwpx": b"same",
        "duplicate-a.hwpx": b"same",
        "b.hwpx": b"different",
    }

    selected, skipped = scan_hwpx._select_unique_raw_samples(
        documents,
        per_bucket=2,
        read_raw=lambda document: raw_by_key[document["key"]],
    )

    assert [item["document"]["key"] for item in selected] == ["a.hwpx", "b.hwpx"]
    assert skipped == [
        {
            "key": "duplicate-a.hwpx",
            "reason": "duplicate_sha256",
            "sha256": hashlib.sha256(b"same").hexdigest(),
        }
    ]


def test_target_sample_total_counts_requested_bucket_slots():
    scan_hwpx = _load_scan_script()

    total = scan_hwpx._target_sample_total(
        {"small": [1, 2, 3], "medium": [1, 2], "large": [1]},
        per_bucket=5,
    )

    assert total == 15


def test_sample_filename_includes_bucket_size_and_stable_slug():
    scan_hwpx = _load_scan_script()

    filename = scan_hwpx._sample_filename(
        "small",
        3,
        {
            "key": "20241201-3-0004/고시 제2024-235호 Omidenepag isopropyl 외용제.hwpx",
            "size": 22852,
        },
    )

    assert filename == (
        "small-03-22852-20241201-3-0004-"
        "고시-제2024-235호-Omidenepag-isopropyl-외용제.hwpx"
    )


def test_html_object_key_is_ascii_safe_for_minio_http_upload():
    scan_hwpx = _load_scan_script()

    key = scan_hwpx._html_object_key(
        "small",
        {
            "key": "20241201-3-0004/고시 제2024-235호 Omidenepag isopropyl 외용제.hwpx",
            "size": 22852,
        },
    )

    key.encode("ascii")
    assert key == (
        "small/29beff631edf00767b66e8c81208d969abcdb4f8/"
        "evidence-units.html"
    )


def test_upload_assets_uses_ascii_object_key_and_public_url(monkeypatch):
    from rag_document_parser.models import PendingAsset
    from rag_document_parser.storage import S3Config

    scan_hwpx = _load_scan_script()
    document = {
        "key": "20260101-1-0005/질의응답(고가의약품 급여관리에 관한 기준).hwpx",
        "size": 653742,
    }
    captured: dict[str, object] = {}

    def fake_put_object(cfg, key, data, content_type):
        captured["key"] = key
        captured["data"] = data
        captured["content_type"] = content_type
        return f"s3://rag-assets/validation/run/{key}"

    monkeypatch.setattr(scan_hwpx, "put_object", fake_put_object)

    uploaded = scan_hwpx._upload_assets(
        S3Config(
            endpoint="http://localhost:10190",
            bucket="rag-assets",
            access_key="minioadmin",
            secret_key="minioadmin",
            prefix="validation/run",
        ),
        [
            PendingAsset(
                id="img-0001",
                kind="image",
                data=b"jpg",
                mime="image/jpeg",
                ext=".jpg",
            )
        ],
        bucket="large",
        document=document,
        public_endpoint="http://203.0.113.10:10190",
    )

    digest = hashlib.sha1(str(document["key"]).encode("utf-8")).hexdigest()
    expected_key = f"large/{digest}/assets/img-0001.jpg"
    captured["key"].encode("ascii")
    assert captured == {
        "key": expected_key,
        "data": b"jpg",
        "content_type": "image/jpeg",
    }
    assert uploaded[0]["uri"] == f"s3://rag-assets/validation/run/{expected_key}"
    assert uploaded[0]["public_url"] == (
        "http://203.0.113.10:10190/rag-assets/validation/run/"
        f"{expected_key}"
    )


def test_s3_uri_for_key_uses_storage_prefix():
    from rag_document_parser.storage import S3Config

    scan_hwpx = _load_scan_script()

    uri = scan_hwpx._s3_uri_for_key(
        S3Config(
            endpoint="http://localhost:10190",
            bucket="rag-assets",
            access_key="minioadmin",
            secret_key="minioadmin",
            prefix="validation/run",
        ),
        "index.html",
    )

    assert uri == "s3://rag-assets/validation/run/index.html"


def test_render_index_html_links_report_and_document_html():
    scan_hwpx = _load_scan_script()

    html = scan_hwpx._render_index_html(
        {
            "started_at": "2026-06-18T15:08:45",
            "report_url": "http://example.test/report.json",
            "summary": {"documents": 1, "failed": 0, "with_warnings": 0},
            "selection": {
                "total_hwpx_documents": 327,
                "size_bytes": {"min": 1, "p50": 5, "max": 9},
            },
            "documents": [
                {
                    "bucket": "large",
                    "index": 1,
                    "key": "raw/<diagram>.hwpx",
                    "bytes": 1234,
                    "validation": {
                        "ok": True,
                        "units": {"by_type": {"diagram": 1, "table": 2}},
                        "assets": 3,
                        "quality_warnings": [],
                        "html_url": "http://example.test/evidence.html",
                        "screenshots": [
                            {
                                "label": "diagram",
                                "url": "http://example.test/diagram.png",
                            }
                        ],
                    },
                }
            ],
        }
    )

    assert 'href="http://example.test/report.json"' in html
    assert 'href="http://example.test/evidence.html"' in html
    assert 'href="http://example.test/diagram.png"' in html
    assert "diagram: 1, table: 2" in html
    assert "raw/&lt;diagram&gt;.hwpx" in html


def test_stable_manifest_omits_runtime_report_fields():
    scan_hwpx = _load_scan_script()

    manifest = scan_hwpx._stable_manifest_from_report(
        {
            "started_at": "2026-06-18T15:08:45",
            "index_url": "http://example.test/index.html",
            "report_url": "http://example.test/report.json",
            "source_prefix": "local/clic/raw",
            "selection": {
                "per_bucket": 5,
                "total_hwpx_documents": 327,
                "size_bytes": {"min": 1, "max": 9},
            },
            "documents": [
                {
                    "bucket": "large",
                    "index": 1,
                    "key": "raw/doc.hwpx",
                    "source_uri": "s3://clic/raw/raw/doc.hwpx",
                    "filename": "large-01-doc.hwpx",
                    "path": "tests/fixtures/corpus/hwpx_size_samples/large/doc.hwpx",
                    "bytes": 1234,
                    "sha256": "doc-sha",
                    "validation": {
                        "ok": True,
                        "elapsed_seconds": 0.1,
                        "units": {"total": 3, "by_type": {"diagram": 1}},
                        "assets": 2,
                        "uploaded_assets": [{"id": "img-0001"}],
                        "quality_warnings": [],
                        "html_uri": "s3://bucket/doc.html",
                        "html_url": "http://example.test/doc.html",
                        "local_html": "/tmp/doc.html",
                        "screenshots": [{"url": "http://example.test/shot.png"}],
                    },
                }
            ],
        }
    )

    assert "started_at" not in manifest
    assert "index_url" not in manifest
    assert manifest["documents"][0]["expected"] == {
        "units": {"total": 3, "by_type": {"diagram": 1}},
        "assets": 2,
        "quality_warnings": [],
    }
    assert "validation" not in manifest["documents"][0]


def test_screenshot_targets_include_representative_rendered_outputs():
    scan_hwpx = _load_scan_script()

    targets = scan_hwpx._screenshot_targets(
        {
            "documents": [
                {
                    "bucket": "small",
                    "index": 1,
                    "validation": {
                        "ok": True,
                        "units": {"by_type": {"table": 1}},
                        "assets": 0,
                        "html_url": "http://example.test/small.html",
                    },
                },
                {
                    "bucket": "large",
                    "index": 1,
                    "validation": {
                        "ok": True,
                        "units": {"by_type": {"diagram": 1}},
                        "assets": 0,
                        "html_url": "http://example.test/diagram.html",
                    },
                },
                {
                    "bucket": "large",
                    "index": 2,
                    "validation": {
                        "ok": True,
                        "units": {"by_type": {"image": 2, "table": 1}},
                        "assets": 2,
                        "html_url": "http://example.test/image.html",
                    },
                },
            ]
        }
    )

    assert targets == [
        {
            "bucket": "small",
            "index": 1,
            "label": "table",
            "selector": ".evidence-table",
            "url": "http://example.test/small.html",
        },
        {
            "bucket": "large",
            "index": 1,
            "label": "diagram",
            "selector": ".diagram-positioned",
            "url": "http://example.test/diagram.html",
        },
        {
            "bucket": "large",
            "index": 2,
            "label": "asset",
            "selector": ".nested-evidence, figure",
            "url": "http://example.test/image.html",
        },
        {
            "bucket": "large",
            "index": 2,
            "label": "table",
            "selector": ".evidence-table",
            "url": "http://example.test/image.html",
        },
    ]


def test_asset_screenshot_lookup_prefers_nested_evidence_by_area():
    scan_hwpx = _load_scan_script()

    lookup = scan_hwpx._element_lookup(
        {"label": "asset", "selector": ".nested-evidence, figure"}
    )

    assert lookup["mode"] == "script"
    assert ".nested-evidence" in lookup["script"]
    assert "nested.length ? nested : candidates" in lookup["script"]
    assert "getBoundingClientRect" in lookup["script"]


def _load_scan_script():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "scan_hwpx_clic_minio_size_samples.py"
    )
    spec = importlib.util.spec_from_file_location(
        "scan_hwpx_clic_minio_size_samples",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
