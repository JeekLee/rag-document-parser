from __future__ import annotations

import hashlib
import json
from pathlib import Path


SAMPLE_DIR = (
    Path(__file__).parent
    / "fixtures"
    / "corpus"
    / "hwp"
    / "minio-size-samples"
)
MANIFEST_PATH = SAMPLE_DIR / "manifest.json"
HWP5_OLE_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


def _sample_documents() -> list[dict[str, object]]:
    assert MANIFEST_PATH.is_file()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return manifest["documents"]


def test_hwp5_minio_size_sample_fixtures_are_pinned_by_hash_and_size():
    documents = _sample_documents()

    assert len(documents) == 15
    assert {
        bucket: sum(1 for document in documents if document["size_bucket"] == bucket)
        for bucket in {"small", "medium", "large"}
    } == {"small": 5, "medium": 5, "large": 5}

    for document in documents:
        path = SAMPLE_DIR / str(document["path"])
        assert path.is_file(), document["id"]
        data = path.read_bytes()
        assert data.startswith(HWP5_OLE_SIGNATURE), document["id"]
        assert len(data) == document["bytes"], document["id"]
        assert hashlib.sha256(data).hexdigest() == document["sha256"], document["id"]
