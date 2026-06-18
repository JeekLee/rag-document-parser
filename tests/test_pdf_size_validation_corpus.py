from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "corpus" / "pdf-size-validation"
MANIFEST_PATH = FIXTURE_DIR / "manifest.json"


def test_pdf_size_validation_corpus_is_pinned_by_hash_and_size():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    documents = manifest["documents"]

    assert manifest["source"]["pdf_count"] == 984
    assert len(documents) == 20
    assert Counter(document["band"] for document in documents) == {
        "small": 5,
        "medium": 5,
        "large": 5,
        "xlarge": 5,
    }
    assert manifest["source"]["outliers_above_largest_band"] == [
        {
            "key": "20071002-2-0001/본문내용.pdf",
            "bytes": 91070704,
        }
    ]

    for document in documents:
        path = FIXTURE_DIR / document["path"]
        assert path.is_file(), document["id"]
        assert path.suffix == ".pdf", document["id"]
        data = path.read_bytes()
        assert len(data) == document["bytes"], document["id"]
        assert hashlib.sha256(data).hexdigest() == document["sha256"], document["id"]
