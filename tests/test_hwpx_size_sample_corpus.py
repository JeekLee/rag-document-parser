from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from rag_document_parser import HwpxBackend


def test_hwpx_size_sample_corpus_matches_manifest_counts():
    manifest_path = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "corpus"
        / "hwpx_size_samples"
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents = manifest["documents"]

    assert len(documents) == 15
    assert Counter(document["bucket"] for document in documents) == {
        "small": 5,
        "medium": 5,
        "large": 5,
    }
    assert any(
        "diagram" in document["expected"]["units"]["by_type"]
        for document in documents
    )
    assert any(document["expected"]["assets"] > 0 for document in documents)
    assert all("validation" not in document for document in documents)

    backend = HwpxBackend()
    repo_root = Path(__file__).resolve().parents[1]
    for document in documents:
        sample_path = repo_root / document["path"]
        raw = sample_path.read_bytes()
        parsed = backend.parse(raw, ".hwpx")
        unit_counts = Counter(unit.type for unit in parsed.units)
        expected = document["expected"]

        assert len(raw) == document["bytes"], document["key"]
        assert len(parsed.units) == expected["units"]["total"], document["key"]
        assert dict(sorted(unit_counts.items())) == expected["units"]["by_type"], (
            document["key"]
        )
        assert len(parsed.assets) == expected["assets"], document["key"]
        assert parsed.quality_warnings == expected["quality_warnings"], (
            document["key"]
        )
