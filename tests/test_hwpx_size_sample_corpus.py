from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

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


@pytest.mark.parametrize(
    ("prefix", "markers"),
    [
        (
            "large-01",
            {
                "고가의약품 급여관리에 대한 내용": 1,
                "고가약 관리 시스템에 대한 내용": 1,
                "명세서 청구 등에 대한 내용": 1,
                "높은 가격, 효과의 불확실성 등으로 인하여 가격관리 및 장기효과 확인이 필요한 약제": 1,
                "요양기관에서 고가의약품 투여 후 약제의 효과 및 부작용 등의 변화에 대하여": 1,
                "성과평가 결과는 제약사의 위험분담 환급액 산출을 위한 데이터로 활용됩니다": 1,
            },
        ),
        (
            "large-04",
            {
                "□ 청구방법": 2,
                "※ 각 행위에 대한 검사항목은 해당 학회에서 제출한 행위정의 내용에 포함된 내용임": 1,
            },
        ),
    ],
)
def test_hwpx_fixtures_preserve_text_beside_objects_and_in_grouped_drawings(prefix, markers):
    directory = Path(__file__).parent / "fixtures" / "corpus" / "hwpx_size_samples" / "large"
    path = next(directory.glob(f"{prefix}-*.hwpx"))
    parsed = HwpxBackend().parse(path.read_bytes(), ".hwpx")
    source = " ".join(" ".join(unit.source.text.split()) for unit in parsed.units)

    for marker, count in markers.items():
        assert source.count(marker) == count, marker
