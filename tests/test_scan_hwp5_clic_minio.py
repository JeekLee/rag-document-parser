from __future__ import annotations

import importlib.util
from pathlib import Path


def test_table_profile_counts_cells_spans_and_blank_ratio():
    scan_hwp5 = _load_scan_script()

    profile = scan_hwp5._table_profile(
        {
            "id": "b12",
            "metadata": {"table": {"table_id": "t7"}},
            "evidence": {
                "content": {
                    "columns": [{"id": "c1"}, {"id": "c2"}, {"id": "c3"}],
                    "header_rows": [
                        {
                            "cells": [
                                {"text": "구분", "rowspan": 2, "colspan": 1},
                                {"text": "현행", "rowspan": 1, "colspan": 2},
                            ]
                        }
                    ],
                    "rows": [
                        {
                            "cells": [
                                {"text": "A", "rowspan": 1, "colspan": 1},
                                {"text": "", "rowspan": 1, "colspan": 1},
                                {"text": "C", "rowspan": 1, "colspan": 1},
                            ]
                        },
                        {
                            "cells": [
                                {"text": "", "rowspan": 1, "colspan": 1},
                                {"text": "", "rowspan": 1, "colspan": 1},
                                {"text": "F", "rowspan": 1, "colspan": 1},
                            ]
                        },
                    ],
                }
            },
        }
    )

    assert profile == {
        "unit_id": "b12",
        "table_id": "t7",
        "columns": 3,
        "header_rows": 1,
        "rows": 2,
        "cells": 8,
        "blank_cells": 3,
        "blank_ratio": 0.375,
        "span_cells": 2,
        "score": 12,
        "flags": [],
    }


def test_diagram_profile_counts_geometry_edges_and_labels():
    scan_hwp5 = _load_scan_script()

    profile = scan_hwp5._diagram_profile(
        {
            "id": "b9",
            "type": "diagram",
            "source": {
                "text": "수급권자\n심사평가원\nrelations:\nn1 -> n2: ①신청"
            },
            "evidence": {
                "content": {
                    "nodes": [
                        {
                            "id": "n1",
                            "text": "수급권자",
                            "bbox": {"x": 0, "y": 0, "width": 10, "height": 10},
                        },
                        {"id": "n2", "text": "①신청", "bbox": None},
                        {
                            "id": "n3",
                            "text": "심사평가원",
                            "bbox": {"x": 30, "y": 0, "width": 10, "height": 10},
                        },
                    ],
                    "connectors": [
                        {"id": "c1", "points": [{"x": 10, "y": 5}, {"x": 30, "y": 5}]},
                        {"id": "c2", "points": [{"x": 30, "y": 5}, {"x": 40, "y": 5}]},
                    ],
                    "edges": [
                        {
                            "from": "n1",
                            "to": "n3",
                            "label": "①신청",
                            "confidence": "inferred_geometry",
                            "connector_id": "c1",
                        },
                        {
                            "from": "n3",
                            "to": "n1",
                            "label": "",
                            "confidence": "inferred_geometry",
                            "connector_id": "c2",
                        },
                    ],
                }
            },
        }
    )

    assert profile == {
        "unit_id": "b9",
        "nodes": 3,
        "bbox_nodes": 2,
        "bbox_node_ratio": 0.667,
        "connectors": 2,
        "edges": 2,
        "labeled_edges": 1,
        "unlabeled_edges": 1,
        "source_relation_lines": 1,
        "connector_edge_ratio": 1.0,
        "score": 20,
        "flags": [
            "diagram_connectors",
            "inferred_edges",
            "unlabeled_edges",
        ],
    }


def test_document_scan_summary_ranks_table_outliers():
    scan_hwp5 = _load_scan_script()

    document = scan_hwp5._document_summary(
        source_uri="s3://clic/raw/sample.hwp",
        raw_bytes=1200,
        elapsed_seconds=0.25,
        units=[
            {"id": "b1", "type": "text", "evidence": {"content": "본문"}},
            _table_unit("b2", "t1", columns=5, rows=3, cells_per_row=5),
            _table_unit("b3", "t2", columns=120, rows=2, cells_per_row=120),
        ],
        assets=[],
        warnings=[{"type": "hwp5_drawing_structure_unsupported"}],
    )

    assert document["source_uri"] == "s3://clic/raw/sample.hwp"
    assert document["unit_counts"] == {"table": 2, "text": 1}
    assert document["tables"]["count"] == 2
    assert document["tables"]["total_cells"] == 255
    assert document["tables"]["max_columns"] == 120
    assert document["tables"]["outliers"][0]["table_id"] == "t2"
    assert document["tables"]["outliers"][0]["flags"] == ["wide_table"]
    assert document["warning_types"] == ["hwp5_drawing_structure_unsupported"]


def test_document_scan_summary_reports_diagram_outliers():
    scan_hwp5 = _load_scan_script()

    document = scan_hwp5._document_summary(
        source_uri="s3://clic/raw/diagram.hwp",
        raw_bytes=900,
        elapsed_seconds=0.1,
        units=[
            {"id": "b1", "type": "text", "evidence": {"content": "본문"}},
            _diagram_unit("b2", nodes=4, bbox_nodes=2, connectors=3, edges=2),
        ],
        assets=[],
        warnings=[],
    )

    assert document["unit_counts"] == {"diagram": 1, "text": 1}
    assert document["diagrams"]["count"] == 1
    assert document["diagrams"]["total_nodes"] == 4
    assert document["diagrams"]["total_connectors"] == 3
    assert document["diagrams"]["total_edges"] == 2
    assert document["diagrams"]["outliers"][0]["unit_id"] == "b2"


def test_corpus_summary_ranks_diagram_outliers():
    scan_hwp5 = _load_scan_script()

    summary = scan_hwp5._corpus_summary(
        [
            {
                "source_uri": "s3://clic/raw/a.hwp",
                "unit_counts": {"diagram": 1},
                "warning_types": [],
                "tables": {"count": 0, "total_cells": 0, "outliers": []},
                "diagrams": {
                    "count": 1,
                    "total_nodes": 4,
                    "total_connectors": 4,
                    "total_edges": 1,
                    "outliers": [
                        {
                            "unit_id": "b1",
                            "nodes": 4,
                            "score": 15,
                            "flags": ["connector_without_edges"],
                        }
                    ],
                },
            }
        ],
        top=5,
    )

    assert summary["total_diagrams"] == 1
    assert summary["total_diagram_edges"] == 1
    assert summary["top_diagram_outliers"] == [
        {
            "source_uri": "s3://clic/raw/a.hwp",
            "unit_id": "b1",
            "nodes": 4,
            "score": 15,
            "flags": ["connector_without_edges"],
        }
    ]


def test_mc_path_to_s3_uri_converts_alias_bucket_and_key():
    scan_hwp5 = _load_scan_script()

    assert (
        scan_hwp5._mc_path_to_s3_uri(
            "local/clic/raw/20230329-12-0001/(제2023-56호) sample.hwp"
        )
        == "s3://clic/raw/20230329-12-0001/(제2023-56호) sample.hwp"
    )


def test_read_mc_path_streams_object_bytes_from_stdout(monkeypatch):
    scan_hwp5 = _load_scan_script()
    calls = []

    class Completed:
        stdout = b"hwp-bytes"
        stderr = b""

    def fake_run(command, check, stdout, stderr):
        calls.append(
            {
                "command": command,
                "check": check,
                "stdout": stdout,
                "stderr": stderr,
            }
        )
        return Completed()

    monkeypatch.setattr(scan_hwp5.subprocess, "run", fake_run)

    raw = scan_hwp5._read_mc_path(
        "local/clic/raw/sample.hwp",
        mc_command="docker exec clic-minio mc",
    )

    assert raw == b"hwp-bytes"
    assert calls == [
        {
            "command": [
                "docker",
                "exec",
                "clic-minio",
                "mc",
                "cat",
                "local/clic/raw/sample.hwp",
            ],
            "check": True,
            "stdout": scan_hwp5.subprocess.PIPE,
            "stderr": scan_hwp5.subprocess.PIPE,
        }
    ]


def test_hwp5_signature_detection_requires_ole_compound_header():
    scan_hwp5 = _load_scan_script()

    assert scan_hwp5._has_hwp5_container_signature(
        b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1rest"
    )
    assert not scan_hwp5._has_hwp5_container_signature(b"PK\x03\x04zip")
    assert not scan_hwp5._has_hwp5_container_signature(b"<html></html>")


def test_scan_mc_path_skips_non_hwp5_signature_without_parsing(monkeypatch):
    scan_hwp5 = _load_scan_script()

    class FailingBackend:
        def parse(self, data, suffix):
            raise AssertionError("parser should not be called")

    monkeypatch.setattr(
        scan_hwp5,
        "_read_mc_path",
        lambda mc_path, *, mc_command: b"PK\x03\x04not-hwp5",
    )
    monkeypatch.setattr(scan_hwp5, "Hwp5Backend", FailingBackend)

    document = scan_hwp5._scan_mc_path(
        "local/clic/raw/not-really-hwp5.hwp",
        mc_command="docker exec clic-minio mc",
    )

    assert document["source_uri"] == "s3://clic/raw/not-really-hwp5.hwp"
    assert document["skipped"] is True
    assert document["skip_reason"] == "non_hwp5_signature"
    assert document["warning_types"] == ["non_hwp5_skipped"]
    assert document["unit_counts"] == {}
    assert document["tables"]["count"] == 0
    assert document["diagrams"]["count"] == 0


def _table_unit(
    unit_id: str,
    table_id: str,
    *,
    columns: int,
    rows: int,
    cells_per_row: int,
) -> dict[str, object]:
    return {
        "id": unit_id,
        "type": "table",
        "metadata": {"table": {"table_id": table_id}},
        "evidence": {
            "content": {
                "columns": [{"id": f"c{index}"} for index in range(1, columns + 1)],
                "header_rows": [],
                "rows": [
                    {
                        "cells": [
                            {"text": f"r{row}c{cell}", "rowspan": 1, "colspan": 1}
                            for cell in range(cells_per_row)
                        ]
                    }
                    for row in range(rows)
                ],
            }
        },
    }


def _diagram_unit(
    unit_id: str,
    *,
    nodes: int,
    bbox_nodes: int,
    connectors: int,
    edges: int,
) -> dict[str, object]:
    node_payload = [
        {
            "id": f"n{index}",
            "text": f"node {index}",
            "bbox": (
                {"x": index * 10, "y": 0, "width": 5, "height": 5}
                if index <= bbox_nodes
                else None
            ),
        }
        for index in range(1, nodes + 1)
    ]
    return {
        "id": unit_id,
        "type": "diagram",
        "source": {
            "text": "\n".join(
                [
                    *(f"node {index}" for index in range(1, nodes + 1)),
                    "relations:",
                    *(f"n{index} -> n{index + 1}" for index in range(1, edges + 1)),
                ]
            )
        },
        "evidence": {
            "content": {
                "nodes": node_payload,
                "connectors": [
                    {
                        "id": f"c{index}",
                        "points": [
                            {"x": index * 10, "y": 1},
                            {"x": index * 10 + 5, "y": 1},
                        ],
                    }
                    for index in range(1, connectors + 1)
                ],
                "edges": [
                    {
                        "from": f"n{index}",
                        "to": f"n{index + 1}",
                        "label": "",
                        "confidence": "inferred_geometry",
                        "connector_id": f"c{index}",
                    }
                    for index in range(1, edges + 1)
                ],
            }
        },
    }


def _load_scan_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "scan_hwp5_clic_minio.py"
    spec = importlib.util.spec_from_file_location("scan_hwp5_clic_minio", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
