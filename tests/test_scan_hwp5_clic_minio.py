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


def _load_scan_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "scan_hwp5_clic_minio.py"
    spec = importlib.util.spec_from_file_location("scan_hwp5_clic_minio", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
