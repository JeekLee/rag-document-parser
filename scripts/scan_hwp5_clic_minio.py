from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from rag_document_parser import Hwp5Backend


def main() -> None:
    args = _parse_args()
    started_at = datetime.now().isoformat(timespec="seconds")
    mc_paths = _list_hwp_paths(
        args.source_prefix,
        mc_command=args.mc_command,
        name_pattern=args.name_pattern,
        max_documents=args.max_documents,
    )

    documents: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, mc_path in enumerate(mc_paths, start=1):
        try:
            documents.append(
                _scan_mc_path(
                    mc_path,
                    mc_command=args.mc_command,
                )
            )
        except Exception as exc:  # pragma: no cover - exercised by real scan.
            failures.append(
                {
                    "mc_path": mc_path,
                    "source_uri": _mc_path_to_s3_uri(mc_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if args.progress:
            print(
                json.dumps(
                    {
                        "scanned": index,
                        "total": len(mc_paths),
                        "mc_path": mc_path,
                        "failures": len(failures),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    payload = {
        "started_at": started_at,
        "source_prefix": args.source_prefix,
        "name_pattern": args.name_pattern,
        "documents_scanned": len(documents),
        "documents_failed": len(failures),
        "summary": _corpus_summary(documents, top=args.top),
        "documents": documents,
        "failures": failures,
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(encoded)
    print(encoded.decode("utf-8"))


def _list_hwp_paths(
    source_prefix: str,
    *,
    mc_command: str,
    name_pattern: str,
    max_documents: int | None,
) -> list[str]:
    output = _run_mc(
        mc_command,
        ["find", source_prefix, "--name", name_pattern],
    )
    paths = [line.strip() for line in output.splitlines() if line.strip()]
    if max_documents is not None:
        return paths[:max_documents]
    return paths


def _scan_mc_path(
    mc_path: str,
    *,
    mc_command: str,
) -> dict[str, Any]:
    raw = _read_mc_path(mc_path, mc_command=mc_command)
    started = time.perf_counter()
    parsed = Hwp5Backend().parse(raw, ".hwp")
    elapsed = time.perf_counter() - started
    return _document_summary(
        source_uri=_mc_path_to_s3_uri(mc_path),
        raw_bytes=len(raw),
        elapsed_seconds=elapsed,
        units=[unit.to_dict() for unit in parsed.units],
        assets=[asset.__dict__ for asset in parsed.assets],
        warnings=parsed.quality_warnings,
    )


def _read_mc_path(mc_path: str, *, mc_command: str) -> bytes:
    command = [*shlex.split(mc_command), "cat", mc_path]
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def _document_summary(
    *,
    source_uri: str,
    raw_bytes: int,
    elapsed_seconds: float,
    units: list[dict[str, Any]],
    assets: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    unit_counts = Counter(str(unit.get("type", "")) for unit in units)
    table_profiles = [
        _table_profile(unit)
        for unit in units
        if unit.get("type") == "table"
    ]
    outliers = sorted(table_profiles, key=lambda table: table["score"], reverse=True)
    return {
        "source_uri": source_uri,
        "bytes": raw_bytes,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "unit_counts": dict(sorted(unit_counts.items())),
        "asset_count": len(assets),
        "warning_types": sorted(
            {
                str(warning.get("type"))
                for warning in warnings
                if warning.get("type")
            }
        ),
        "tables": {
            "count": len(table_profiles),
            "total_cells": sum(table["cells"] for table in table_profiles),
            "max_columns": max(
                (table["columns"] for table in table_profiles),
                default=0,
            ),
            "max_cells": max((table["cells"] for table in table_profiles), default=0),
            "outliers": [
                table
                for table in outliers
                if table["flags"]
            ][:10],
        },
    }


def _table_profile(unit: dict[str, Any]) -> dict[str, Any]:
    content = unit.get("evidence", {}).get("content", {})
    columns = content.get("columns", [])
    header_rows = content.get("header_rows", [])
    data_rows = content.get("rows", [])
    rows = [*header_rows, *data_rows]
    cells = [
        cell
        for row in rows
        for cell in row.get("cells", [])
        if isinstance(cell, dict)
    ]
    blank_cells = sum(1 for cell in cells if not _cell_has_content(cell))
    span_cells = sum(
        1
        for cell in cells
        if int(cell.get("rowspan", 1) or 1) > 1
        or int(cell.get("colspan", 1) or 1) > 1
    )
    column_count = len(columns) if isinstance(columns, list) else 0
    cell_count = len(cells)
    blank_ratio = round(blank_cells / cell_count, 3) if cell_count else 0.0
    nonblank_cells = cell_count - blank_cells
    flags = _table_flags(
        columns=column_count,
        cells=cell_count,
        blank_ratio=blank_ratio,
        span_cells=span_cells,
    )
    return {
        "unit_id": str(unit.get("id", "")),
        "table_id": str(unit.get("metadata", {}).get("table", {}).get("table_id", "")),
        "columns": column_count,
        "header_rows": len(header_rows) if isinstance(header_rows, list) else 0,
        "rows": len(data_rows) if isinstance(data_rows, list) else 0,
        "cells": cell_count,
        "blank_cells": blank_cells,
        "blank_ratio": blank_ratio,
        "span_cells": span_cells,
        "score": column_count + nonblank_cells + span_cells * 2,
        "flags": flags,
    }


def _cell_has_content(cell: dict[str, Any]) -> bool:
    text = str(cell.get("text", "")).strip()
    children = cell.get("children", [])
    return bool(text or children)


def _table_flags(
    *,
    columns: int,
    cells: int,
    blank_ratio: float,
    span_cells: int,
) -> list[str]:
    flags: list[str] = []
    if columns >= 50:
        flags.append("wide_table")
    if cells >= 1000:
        flags.append("large_table")
    if cells >= 50 and blank_ratio >= 0.85:
        flags.append("sparse_table")
    if span_cells >= 100:
        flags.append("span_heavy_table")
    return flags


def _corpus_summary(documents: list[dict[str, Any]], *, top: int) -> dict[str, Any]:
    unit_counts: Counter[str] = Counter()
    warning_types: Counter[str] = Counter()
    table_outliers: list[dict[str, Any]] = []
    for document in documents:
        unit_counts.update(document.get("unit_counts", {}))
        warning_types.update(document.get("warning_types", []))
        for table in document.get("tables", {}).get("outliers", []):
            table_outliers.append(
                {
                    "source_uri": document["source_uri"],
                    **table,
                }
            )
    table_outliers.sort(key=lambda table: table["score"], reverse=True)
    return {
        "unit_counts": dict(sorted(unit_counts.items())),
        "warning_types": dict(sorted(warning_types.items())),
        "total_tables": sum(document.get("tables", {}).get("count", 0) for document in documents),
        "total_table_cells": sum(
            document.get("tables", {}).get("total_cells", 0)
            for document in documents
        ),
        "top_table_outliers": table_outliers[:top],
    }


def _mc_path_to_s3_uri(mc_path: str) -> str:
    alias, bucket, key = mc_path.split("/", 2)
    if not alias or not bucket or not key:
        raise ValueError(f"expected mc path '<alias>/<bucket>/<key>': {mc_path}")
    return f"s3://{bucket}/{key}"


def _run_mc(mc_command: str, args: list[str]) -> str:
    command = [*shlex.split(mc_command), *args]
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan clic MinIO HWP5 corpus for table extraction outliers.",
    )
    parser.add_argument(
        "--source-prefix",
        default=os.getenv("RDP_SCAN_SOURCE_PREFIX", "local/clic/raw"),
    )
    parser.add_argument(
        "--mc-command",
        default=os.getenv("RDP_MC_COMMAND", "docker exec clic-minio mc"),
    )
    parser.add_argument("--name-pattern", default="*.hwp")
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
