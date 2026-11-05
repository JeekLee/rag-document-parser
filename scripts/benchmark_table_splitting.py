"""Measure deterministic table splitting without LLM calls.

Run with an XLSX path and --rows 100 200 400 2594. The largest table is
measured alone by default; --all-units includes the other sheets unchanged.
Use PYTHONPATH to select a baseline checkout and compare output_sha256 values.
Parsing, fallback construction, and output serialization are outside the timer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
from pathlib import Path
from time import perf_counter

from rag_document_parser import XlsxBackend
from rag_document_parser.chunk.agentic import _improve_chunk_boundaries
from rag_document_parser.chunk.materialization import ChunkPlanMaterializer
from rag_document_parser.chunk.tokens import _token_encoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xlsx", type=Path)
    parser.add_argument("--rows", type=int, nargs="+", required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--all-units", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeat < 1 or min(args.rows) < 1:
        parser.error("row counts and repeat must be positive")

    data = args.xlsx.read_bytes()
    units = XlsxBackend().parse(data, ".xlsx").units
    largest = max((u for u in units if u.format == "structured_table"),
                  key=lambda u: len(u.content.rows))
    encoder = _token_encoder()
    report = {
        "input_sha256": hashlib.sha256(data).hexdigest(),
        "python_version": platform.python_version(),
        "tokenizer": "cl100k_base" if encoder else "estimated",
        "all_units": args.all_units,
        "target_tokens_per_chunk": 800,
        "max_tokens_per_chunk": 2048,
        "largest_table_rows": len(largest.content.rows),
        "measurements": [],
    }
    for requested in args.rows:
        table = largest.content.model_copy(update={"rows": largest.content.rows[:requested]})
        limited = largest.model_copy(update={"content": table})
        selected = [limited if u is largest else u for u in units] if args.all_units else [limited]
        fallback = ChunkPlanMaterializer(10).fallback(selected, "benchmark plan failure")
        times = []
        for _ in range(args.repeat):
            start = perf_counter()
            chunks = _improve_chunk_boundaries(fallback, selected, 10, 800, 2048)
            times.append(perf_counter() - start)
        output = json.dumps([c.to_dict() for c in chunks], ensure_ascii=False,
                            sort_keys=True, separators=(",", ":")).encode()
        measurement = {
            "rows": len(table.rows), "seconds": times,
            "median_seconds": statistics.median(times), "chunks": len(chunks),
            "output_sha256": hashlib.sha256(output).hexdigest(),
        }
        report["measurements"].append(measurement)
        print(json.dumps(measurement), flush=True)
        if args.output:
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
