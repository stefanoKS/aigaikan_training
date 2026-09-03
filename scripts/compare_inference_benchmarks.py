"""Compare portable industrial checkpoint benchmark JSON documents."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

script_directory = Path(__file__).resolve().parent
package_root = script_directory if (script_directory / "app").is_dir() else script_directory.parent
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))

from app.core.benchmark_comparison import compare_benchmark_documents, write_benchmark_comparison_csv


def main() -> int:
    """Write a CSV comparison and report incompatible comparison conditions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmarks", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        rows, warnings = compare_benchmark_documents(args.benchmarks)
        write_benchmark_comparison_csv(args.output, rows)
    except Exception as exc:
        print(f"Benchmark comparison failed: {exc}", file=sys.stderr)
        return 1
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    print(f"Wrote benchmark comparison to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())