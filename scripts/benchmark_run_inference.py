"""Benchmark a completed SuperADD checkpoint; this is not a validated deployment export."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

script_directory = Path(__file__).resolve().parent
package_root = script_directory if (script_directory / "app").is_dir() else script_directory.parent
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))

from app.core.inference_benchmark import (
    BenchmarkCancelled,
    BenchmarkMode,
    BenchmarkRequest,
    CheckpointBenchmarkRunner,
    write_benchmark_csv,
    write_benchmark_json,
)


def main() -> int:
    """Run the in-memory batch-one checkpoint benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark a completed SuperADD checkpoint; not a validated deployment export.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--mode", choices=tuple(mode.value for mode in BenchmarkMode), default=BenchmarkMode.CAMERA_EQUIVALENT.value)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--reserve-percent", type=float, default=20.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path)
    args = parser.parse_args()
    try:
        runner = CheckpointBenchmarkRunner.load_completed_run(args.run_dir, args.device)
        result = runner.run(
            args.input,
            BenchmarkRequest(
                mode=BenchmarkMode(args.mode),
                warmup_frames=args.warmup,
                measured_frames=args.iterations,
                target_fps=args.target_fps,
                reserve_percent=args.reserve_percent,
            ),
            progress=lambda current, total: print(f"Measured frame {current}/{total}", flush=True),
        )
        write_benchmark_json(args.output, result)
        if args.csv_output is not None:
            write_benchmark_csv(args.csv_output, result)
        print(f"Wrote checkpoint benchmark to {args.output}")
        print("This benchmark is not a validated deployment export.")
        return 0
    except (BenchmarkCancelled, KeyboardInterrupt) as exc:
        print(str(exc) or "Industrial inference benchmark cancelled.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())