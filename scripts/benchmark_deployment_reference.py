"""Benchmark verified Torch deployment reference inference without artifact I/O."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter_ns

script_directory = Path(__file__).resolve().parent
package_root = script_directory if (script_directory / "app").is_dir() else script_directory.parent
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))

import numpy as np
from PIL import Image

from app.core.deployment_reference import TorchDeploymentReferenceInferencer
from app.core.inference_timing import timing_percentiles
from app.models.dataset_config import SUPPORTED_IMAGE_EXTENSIONS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmup < 10 or args.frames < 100:
        parser.error("--warmup must be at least 10 and --frames must be at least 100")
    if args.batch_size != 1:
        parser.error("The reference benchmark currently measures true batch-one latency only.")
    paths = _image_paths(args.input)
    if not paths:
        parser.error("input must be one image or a non-empty folder of supported images")
    frames = [_read_rgb(paths[index % len(paths)]) for index in range(args.frames + args.warmup)]
    reference = TorchDeploymentReferenceInferencer.load(args.package, device=args.device)
    for frame in frames[: args.warmup]:
        reference.infer_rgb(frame)
    measured = []
    started = perf_counter_ns()
    for frame in frames[args.warmup :]:
        measured.append(reference.infer_rgb(frame))
    wall_ms = (perf_counter_ns() - started) / 1_000_000
    inference_latencies = [float(item.timing.inference_total_ms or 0) for item in measured]
    end_to_end_latencies = [float(item.timing.end_to_end_ms or 0) for item in measured]
    output = {
        "benchmark_version": 1,
        "warmup_frames": args.warmup,
        "measured_frames": args.frames,
        "batch_size": args.batch_size,
        "device": args.device,
        "model_inference": timing_percentiles(inference_latencies),
        "end_to_end": timing_percentiles(end_to_end_latencies),
        "throughput_images_per_second": args.frames * 1000 / wall_ms if wall_ms else 0.0,
        "artifact_io_ms": 0.0,
        "input_decode_ms": "not included; inputs decoded before timed measurements",
        "raw_input_size": list(frames[args.warmup].shape[1::-1]),
        "rectified_size": list(measured[0].timing.rectified_size),
        "model_input_size": list(measured[0].timing.model_input_size),
        "tile_count": measured[0].timing.tile_count,
    }
    rendered = json.dumps(output, indent=2)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


def _image_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    return sorted(
        (item for item in path.iterdir() if item.is_file() and item.suffix.casefold() in SUPPORTED_IMAGE_EXTENSIONS),
        key=lambda item: item.name.casefold(),
    )


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))