"""Comparison of portable industrial inference benchmark documents."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping

from app.core.inference_benchmark import read_benchmark_json


def compare_benchmark_documents(paths: list[Path]) -> tuple[list[dict[str, object]], list[str]]:
    """Return flat comparison rows and warnings for incompatible benchmark conditions."""
    documents = [(path, read_benchmark_json(path)) for path in paths]
    rows = [_comparison_row(path, document) for path, document in documents]
    warnings: list[str] = []
    if documents:
        reference_path, reference = documents[0]
        for path, document in documents[1:]:
            for field in _COMPARISON_CONDITIONS:
                left = _nested_value(reference, field)
                right = _nested_value(document, field)
                if left != right:
                    warnings.append(
                        f"{path.name} differs from {reference_path.name} for {field.replace('.', ' ')}: {right!r} != {left!r}."
                    )
    return rows, warnings


def write_benchmark_comparison_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    """Write benchmark rows to a stable, spreadsheet-friendly comparison CSV."""
    if not rows:
        raise ValueError("At least one benchmark result is required for comparison.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


_COMPARISON_CONDITIONS = (
    "metadata.input_manifest_sha256",
    "metadata.roi_hash",
    "metadata.preprocessing_hash",
    "metadata.prepared_canvas_size",
    "metadata.warmup_count",
    "metadata.measured_count",
    "deadline.target_fps",
)


def _comparison_row(path: Path, document: Mapping[str, object]) -> dict[str, object]:
    metadata = document.get("metadata", {})
    timing = document.get("timing", {})
    deadline = document.get("deadline", {})
    if not isinstance(metadata, Mapping) or not isinstance(timing, Mapping) or not isinstance(deadline, Mapping):
        raise ValueError(f"Malformed benchmark document: {path}")
    return {
        "source": str(path),
        "backbone": metadata.get("backbone", ""),
        "precision": metadata.get("model_precision", ""),
        "checkpoint_sha256": metadata.get("checkpoint_sha256", ""),
        "memory_bank_shape": json.dumps(metadata.get("memory_bank_shape", [])),
        "memory_bank_dtype": metadata.get("memory_bank_dtype", ""),
        "preprocessing_p95_ms": _timing_value(timing, "preprocess_total_ms", "p95_ms"),
        "model_forward_p95_ms": _timing_value(timing, "model_forward_ms", "p95_ms"),
        "end_to_end_compute_p95_ms": _timing_value(timing, "end_to_end_compute_ms", "p95_ms"),
        "measured_steady_state_fps": document.get("measured_steady_state_fps", ""),
        "conservative_p95_fps": document.get("conservative_p95_fps", ""),
        "target_pass": deadline.get("pass", ""),
        "peak_cuda_memory_allocated": metadata.get("peak_cuda_memory_allocated", ""),
        "peak_cuda_memory_reserved": metadata.get("peak_cuda_memory_reserved", ""),
    }


def _timing_value(timing: Mapping[str, object], phase: str, value: str) -> object:
    summary = timing.get(phase, {})
    return summary.get(value, "") if isinstance(summary, Mapping) else ""


def _nested_value(payload: Mapping[str, object], path: str) -> object:
    value: object = payload
    for key in path.split("."):
        value = value.get(key) if isinstance(value, Mapping) else None
    return value