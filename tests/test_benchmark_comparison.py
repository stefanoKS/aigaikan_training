"""Industrial benchmark comparison tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.core.benchmark_comparison import compare_benchmark_documents, write_benchmark_comparison_csv


def _write(path: Path, *, input_manifest: str, target_fps: float) -> Path:
    payload = {
        "benchmark_version": 1,
        "metadata": {
            "backbone": "vit_small_plus_patch16_dinov3.lvd1689m",
            "model_precision": "float16",
            "checkpoint_sha256": "checkpoint",
            "memory_bank_shape": [100, 384],
            "memory_bank_dtype": "torch.float16",
            "input_manifest_sha256": input_manifest,
            "roi_hash": "roi",
            "preprocessing_hash": "preprocessing",
            "prepared_canvas_size": [448, 448],
            "warmup_count": 20,
            "measured_count": 200,
            "peak_cuda_memory_allocated": 123,
            "peak_cuda_memory_reserved": 456,
        },
        "timing": {
            "preprocess_total_ms": {"p95_ms": 3.0},
            "model_forward_ms": {"p95_ms": 40.0},
            "end_to_end_compute_ms": {"p95_ms": 45.0},
        },
        "measured_steady_state_fps": 20.0,
        "conservative_p95_fps": 22.2,
        "deadline": {"target_fps": target_fps, "pass": True},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_benchmark_comparison_warns_for_incompatible_conditions_and_writes_csv(tmp_path: Path) -> None:
    first = _write(tmp_path / "first.json", input_manifest="same", target_fps=10)
    second = _write(tmp_path / "second.json", input_manifest="different", target_fps=15)

    rows, warnings = compare_benchmark_documents([first, second])
    csv_path = write_benchmark_comparison_csv(tmp_path / "comparison.csv", rows)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))

    assert len(rows) == 2
    assert any("input_manifest_sha256" in warning for warning in warnings)
    assert any("target_fps" in warning for warning in warnings)
    assert row["backbone"] == "vit_small_plus_patch16_dinov3.lvd1689m"