"""Batch-one industrial checkpoint benchmark tests."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.core.inference_benchmark import (
    BenchmarkCancelled,
    BenchmarkMode,
    BenchmarkRequest,
    CheckpointBenchmarkRunner,
    assess_industrial_deadline,
    read_benchmark_json,
    write_benchmark_csv,
    write_benchmark_json,
)
from app.core.inference_benchmark import _load_checkpoint_model
from app.core.model_registry import ModelRegistry
from app.core.prediction_contract import SUPERADD_NATIVE_IMAGE_SCORE_SEMANTIC
from app.models.inspection_region import InspectionRegionConfig
from app.models.preprocessing_config import PreprocessingConfig
from app.models.training_config import TrainingConfig
from app.core.preprocessing_pipeline import PreprocessingPipeline


class _PostProcessor:
    def post_process_batch(self, output: dict[str, object]) -> None:
        output["pred_score"] = 0.25


class _Model:
    def __init__(self) -> None:
        self.pre_processor = lambda values: values
        self.post_processor = _PostProcessor()
        self.memory_bank = np.zeros((12, 4), dtype=np.float32)
        self.evaluated = False

    def eval(self) -> "_Model":
        self.evaluated = True
        return self

    def __call__(self, _values: object) -> dict[str, object]:
        return {"pred_score": 2.5, "anomaly_map": np.zeros((1, 1, 4, 4), dtype=np.float32)}


def _runner() -> CheckpointBenchmarkRunner:
    config = TrainingConfig(model_name="super_add")
    plan = PreprocessingConfig().resolve("super_add", (16, 12))
    return CheckpointBenchmarkRunner(
        model=_Model(),
        config=config,
        pipeline=PreprocessingPipeline(InspectionRegionConfig(), plan),
        threshold=2.0,
        threshold_semantic=SUPERADD_NATIVE_IMAGE_SCORE_SEMANTIC,
        device="cpu",
        model_load_ms=12.5,
        metadata={"backbone": config.superadd_backbone_name},
    )


def _images(tmp_path: Path) -> Path:
    directory = tmp_path / "frames"
    directory.mkdir()
    for name, color in (("b.png", (20, 30, 40)), ("a.png", (50, 60, 70))):
        Image.new("RGB", (16, 12), color).save(directory / name)
    return directory


def test_camera_equivalent_benchmark_is_warm_batch_one_and_excludes_decode(tmp_path: Path) -> None:
    result = _runner().run(
        _images(tmp_path),
        BenchmarkRequest(mode=BenchmarkMode.CAMERA_EQUIVALENT, warmup_frames=2, measured_frames=3, target_fps=10),
    )

    assert result.metadata["warmup_count"] == 2
    assert result.metadata["measured_count"] == 3
    assert result.metadata["batch_size"] == 1
    assert result.metadata["input_decode_excluded_from_camera_equivalent"] is True
    assert result.timing["input_decode_ms"]["count"] == 3
    assert result.timing["input_decode_ms"]["maximum_ms"] == 0.0
    assert result.timing["artifact_io_ms"]["maximum_ms"] == 0.0
    assert result.timing["end_to_end_compute_ms"]["count"] == 3
    assert result.metadata["memory_bank_shape"] == [12, 4]
    assert result.metadata["memory_bank_dtype"] == "float32"
    assert result.measured_steady_state_fps > 0
    assert result.conservative_p95_fps > 0


def test_file_mode_includes_decoding_once_without_artifact_io(tmp_path: Path) -> None:
    result = _runner().run(
        _images(tmp_path),
        BenchmarkRequest(mode=BenchmarkMode.FILE_END_TO_END, warmup_frames=0, measured_frames=2),
    )

    assert result.metadata["input_decode_excluded_from_camera_equivalent"] is False
    assert result.timing["input_decode_ms"]["count"] == 2
    assert result.file_source_p95_fps is not None
    assert result.timing["file_source_end_to_end_ms"]["p95_ms"] >= result.timing["end_to_end_compute_ms"]["p95_ms"]
    assert result.timing["artifact_io_ms"]["maximum_ms"] == 0.0


def test_benchmark_cancellation_and_deadline_assessment(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkCancelled, match="cancelled"):
        _runner().run(_images(tmp_path), BenchmarkRequest(measured_frames=1), cancelled=lambda: True)

    passed = assess_industrial_deadline(10, 20, 79)
    failed = assess_industrial_deadline(10, 20, 81)

    assert passed.passed and passed.allowed_compute_budget_ms == 80
    assert not failed.passed and "exceeds" in failed.reason


def test_benchmark_json_and_csv_round_trip(tmp_path: Path) -> None:
    result = _runner().run(_images(tmp_path), BenchmarkRequest(measured_frames=1))
    json_path = write_benchmark_json(tmp_path / "benchmark.json", result)
    csv_path = write_benchmark_csv(tmp_path / "benchmark.csv", result)

    loaded = read_benchmark_json(json_path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))

    assert loaded["benchmark_version"] == 1
    assert row["metadata.backbone"] == "vit_huge_plus_patch16_dinov3"
    assert row["timing.end_to_end_compute_ms.p95_ms"]


def test_checkpoint_loader_constructs_one_model_with_the_saved_superadd_contract(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    class LoadedModel:
        def to(self, _device: object) -> "LoadedModel":
            return self

        def eval(self) -> "LoadedModel":
            return self

    class ModelClass:
        @classmethod
        def load_from_checkpoint(cls, checkpoint: str, **kwargs: object) -> LoadedModel:
            calls.append({"checkpoint": checkpoint, **kwargs})
            return LoadedModel()

    class Service:
        @staticmethod
        def _anomalib_models() -> object:
            return type("Models", (), {"SuperADD": ModelClass})

        @staticmethod
        def _create_v2_pre_processor(_model_class: object, _definition: object, _plan: object) -> str:
            return "frozen-preprocessor"

    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    config = TrainingConfig(model_name="super_add", superadd_backbone_id="vit_base_patch16_dinov3.lvd1689m")
    plan = PreprocessingConfig().resolve("super_add", (16, 12))

    model = _load_checkpoint_model(Service(), ModelRegistry().get("super_add"), config, plan, checkpoint, "cpu")

    assert isinstance(model, LoadedModel)
    assert len(calls) == 1
    assert calls[0]["backbone"] == "vit_base_patch16_dinov3.lvd1689m"
    assert calls[0]["precision"] == "float32"
    assert calls[0]["pre_processor"] == "frozen-preprocessor"