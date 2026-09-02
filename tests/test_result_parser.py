"""Tests for worker message parsing and CSV export."""

from __future__ import annotations

from pathlib import Path

from app.core.result_parser import ResultParser
from app.core.run_artifacts import CanonicalCheckpoint, read_canonical_checkpoint, read_persisted_threshold, write_run_manifest
from app.models.prediction_result import PredictionResult
from app.models.training_run import TrainingRun
from app.workers.inference_worker import _count_images, configure_worker_stdio


def test_parse_worker_json_messages() -> None:
    parser = ResultParser()
    state = parser.collect(
        [
            '{"type":"stage","name":"Validating dataset"}',
            '{"type":"metric","name":"image_AUROC","value":0.98}',
            '{"type":"completed","result_dir":"runs/one"}',
        ]
    )
    assert state.stages == ["Validating dataset"]
    assert state.metrics["Image AUROC"] == 0.98
    assert state.completed_result_dir == "runs/one"


def test_export_and_read_predictions_csv(tmp_path: Path) -> None:
    parser = ResultParser()
    path = tmp_path / "predictions.csv"
    predictions = [
        PredictionResult(
            source_path="C:/space path/日本語/image.png",
            predicted_label="OK",
            ground_truth_label="OK",
            anomaly_score=0.12,
            threshold=0.5,
            pixel_threshold=3.5,
            pixel_threshold_comparator="greater_than_or_equal",
            pixel_threshold_semantic="continuous_anomaly_map_gte_v1",
        )
    ]
    parser.export_predictions_csv(path, predictions)
    restored = parser.read_predictions_csv(path)
    assert restored[0].source_path.endswith("image.png")
    assert restored[0].classification_bucket() == "Correct OK"
    assert restored[0].pixel_threshold == 3.5
    assert restored[0].pixel_threshold_comparator == "greater_than_or_equal"
    assert restored[0].pixel_threshold_semantic == "continuous_anomaly_map_gte_v1"


def test_write_and_read_training_run(tmp_path: Path) -> None:
    parser = ResultParser()
    path = tmp_path / "results.json"
    parser.write_training_run(
        path,
        TrainingRun(
            run_name="2026-08-12_12-47-12_padim",
            run_dir=str(tmp_path),
            model_name="PaDiM",
            device="gpu",
            run_date="2026-08-12T03:47:12+00:00",
            training_duration_seconds=12.5,
            evaluation_duration_seconds=3.25,
            anomalib_export_parity_status="Validated with Anomalib deployment inferencer: TORCH",
            aigaikan_compatibility_status="Pending AIGAIKAN runtime validation",
            metrics={"Image AUROC": 1.0, "Image F1": 0.98},
        ),
    )

    restored = parser.read_training_run(path)

    assert restored.model_name == "PaDiM"
    assert restored.metrics == {"Image AUROC": 1.0, "Image F1": 0.98}
    assert restored.anomalib_export_parity_status == "Validated with Anomalib deployment inferencer: TORCH"
    assert restored.aigaikan_compatibility_status == "Pending AIGAIKAN runtime validation"


def test_inference_worker_uses_manifest_checkpoint_and_counts_images(tmp_path: Path) -> None:
    checkpoint = tmp_path / "Padim" / "custom" / "latest" / "weights" / "lightning" / "model.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "one.png").touch()
    (tmp_path / "images" / "ignored.txt").touch()
    (tmp_path / "images" / "nested").mkdir()
    (tmp_path / "images" / "nested" / "two.jpg").touch()
    write_run_manifest(
        tmp_path / "run_manifest.json",
        canonical_checkpoint=CanonicalCheckpoint(checkpoint.resolve(), ""),
        dataset_manifest_sha256="a" * 64,
        split_counts={"final_test": {"ok": 1, "ng": 1}},
        threshold=128.47,
    )

    checkpoint.write_text("checkpoint", encoding="utf-8")
    canonical = CanonicalCheckpoint(checkpoint.resolve(), __import__("hashlib").sha256(b"checkpoint").hexdigest())
    write_run_manifest(
        tmp_path / "run_manifest.json",
        canonical_checkpoint=canonical,
        dataset_manifest_sha256="a" * 64,
        split_counts={"final_test": {"ok": 1, "ng": 1}},
        threshold=128.47,
    )
    assert read_canonical_checkpoint(tmp_path) == canonical
    assert read_persisted_threshold(tmp_path) == 128.47
    assert _count_images(tmp_path / "images") == 2


def test_inference_worker_stdio_configuration_is_safe_under_pytest_capture() -> None:
    configure_worker_stdio()

