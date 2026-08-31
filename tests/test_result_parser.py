"""Tests for worker message parsing and CSV export."""

from __future__ import annotations

from pathlib import Path

from app.core.result_parser import ResultParser
from app.models.prediction_result import PredictionResult
from app.models.training_run import TrainingRun
from app.workers.inference_worker import _count_images, _find_checkpoint
from app.workers.inference_worker import _threshold_from_model


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
        )
    ]
    parser.export_predictions_csv(path, predictions)
    restored = parser.read_predictions_csv(path)
    assert restored[0].source_path.endswith("image.png")
    assert restored[0].classification_bucket() == "Correct OK"


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
            metrics={"Image AUROC": 1.0, "Image F1": 0.98},
        ),
    )

    restored = parser.read_training_run(path)

    assert restored.model_name == "PaDiM"
    assert restored.metrics == {"Image AUROC": 1.0, "Image F1": 0.98}


def test_inference_worker_finds_checkpoint_and_counts_images(tmp_path: Path) -> None:
    checkpoint = tmp_path / "Padim" / "custom" / "latest" / "weights" / "lightning" / "model.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "one.png").touch()
    (tmp_path / "images" / "ignored.txt").touch()

    assert _find_checkpoint(tmp_path) == checkpoint
    assert _count_images(tmp_path / "images") == 1


def test_inference_worker_uses_normalized_threshold_for_display() -> None:
    class Threshold:
        value = 128.47

    class PostProcessor:
        image_threshold = Threshold()

    class Model:
        post_processor = PostProcessor()

    assert _threshold_from_model(Model()) == 0.5

