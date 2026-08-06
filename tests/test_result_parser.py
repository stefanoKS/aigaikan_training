"""Tests for worker message parsing and CSV export."""

from __future__ import annotations

from pathlib import Path

from app.core.result_parser import ResultParser
from app.models.prediction_result import PredictionResult


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

