"""Tests for factory-relevant final-test quality reporting."""

from __future__ import annotations

from app.core.quality_metrics import calculate_quality_metrics
from app.models.prediction_result import PredictionResult


def _prediction(actual: str, predicted: str, score: float) -> PredictionResult:
    return PredictionResult(
        source_path=f"{actual}_{predicted}_{score}.png",
        ground_truth_label=actual,
        predicted_label=predicted,
        anomaly_score=score,
        threshold=0.6,
    )


def test_quality_metrics_prioritize_escaped_ng_and_false_rejects() -> None:
    report = calculate_quality_metrics(
        [
            _prediction("OK", "OK", 0.1),
            _prediction("OK", "NG", 0.8),
            _prediction("NG", "OK", 0.3),
            _prediction("NG", "NG", 0.9),
        ]
    )

    assert report.status == "FAIL"
    assert report.metrics["NG Detected"] == 1
    assert report.metrics["NG Missed"] == 1
    assert report.metrics["Escape Rate"] == 0.5
    assert report.metrics["False Reject Count"] == 1
    assert report.metrics["Actual NG -> Predicted OK (Escaped NG)"] == 1
    assert report.metrics["Decision Threshold"] == 0.6
    assert report.metrics["AUROC"] == 0.75


def test_small_but_clean_final_test_is_a_quality_warning() -> None:
    report = calculate_quality_metrics([_prediction("OK", "OK", 0.1), _prediction("NG", "NG", 0.9)])

    assert report.status == "WARNING"


def test_normal_only_final_test_does_not_report_defect_detection_performance() -> None:
    report = calculate_quality_metrics([_prediction("OK", "OK", 0.1), _prediction("OK", "NG", 0.8)])

    assert report.status == "NOT VERIFIED"
    assert report.metrics["NG Detected"] == "NOT MEASURED"
    assert report.metrics["NG Missed"] == "NOT MEASURED"
    assert report.metrics["Escape Rate"] == "NOT MEASURED"
    assert report.metrics["AUROC"] == "NOT MEASURED"
    assert report.metrics["Precision"] == "NOT MEASURED"
    assert "NO GENUINE NG TEST DATA" in report.warning
