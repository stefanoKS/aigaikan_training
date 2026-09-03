"""Tests for factory-relevant final-test quality reporting."""

from __future__ import annotations

from app.core.quality_metrics import FinalTestAcceptancePolicy, calculate_quality_metrics
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


def test_normal_only_final_test_fails_when_false_rejects_exceed_the_policy() -> None:
    policy = FinalTestAcceptancePolicy(
        maximum_false_reject_rate=0.005,
        minimum_ok_test_images=10,
        minimum_ng_test_images=10,
    )

    report = calculate_quality_metrics([_prediction("OK", "NG", 0.8)], policy)

    assert report.status == "FAIL"
    assert "False reject rate 1 exceeds the configured maximum" in report.warning
    assert "NO GENUINE NG TEST DATA" in report.warning


def test_clean_final_test_requires_an_explicit_false_reject_acceptance_policy() -> None:
    predictions = [
        *[_prediction("OK", "OK", 0.1) for _ in range(10)],
        *[_prediction("NG", "NG", 0.9) for _ in range(10)],
    ]

    report = calculate_quality_metrics(predictions)

    assert report.status == "WARNING"
    assert "not configured" in report.warning


def test_false_reject_rate_above_the_acceptance_policy_fails_final_test() -> None:
    predictions = [
        *[_prediction("OK", "NG", 0.8) for _ in range(2)],
        *[_prediction("OK", "OK", 0.1) for _ in range(8)],
        *[_prediction("NG", "NG", 0.9) for _ in range(10)],
    ]
    policy = FinalTestAcceptancePolicy(
        maximum_false_reject_rate=0.1,
        minimum_ok_test_images=10,
        minimum_ng_test_images=10,
    )

    report = calculate_quality_metrics(predictions, policy)

    assert report.status == "FAIL"
    assert report.metrics["False Reject Rate"] == 0.2
    assert "exceeds the configured maximum" in report.warning


def test_clean_final_test_passes_when_it_meets_the_configured_policy() -> None:
    predictions = [
        *[_prediction("OK", "OK", 0.1) for _ in range(10)],
        *[_prediction("NG", "NG", 0.9) for _ in range(10)],
    ]
    policy = FinalTestAcceptancePolicy(0.005, 10, 10)

    assert calculate_quality_metrics(predictions, policy).status == "PASS"
