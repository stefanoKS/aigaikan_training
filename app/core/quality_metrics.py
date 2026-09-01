"""Production-relevant quality metrics for final test predictions."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable

from app.models.prediction_result import PredictionResult


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Confusion matrix and inspection-focused summary from final-test predictions."""

    metrics: dict[str, float | int | str | None]
    status: str
    warning: str = ""


def calculate_quality_metrics(predictions: Iterable[PredictionResult]) -> QualityReport:
    """Calculate quality metrics without treating AUROC as the primary safety signal."""
    items = list(predictions)
    if not items:
        raise ValueError("Cannot calculate quality metrics without final-test predictions.")
    if any(not isfinite(prediction.anomaly_score) for prediction in items):
        raise ValueError("Prediction scores must all be finite.")

    actual_ng = [item for item in items if item.ground_truth_label.upper() == "NG"]
    actual_ok = [item for item in items if item.ground_truth_label.upper() == "OK"]
    true_ng = sum(item.predicted_label.upper() == "NG" for item in actual_ng)
    escaped_ng = len(actual_ng) - true_ng
    true_ok = sum(item.predicted_label.upper() == "OK" for item in actual_ok)
    false_reject = len(actual_ok) - true_ok
    predicted_ng = true_ng + false_reject
    precision = true_ng / predicted_ng if actual_ng and predicted_ng else None
    recall = true_ng / len(actual_ng) if actual_ng else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall and precision + recall else None
    threshold_values = {round(item.threshold, 12) for item in items if isfinite(item.threshold)}
    threshold = next(iter(threshold_values)) if len(threshold_values) == 1 else None
    metrics: dict[str, float | int | str | None] = {
        "NG Tested": len(actual_ng) if actual_ng else "NOT MEASURED",
        "NG Detected": true_ng if actual_ng else "NOT MEASURED",
        "NG Missed": escaped_ng if actual_ng else "NOT MEASURED",
        "NG Detection Rate": recall if actual_ng else "NOT MEASURED",
        "Escape Rate": escaped_ng / len(actual_ng) if actual_ng else "NOT MEASURED",
        "OK Tested": len(actual_ok),
        "OK Correct": true_ok,
        "False Reject Count": false_reject,
        "False Reject Rate": false_reject / len(actual_ok) if actual_ok else None,
        "Actual OK -> Predicted OK": true_ok,
        "Actual OK -> Predicted NG": false_reject,
        "Actual NG -> Predicted OK (Escaped NG)": escaped_ng if actual_ng else "NOT MEASURED",
        "Actual NG -> Predicted NG": true_ng if actual_ng else "NOT MEASURED",
        "Precision": precision if actual_ng else "NOT MEASURED",
        "Recall": recall if actual_ng else "NOT MEASURED",
        "F1": f1 if actual_ng else "NOT MEASURED",
        "AUROC": _auroc(items) if actual_ng else "NOT MEASURED",
        "Decision Threshold": threshold,
    }
    status = "PASS"
    warning = ""
    if not actual_ng:
        status = "NOT VERIFIED"
        warning = "NO GENUINE NG TEST DATA. DEFECT-DETECTION PERFORMANCE HAS NOT BEEN VERIFIED."
        metrics["Defect Detection Evidence"] = "NOT MEASURED"
    elif escaped_ng:
        status = "FAIL"
    elif len(actual_ng) < 10 or len(actual_ok) < 10:
        status = "WARNING"
    return QualityReport(metrics=metrics, status=status, warning=warning)


def _auroc(predictions: list[PredictionResult]) -> float | None:
    positives = [item.anomaly_score for item in predictions if item.ground_truth_label.upper() == "NG"]
    negatives = [item.anomaly_score for item in predictions if item.ground_truth_label.upper() == "OK"]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))