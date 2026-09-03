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


@dataclass(frozen=True, slots=True)
class FinalTestAcceptancePolicy:
    """Production acceptance limits applied only to independent final-test evidence."""

    maximum_false_reject_rate: float
    minimum_ok_test_images: int
    minimum_ng_test_images: int

    def validate(self) -> None:
        """Reject ambiguous or unachievable acceptance-policy settings."""
        if not 0 <= self.maximum_false_reject_rate <= 1:
            raise ValueError("Maximum false reject rate must be between zero and one.")
        if self.minimum_ok_test_images <= 0 or self.minimum_ng_test_images <= 0:
            raise ValueError("Minimum final-test evidence counts must be positive.")


def calculate_quality_metrics(
    predictions: Iterable[PredictionResult],
    acceptance_policy: FinalTestAcceptancePolicy | None = None,
) -> QualityReport:
    """Calculate quality metrics without treating AUROC as the primary safety signal."""
    items = list(predictions)
    if not items:
        raise ValueError("Cannot calculate quality metrics without final-test predictions.")
    if any(not isfinite(prediction.anomaly_score) for prediction in items):
        raise ValueError("Prediction scores must all be finite.")
    if acceptance_policy is not None:
        acceptance_policy.validate()

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
        "Final-Test Acceptance Policy": (
            "NOT CONFIGURED"
            if acceptance_policy is None
            else (
                f"False reject rate <= {acceptance_policy.maximum_false_reject_rate:.6g}; "
                f"OK >= {acceptance_policy.minimum_ok_test_images}; NG >= {acceptance_policy.minimum_ng_test_images}"
            )
        ),
    }
    status = "WARNING"
    warning = ""
    no_ng_warning = "NO GENUINE NG TEST DATA. DEFECT-DETECTION PERFORMANCE HAS NOT BEEN VERIFIED."
    if not actual_ng:
        metrics["Defect Detection Evidence"] = "NOT MEASURED"
    if escaped_ng:
        status = "FAIL"
        warning = "Escaped NG final-test images exceed the zero-escape requirement."
    elif (
        acceptance_policy is not None
        and actual_ok
        and false_reject / len(actual_ok) > acceptance_policy.maximum_false_reject_rate
    ):
        status = "FAIL"
        warning = (
            f"False reject rate {false_reject / len(actual_ok):.6g} exceeds the configured maximum "
            f"{acceptance_policy.maximum_false_reject_rate:.6g}."
        )
        if not actual_ng:
            warning = f"{warning} {no_ng_warning}"
    elif not actual_ng:
        status = "NOT VERIFIED"
        warning = no_ng_warning
    elif acceptance_policy is None:
        warning = "Final-test acceptance policy is not configured; false-reject acceptability is not established."
    elif len(actual_ok) < acceptance_policy.minimum_ok_test_images or len(actual_ng) < acceptance_policy.minimum_ng_test_images:
        evidence_gaps = []
        if len(actual_ok) < acceptance_policy.minimum_ok_test_images:
            evidence_gaps.append(f"OK {len(actual_ok)}/{acceptance_policy.minimum_ok_test_images}")
        if len(actual_ng) < acceptance_policy.minimum_ng_test_images:
            evidence_gaps.append(f"NG {len(actual_ng)}/{acceptance_policy.minimum_ng_test_images}")
        warning = f"Final-test evidence is insufficient for acceptance: {', '.join(evidence_gaps)}."
    else:
        status = "PASS"
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