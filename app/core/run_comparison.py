"""Comparison eligibility and metric extraction for persisted anomaly-detection runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from app.models.training_run import TrainingRun


COMPARISON_METRICS = (
    "NG Detection Rate",
    "Escape Rate",
    "False Reject Rate",
    "Precision",
    "Recall",
    "F1",
    "AUROC",
    "Decision Threshold",
    "Threshold Method",
    "Training Duration",
    "Mean Inference Latency (ms/image)",
    "P95 Inference Latency (ms/image)",
    "Peak GPU Memory (MB)",
    "Model Size (bytes)",
    "Export Size (bytes)",
)


@dataclass(frozen=True, slots=True)
class RunComparisonReport:
    """A comparison with an explicit evidence-basis eligibility result."""

    runs: tuple[TrainingRun, ...]
    direct_quality_comparison_allowed: bool
    reason: str
    metric_rows: dict[str, tuple[object, ...]]


def compare_training_runs(runs: Iterable[TrainingRun]) -> RunComparisonReport:
    """Compare at least two runs without requiring their preprocessing to match."""
    selected_runs = tuple(runs)
    if len(selected_runs) < 2:
        raise ValueError("Select at least two runs to compare.")
    split_identities = {
        (
            run.dataset_manifest_sha256,
            run.calibration_manifest_sha256,
            run.final_test_manifest_sha256,
            run.inspection_region_hash,
        )
        for run in selected_runs
    }
    complete_identity = all(all(identity) for identity in split_identities)
    direct_allowed = complete_identity and len(split_identities) == 1
    reason = (
        "All runs use the same source dataset, train/calibration/final-test assignments, and inspection ROI."
        if direct_allowed
        else "DIRECT QUALITY COMPARISON NOT ALLOWED: runs do not share the same complete source split manifest and inspection ROI."
    )
    metric_rows: dict[str, tuple[object, ...]] = {}
    for metric in COMPARISON_METRICS:
        values = tuple(_metric_value(run, metric) for run in selected_runs)
        if any(value is not None for value in values):
            metric_rows[metric] = values
    return RunComparisonReport(
        runs=selected_runs,
        direct_quality_comparison_allowed=direct_allowed,
        reason=reason,
        metric_rows=metric_rows,
    )


def _metric_value(run: TrainingRun, metric: str) -> object | None:
    if metric == "Training Duration":
        return run.training_duration_seconds
    if metric == "Mean Inference Latency (ms/image)":
        return run.mean_inference_latency_ms if run.mean_inference_latency_ms is not None else run.metrics.get(metric)
    if metric == "P95 Inference Latency (ms/image)":
        return run.p95_inference_latency_ms if run.p95_inference_latency_ms is not None else run.metrics.get(metric)
    if metric == "Peak GPU Memory (MB)":
        return run.peak_gpu_memory_mb if run.peak_gpu_memory_mb is not None else run.metrics.get(metric)
    if metric == "Decision Threshold":
        return run.threshold_metadata.get("threshold_value") or run.metrics.get(metric)
    if metric == "Threshold Method":
        return run.threshold_metadata.get("threshold_method") or run.metrics.get(metric)
    return run.metrics.get(metric)