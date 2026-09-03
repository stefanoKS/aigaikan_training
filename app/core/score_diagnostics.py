"""Semantic-aware score range diagnostics that never affect decisions."""

from __future__ import annotations

from collections import defaultdict
from math import isfinite
from typing import Iterable

from app.models.prediction_result import PredictionResult


def summarize_prediction_score_ranges(predictions: Iterable[PredictionResult]) -> dict[str, dict[str, dict[str, float | int]]]:
    """Summarize finite decision and raw scores independently for each declared semantic."""
    decision_scores: dict[str, list[float]] = defaultdict(list)
    raw_scores: dict[str, list[float]] = defaultdict(list)
    for prediction in predictions:
        _record(decision_scores, prediction.score_semantic, prediction.anomaly_score, "decision")
        if prediction.raw_image_score is not None:
            _record(raw_scores, prediction.raw_score_semantic, prediction.raw_image_score, "raw")
    return {
        "decision": _summaries(decision_scores),
        "raw": _summaries(raw_scores),
    }


def _record(target: dict[str, list[float]], semantic: str, value: float, score_kind: str) -> None:
    if not semantic:
        raise ValueError(f"{score_kind.capitalize()} score diagnostics require a declared score semantic.")
    numeric_value = float(value)
    if not isfinite(numeric_value):
        raise ValueError(f"{score_kind.capitalize()} score diagnostics require finite scores.")
    target[semantic].append(numeric_value)


def _summaries(scores_by_semantic: dict[str, list[float]]) -> dict[str, dict[str, float | int]]:
    return {
        semantic: {
            "count": len(values),
            "minimum": min(values),
            "maximum": max(values),
        }
        for semantic, values in sorted(scores_by_semantic.items())
    }