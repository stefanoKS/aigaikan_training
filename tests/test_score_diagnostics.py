"""Tests for non-decision score range diagnostics."""

from __future__ import annotations

import pytest

from app.core.score_diagnostics import summarize_prediction_score_ranges
from app.models.prediction_result import PredictionResult


def test_score_diagnostics_keep_raw_and_decision_domains_separate() -> None:
    predictions = [
        PredictionResult(
            source_path="one.png",
            predicted_label="OK",
            ground_truth_label="OK",
            anomaly_score=0.2,
            threshold=0.5,
            score_semantic="postprocessed",
            raw_image_score=12.0,
            raw_score_semantic="raw",
        ),
        PredictionResult(
            source_path="two.png",
            predicted_label="NG",
            ground_truth_label="NG",
            anomaly_score=0.8,
            threshold=0.5,
            score_semantic="postprocessed",
            raw_image_score=18.0,
            raw_score_semantic="raw",
        ),
    ]

    diagnostics = summarize_prediction_score_ranges(predictions)

    assert diagnostics == {
        "decision": {"postprocessed": {"count": 2, "minimum": 0.2, "maximum": 0.8}},
        "raw": {"raw": {"count": 2, "minimum": 12.0, "maximum": 18.0}},
    }


def test_score_diagnostics_reject_scores_without_a_declared_semantic() -> None:
    with pytest.raises(ValueError, match="declared score semantic"):
        summarize_prediction_score_ranges(
            [
                PredictionResult(
                    source_path="one.png",
                    predicted_label="OK",
                    ground_truth_label="OK",
                    anomaly_score=0.2,
                    threshold=0.5,
                )
            ]
        )