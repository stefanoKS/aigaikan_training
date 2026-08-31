"""Tests for strict conversion of Anomalib prediction structures."""

from __future__ import annotations

from math import nan

import pytest

from app.core.prediction_adapter import iter_anomalib_predictions


def test_prediction_adapter_requires_one_finite_score_per_image() -> None:
    output = [{"image_path": ["first.png", "second.png"], "pred_score": [0.1, 0.9]}]

    predictions = list(iter_anomalib_predictions(output))

    assert [prediction.score for prediction in predictions] == [0.1, 0.9]


def test_prediction_adapter_rejects_nonfinite_or_mismatched_output() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        list(iter_anomalib_predictions([{"image_path": ["first.png"], "pred_score": [nan]}]))
    with pytest.raises(ValueError, match="one image path and score"):
        list(iter_anomalib_predictions([{"image_path": ["first.png"], "pred_score": []}]))