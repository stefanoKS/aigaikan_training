"""Tests for reproducible threshold calibration policies."""

from __future__ import annotations

from math import nextafter

import pytest

from app.core.threshold_calibrator import (
    CalibrationSample,
    ThresholdCalibrationConfig,
    ThresholdCalibrator,
    ThresholdMethod,
)


def _samples(*values: tuple[float, str]) -> list[CalibrationSample]:
    return [CalibrationSample(score=score, label=label) for score, label in values]


def test_labeled_f1_selects_the_highest_f1_validation_threshold() -> None:
    result = ThresholdCalibrator().calibrate(
        _samples((0.1, "OK"), (0.2, "OK"), (0.4, "NG"), (0.8, "NG")),
        ThresholdCalibrationConfig(ThresholdMethod.LABELED_F1),
    )

    assert result.threshold_method == "labeled_f1"
    assert result.threshold_value == 0.4
    assert result.f1 == 1.0
    assert result.ng_recall == 1.0


def test_recall_priority_preserves_ng_recall_before_false_rejects() -> None:
    result = ThresholdCalibrator().calibrate(
        _samples((0.1, "OK"), (0.9, "OK"), (0.2, "NG"), (0.3, "NG")),
        ThresholdCalibrationConfig(ThresholdMethod.LABELED_RECALL_PRIORITY, minimum_required_ng_recall=1.0),
    )

    assert result.threshold_method == "labeled_recall_priority"
    assert result.threshold_value == 0.2
    assert result.ng_recall == 1.0
    assert result.observed_calibration_false_reject_rate == 0.5


def test_recall_priority_minimizes_false_rejects_after_meeting_the_ng_recall_target() -> None:
    result = ThresholdCalibrator().calibrate(
        _samples((0.15, "OK"), (0.1, "NG"), (0.9, "NG")),
        ThresholdCalibrationConfig(ThresholdMethod.LABELED_RECALL_PRIORITY, minimum_required_ng_recall=0.5),
    )

    assert result.threshold_value == 0.9
    assert result.ng_recall == 0.5
    assert result.observed_calibration_false_reject_rate == 0.0


def test_auto_honors_the_minimum_required_ng_recall_before_optimizing_f1() -> None:
    result = ThresholdCalibrator().calibrate(
        _samples((0.1, "OK"), (0.2, "OK"), (0.15, "NG"), (0.9, "NG")),
        ThresholdCalibrationConfig(ThresholdMethod.AUTO, minimum_required_ng_recall=0.5),
    )

    assert result.threshold_method == "labeled_recall_priority"
    assert result.threshold_value == 0.9
    assert result.ng_recall == 0.5


def test_normal_only_conformal_uses_deterministic_finite_sample_order_statistic() -> None:
    samples = _samples((0.8, "OK"), (0.1, "OK"), (0.5, "OK"), (0.3, "OK"))
    config = ThresholdCalibrationConfig(ThresholdMethod.NORMAL_ONLY_CONFORMAL, target_normal_false_reject_rate=0.5)

    first = ThresholdCalibrator().calibrate(samples, config)
    second = ThresholdCalibrator().calibrate(list(reversed(samples)), config)

    assert first.threshold_raw == 0.5
    assert first.threshold_deployed == nextafter(0.5, float("inf"))
    assert first.threshold_value == first.threshold_deployed
    assert first == second
    assert first.calibration_sample_count == 4
    assert first.observed_calibration_false_reject_rate == 0.25


def test_normal_only_conformal_lifts_a_tied_raw_quantile_for_greater_equal_decisions() -> None:
    result = ThresholdCalibrator().calibrate(
        _samples((0.1, "OK"), (0.5, "OK"), (0.5, "OK")),
        ThresholdCalibrationConfig(ThresholdMethod.NORMAL_ONLY_CONFORMAL, target_normal_false_reject_rate=0.5),
    )

    assert result.threshold_raw == 0.5
    assert result.threshold_value > result.threshold_raw
    assert all(score < result.threshold_value for score in (0.1, 0.5, 0.5))
    assert result.observed_calibration_false_reject_rate == 0.0


def test_normal_only_conformal_warns_when_the_sample_count_cannot_resolve_alpha() -> None:
    result = ThresholdCalibrator().calibrate(
        _samples(*((0.1, "OK"),) * 199),
        ThresholdCalibrationConfig(ThresholdMethod.NORMAL_ONLY_CONFORMAL, target_normal_false_reject_rate=0.005),
    )

    assert "at least 200" in result.warning


def test_normal_only_calibration_has_no_ng_dependent_metrics() -> None:
    result = ThresholdCalibrator().calibrate(
        _samples((0.1, "OK"), (0.2, "OK"), (0.3, "NG")),
        ThresholdCalibrationConfig(ThresholdMethod.NORMAL_ONLY_CONFORMAL, target_normal_false_reject_rate=0.1),
    )

    assert result.abnormal_calibration_sample_count == 0
    assert result.ng_recall is None
    assert result.precision is None
    assert result.f1 is None


def test_normal_only_max_is_explicitly_a_conservative_fallback() -> None:
    result = ThresholdCalibrator().calibrate(
        _samples((0.1, "OK"), (0.4, "OK")),
        ThresholdCalibrationConfig(ThresholdMethod.NORMAL_ONLY_MAX),
    )

    assert result.threshold_raw == 0.4
    assert result.threshold_value == nextafter(0.4, float("inf"))
    assert result.observed_calibration_false_reject_rate == 0.0
    assert result.threshold_method == "normal_only_max"
    assert "without known abnormal samples" in result.warning


def test_calibration_rejects_scores_from_different_semantic_domains() -> None:
    with pytest.raises(ValueError, match="one declared score semantic"):
        ThresholdCalibrator().calibrate(
            [
                CalibrationSample(0.1, "OK", "postprocessed-v1"),
                CalibrationSample(0.8, "NG", "raw-v1"),
            ],
            ThresholdCalibrationConfig(ThresholdMethod.LABELED_F1),
        )