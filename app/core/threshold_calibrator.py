"""Reproducible decision-threshold calibration for anomaly scores."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from math import ceil, isfinite, nextafter
from typing import Iterable


class ThresholdMethod(StrEnum):
    """Supported calibration policies with explicit evidence requirements."""

    AUTO = "auto"
    LABELED_F1 = "labeled_f1"
    LABELED_RECALL_PRIORITY = "labeled_recall_priority"
    NORMAL_ONLY_CONFORMAL = "normal_only_conformal"
    NORMAL_ONLY_MAX = "normal_only_max"
    SYNTHETIC_ANOMALY = "synthetic_anomaly"


@dataclass(frozen=True, slots=True)
class CalibrationSample:
    """A finite validation anomaly score with its known calibration label."""

    score: float
    label: str

    @property
    def is_normal(self) -> bool:
        """Return whether the sample is a known normal image."""
        return self.label.upper() == "OK"

    @property
    def is_abnormal(self) -> bool:
        """Return whether the sample is a known abnormal image."""
        return self.label.upper() == "NG"


@dataclass(frozen=True, slots=True)
class ThresholdCalibrationConfig:
    """Policy settings persisted with a training or evaluation revision."""

    method: ThresholdMethod = ThresholdMethod.LABELED_F1
    target_normal_false_reject_rate: float = 0.005
    minimum_required_ng_recall: float | None = None

    def validate(self) -> None:
        """Validate the selected policy without making distributional assumptions."""
        if not 0 < self.target_normal_false_reject_rate < 1:
            raise ValueError("Target normal false reject rate must be between zero and one.")
        if self.minimum_required_ng_recall is not None and not 0 <= self.minimum_required_ng_recall <= 1:
            raise ValueError("Minimum required NG recall must be between zero and one.")


@dataclass(frozen=True, slots=True)
class ThresholdCalibrationResult:
    """Threshold evidence persisted independently from the model checkpoint."""

    threshold_method: str
    threshold_value: float
    threshold_raw: float
    threshold_deployed: float
    calibration_sample_count: int
    normal_calibration_sample_count: int
    abnormal_calibration_sample_count: int
    target_false_reject_rate: float | None
    observed_calibration_false_reject_rate: float | None
    ng_recall: float | None
    precision: float | None
    f1: float | None
    normal_score_quantiles: dict[str, float] = field(default_factory=dict)
    abnormal_score_quantiles: dict[str, float] = field(default_factory=dict)
    normal_score_iqr: float | None = None
    abnormal_score_iqr: float | None = None
    warning: str = ""
    experimental: bool = False

    def to_dict(self) -> dict[str, object]:
        """Serialize all calibration evidence for a run or deployment manifest."""
        return asdict(self)


class ThresholdCalibrator:
    """Select a reproducible operating point from calibration-only scores."""

    def calibrate(
        self,
        samples: Iterable[CalibrationSample],
        config: ThresholdCalibrationConfig,
    ) -> ThresholdCalibrationResult:
        """Calibrate a threshold without inspecting final-test predictions."""
        config.validate()
        values = tuple(samples)
        self._validate_samples(values)
        if config.method is ThresholdMethod.AUTO:
            if any(sample.is_abnormal for sample in values):
                return self._labeled_f1(values)
            return self._normal_only_conformal(values, config.target_normal_false_reject_rate)
        if config.method is ThresholdMethod.LABELED_F1:
            return self._labeled_f1(values)
        if config.method is ThresholdMethod.LABELED_RECALL_PRIORITY:
            return self._labeled_recall_priority(values, config.minimum_required_ng_recall)
        if config.method is ThresholdMethod.NORMAL_ONLY_CONFORMAL:
            return self._normal_only_conformal(values, config.target_normal_false_reject_rate)
        if config.method is ThresholdMethod.NORMAL_ONLY_MAX:
            return self._normal_only_max(values)
        if config.method is ThresholdMethod.SYNTHETIC_ANOMALY:
            raise ValueError(
                "Synthetic anomaly calibration is experimental and requires a dedicated synthetic-data generator, "
                "which is not available in this build."
            )
        raise ValueError(f"Unsupported threshold calibration method: {config.method}")

    @staticmethod
    def _validate_samples(samples: tuple[CalibrationSample, ...]) -> None:
        if not samples:
            raise ValueError("Threshold calibration requires at least one validation score.")
        if any(not isfinite(sample.score) for sample in samples):
            raise ValueError("Threshold calibration scores must all be finite.")
        invalid_labels = sorted({sample.label for sample in samples if not (sample.is_normal or sample.is_abnormal)})
        if invalid_labels:
            raise ValueError(f"Calibration labels must be OK or NG, received: {', '.join(invalid_labels)}")

    def _labeled_f1(self, samples: tuple[CalibrationSample, ...]) -> ThresholdCalibrationResult:
        normal_scores, abnormal_scores = self._labeled_scores(samples)
        candidates = [self._labeled_metrics(normal_scores, abnormal_scores, threshold) for threshold in sorted(set(
            [*normal_scores, *abnormal_scores]
        ))]
        selected = max(candidates, key=lambda candidate: (candidate["f1"], candidate["precision"], candidate["recall"], -candidate["threshold"]))
        return self._labeled_result(ThresholdMethod.LABELED_F1, normal_scores, abnormal_scores, selected)

    def _labeled_recall_priority(
        self,
        samples: tuple[CalibrationSample, ...],
        minimum_required_ng_recall: float | None,
    ) -> ThresholdCalibrationResult:
        normal_scores, abnormal_scores = self._labeled_scores(samples)
        candidates = [self._labeled_metrics(normal_scores, abnormal_scores, threshold) for threshold in sorted(set(
            [*normal_scores, *abnormal_scores]
        ))]
        selected = max(
            candidates,
            key=lambda candidate: (
                candidate["recall"],
                candidate["precision"],
                -candidate["false_reject_count"],
                candidate["threshold"],
            ),
        )
        if minimum_required_ng_recall is not None and selected["recall"] < minimum_required_ng_recall:
            raise ValueError(
                f"The selected threshold reaches NG recall {selected['recall']:.6g}, below the required "
                f"{minimum_required_ng_recall:.6g}."
            )
        return self._labeled_result(ThresholdMethod.LABELED_RECALL_PRIORITY, normal_scores, abnormal_scores, selected)

    @staticmethod
    def _labeled_scores(samples: tuple[CalibrationSample, ...]) -> tuple[list[float], list[float]]:
        normal_scores = [sample.score for sample in samples if sample.is_normal]
        abnormal_scores = [sample.score for sample in samples if sample.is_abnormal]
        if not normal_scores or not abnormal_scores:
            raise ValueError("Labeled calibration requires at least one real OK score and one real NG score.")
        return normal_scores, abnormal_scores

    @staticmethod
    def _labeled_metrics(normal_scores: list[float], abnormal_scores: list[float], threshold: float) -> dict[str, float | int]:
        true_positive = sum(score >= threshold for score in abnormal_scores)
        false_reject = sum(score >= threshold for score in normal_scores)
        predicted_ng = true_positive + false_reject
        recall = true_positive / len(abnormal_scores)
        precision = true_positive / predicted_ng if predicted_ng else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "threshold": threshold,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "false_reject_count": false_reject,
        }

    @staticmethod
    def _labeled_result(
        method: ThresholdMethod,
        normal_scores: list[float],
        abnormal_scores: list[float],
        selected: dict[str, float | int],
    ) -> ThresholdCalibrationResult:
        threshold = float(selected["threshold"])
        return ThresholdCalibrationResult(
            threshold_method=method.value,
            threshold_value=threshold,
            threshold_raw=threshold,
            threshold_deployed=threshold,
            calibration_sample_count=len(normal_scores) + len(abnormal_scores),
            normal_calibration_sample_count=len(normal_scores),
            abnormal_calibration_sample_count=len(abnormal_scores),
            target_false_reject_rate=None,
            observed_calibration_false_reject_rate=int(selected["false_reject_count"]) / len(normal_scores),
            ng_recall=float(selected["recall"]),
            precision=float(selected["precision"]),
            f1=float(selected["f1"]),
            normal_score_quantiles=_score_quantiles(normal_scores),
            abnormal_score_quantiles=_score_quantiles(abnormal_scores),
            normal_score_iqr=_score_iqr(normal_scores),
            abnormal_score_iqr=_score_iqr(abnormal_scores),
        )

    @staticmethod
    def _normal_only_conformal(
        samples: tuple[CalibrationSample, ...],
        alpha: float,
    ) -> ThresholdCalibrationResult:
        normal_scores = ThresholdCalibrator._normal_scores(samples)
        ordered_scores = sorted(normal_scores)
        rank = ceil((len(ordered_scores) + 1) * (1 - alpha))
        index = min(max(rank, 1), len(ordered_scores)) - 1
        raw_threshold = ordered_scores[index]
        deployed_threshold = nextafter(raw_threshold, float("inf"))
        resolution_warning = _calibration_resolution_warning(len(normal_scores), alpha)
        return ThresholdCalibrationResult(
            threshold_method=ThresholdMethod.NORMAL_ONLY_CONFORMAL.value,
            threshold_value=deployed_threshold,
            threshold_raw=raw_threshold,
            threshold_deployed=deployed_threshold,
            calibration_sample_count=len(normal_scores),
            normal_calibration_sample_count=len(normal_scores),
            abnormal_calibration_sample_count=0,
            target_false_reject_rate=alpha,
            observed_calibration_false_reject_rate=(
                sum(score >= deployed_threshold for score in normal_scores) / len(normal_scores)
            ),
            ng_recall=None,
            precision=None,
            f1=None,
            normal_score_quantiles=_score_quantiles(normal_scores),
            normal_score_iqr=_score_iqr(normal_scores),
            warning=(
                "Threshold was estimated from normal calibration images only; defect separation has not been calibrated."
                f" {resolution_warning}"
            ).strip(),
        )

    @staticmethod
    def _normal_only_max(samples: tuple[CalibrationSample, ...]) -> ThresholdCalibrationResult:
        normal_scores = ThresholdCalibrator._normal_scores(samples)
        threshold = max(normal_scores)
        return ThresholdCalibrationResult(
            threshold_method=ThresholdMethod.NORMAL_ONLY_MAX.value,
            threshold_value=threshold,
            threshold_raw=threshold,
            threshold_deployed=threshold,
            calibration_sample_count=len(normal_scores),
            normal_calibration_sample_count=len(normal_scores),
            abnormal_calibration_sample_count=0,
            target_false_reject_rate=None,
            observed_calibration_false_reject_rate=sum(score >= threshold for score in normal_scores) / len(normal_scores),
            ng_recall=None,
            precision=None,
            f1=None,
            normal_score_quantiles=_score_quantiles(normal_scores),
            normal_score_iqr=_score_iqr(normal_scores),
            warning="This threshold was estimated without known abnormal samples and may provide poor defect separation.",
        )

    @staticmethod
    def _normal_scores(samples: tuple[CalibrationSample, ...]) -> list[float]:
        normal_scores = [sample.score for sample in samples if sample.is_normal]
        if not normal_scores:
            raise ValueError("Normal-only calibration requires at least one held-out OK validation score.")
        return normal_scores


def _score_quantiles(scores: list[float]) -> dict[str, float]:
    """Return deterministic observed-score quantiles for provenance and stability review."""
    ordered = sorted(scores)
    return {
        "min": ordered[0],
        "p05": _quantile(ordered, 0.05),
        "p50": _quantile(ordered, 0.5),
        "p95": _quantile(ordered, 0.95),
        "max": ordered[-1],
    }


def _score_iqr(scores: list[float]) -> float:
    """Return the observed interquartile range without distributional assumptions."""
    ordered = sorted(scores)
    return _quantile(ordered, 0.75) - _quantile(ordered, 0.25)


def _quantile(ordered_scores: list[float], fraction: float) -> float:
    """Use a deterministic nearest-rank observed score rather than interpolated values."""
    index = min(max(ceil(len(ordered_scores) * fraction), 1), len(ordered_scores)) - 1
    return ordered_scores[index]


def _calibration_resolution_warning(sample_count: int, alpha: float) -> str:
    """Describe when finite calibration data cannot resolve the requested false-reject target."""
    minimum_sample_count = ceil(1 / alpha)
    if sample_count >= minimum_sample_count:
        return ""
    return (
        f"The {alpha:.6g} false-reject target cannot be resolved with {sample_count} normal calibration images; "
        f"at least {minimum_sample_count} are needed to observe one false reject at that rate."
    )