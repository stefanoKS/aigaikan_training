"""Versioned raw and postprocessed prediction-domain contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Mapping

import numpy as np

PREDICTION_CONTRACT_VERSION = 1
RAW_SCORE_SEMANTIC = "anomalib_model_raw_score_v1"
POSTPROCESSED_SCORE_SEMANTIC = "anomalib_postprocessed_pred_score_v1"
POSTPROCESSED_MAP_SEMANTIC = "anomalib_postprocessed_anomaly_map_v1"
SUPERADD_NATIVE_IMAGE_SCORE_SEMANTIC = "superadd_native_top_quantile_score_v1"
DECISION_COMPARATOR = "greater_than_or_equal"


@dataclass(frozen=True, slots=True)
class ImageThreshold:
    """An image decision threshold bound to one declared score semantic."""

    value: float
    score_semantic: str

    def validate(self) -> None:
        if not isfinite(self.value):
            raise ValueError("Image threshold must be finite.")
        if not self.score_semantic:
            raise ValueError("Image threshold must declare its score semantic.")


@dataclass(frozen=True, slots=True)
class PredictionContract:
    """One prediction with explicit raw and postprocessed values and geometry."""

    raw_image_score: float | None
    raw_anomaly_map: Any
    postprocessed_image_score: float
    postprocessed_anomaly_map: Any
    image_threshold: ImageThreshold
    pixel_threshold: float | None
    predicted_label: str
    decision_image_score: float | None = None
    score_semantic: str = POSTPROCESSED_SCORE_SEMANTIC
    map_semantic: str = POSTPROCESSED_MAP_SEMANTIC
    valid_roi_mask: Any = None
    padding_geometry: Mapping[str, object] = field(default_factory=dict)
    tiling_geometry: Mapping[str, object] = field(default_factory=dict)
    normalization_version: str = POSTPROCESSED_SCORE_SEMANTIC
    contract_version: int = PREDICTION_CONTRACT_VERSION

    def __post_init__(self) -> None:
        """Reject ambiguous, non-finite, or cross-domain image decisions."""
        if self.contract_version != PREDICTION_CONTRACT_VERSION:
            raise ValueError("Unsupported prediction contract version.")
        if self.raw_image_score is not None and not isfinite(self.raw_image_score):
            raise ValueError("Raw image score must be finite when provided.")
        if not self.score_semantic or not self.map_semantic or not self.normalization_version:
            raise ValueError("Prediction contract must declare score, map, and normalization semantics.")
        self.image_threshold.validate()
        if self.image_threshold.score_semantic != self.score_semantic:
            raise ValueError("Image threshold score semantic does not match the prediction decision score.")
        validate_postprocessed_values(
            self.postprocessed_image_score,
            self.postprocessed_anomaly_map,
            self.valid_roi_mask,
        )
        if self.pixel_threshold is not None and not isfinite(self.pixel_threshold):
            raise ValueError("Pixel threshold must be finite when provided.")
        decision_score = self.postprocessed_image_score if self.decision_image_score is None else self.decision_image_score
        if not isfinite(decision_score):
            raise ValueError("Prediction decision score must be finite.")
        expected_label = "NG" if decision_score >= self.image_threshold.value else "OK"
        if self.predicted_label.upper() != expected_label:
            raise ValueError("Predicted label does not match the decision score and image threshold.")


def validate_postprocessed_values(image_score: float, anomaly_map: Any, valid_roi_mask: Any = None) -> None:
    """Validate the documented normalized postprocessor output domain without restricting raw values."""
    if not isfinite(image_score):
        raise ValueError("Postprocessed image score must be finite.")
    if not 0 <= image_score <= 1:
        raise ValueError("Postprocessed image score must be in the normalized range [0, 1].")
    if anomaly_map is None:
        if valid_roi_mask is not None:
            raise ValueError("A valid ROI mask requires a postprocessed anomaly map.")
        return
    values = _map_array(anomaly_map)
    mask = np.isfinite(values) if valid_roi_mask is None else _valid_mask(valid_roi_mask, values.shape)
    if not mask.any():
        raise ValueError("Postprocessed anomaly map valid ROI must contain at least one pixel.")
    valid_values = values[mask]
    if not np.isfinite(valid_values).all():
        raise ValueError("Postprocessed anomaly map contains non-finite valid ROI values.")
    if (valid_values < 0).any() or (valid_values > 1).any():
        raise ValueError("Postprocessed anomaly map valid ROI values must be in the normalized range [0, 1].")


def _map_array(anomaly_map: Any) -> np.ndarray:
    values = anomaly_map.detach().float().cpu().numpy() if hasattr(anomaly_map, "detach") else np.asarray(anomaly_map)
    while values.ndim > 2:
        values = values[0]
    if values.ndim != 2 or values.size == 0:
        raise ValueError("Anomaly map must contain one non-empty two-dimensional image.")
    return np.asarray(values, dtype=np.float32)


def _valid_mask(valid_roi_mask: Any, expected_shape: tuple[int, int]) -> np.ndarray:
    mask = valid_roi_mask.detach().cpu().numpy() if hasattr(valid_roi_mask, "detach") else np.asarray(valid_roi_mask)
    if mask.shape != expected_shape:
        raise ValueError("Valid ROI mask and anomaly map must share dimensions.")
    return mask.astype(bool, copy=False)