"""Versioned threshold contracts for deployed image and pixel decisions."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

PIXEL_OPERATING_POINT_VERSION = 1
PIXEL_THRESHOLD_COMPARATOR = "greater_than_or_equal"
PIXEL_THRESHOLD_SEMANTIC = "continuous_anomaly_map_gte_v1"
IMAGE_OPERATING_POINT_VERSION = 1
IMAGE_THRESHOLD_COMPARATOR = "greater_than_or_equal"
IMAGE_THRESHOLD_SEMANTIC = "anomalib_postprocessed_pred_score_v1"


@dataclass(frozen=True, slots=True)
class ImageThresholdOperatingPoint:
    """A versioned image decision threshold bound to one postprocessed score domain."""

    threshold: float
    score_semantic: str = IMAGE_THRESHOLD_SEMANTIC

    def validate(self) -> None:
        if not isfinite(self.threshold):
            raise ValueError("Image threshold must be finite.")
        if not self.score_semantic:
            raise ValueError("Image threshold must declare its score semantic.")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "version": IMAGE_OPERATING_POINT_VERSION,
            "threshold": self.threshold,
            "comparator": IMAGE_THRESHOLD_COMPARATOR,
            "score_semantic": self.score_semantic,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ImageThresholdOperatingPoint":
        if payload.get("version") != IMAGE_OPERATING_POINT_VERSION:
            raise ValueError("Unsupported image operating-point version.")
        if payload.get("comparator") != IMAGE_THRESHOLD_COMPARATOR:
            raise ValueError("Unsupported image threshold comparator.")
        score_semantic = payload.get("score_semantic")
        if not isinstance(score_semantic, str) or not score_semantic:
            raise ValueError("Image threshold score semantic must be a non-empty string.")
        try:
            threshold = float(payload.get("threshold"))
        except (TypeError, ValueError) as exc:
            raise ValueError("Image threshold must be finite.") from exc
        result = cls(threshold=threshold, score_semantic=score_semantic)
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class PixelThresholdOperatingPoint:
    """An opt-in pixel mask threshold independent from the image decision threshold."""

    enabled: bool = False
    threshold: float = 0.5

    def validate(self) -> None:
        """Reject invalid values even when a currently disabled value is persisted."""
        if not isfinite(self.threshold):
            raise ValueError("Pixel threshold must be finite.")

    @property
    def active_threshold(self) -> float | None:
        """Return the threshold used for masks, or ``None`` when mask output is disabled."""
        return self.threshold if self.enabled else None

    def to_dict(self) -> dict[str, object]:
        """Serialize fixed semantic metadata alongside the optional numeric threshold."""
        self.validate()
        return {
            "version": PIXEL_OPERATING_POINT_VERSION,
            "enabled": self.enabled,
            "threshold": self.active_threshold,
            "comparator": PIXEL_THRESHOLD_COMPARATOR,
            "semantic": PIXEL_THRESHOLD_SEMANTIC,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PixelThresholdOperatingPoint":
        """Read a persisted operating point without accepting altered comparison semantics."""
        if payload.get("version") != PIXEL_OPERATING_POINT_VERSION:
            raise ValueError("Unsupported pixel operating-point version.")
        if payload.get("comparator") != PIXEL_THRESHOLD_COMPARATOR:
            raise ValueError("Unsupported pixel threshold comparator.")
        if payload.get("semantic") != PIXEL_THRESHOLD_SEMANTIC:
            raise ValueError("Unsupported pixel threshold semantic.")
        enabled = payload.get("enabled")
        if not isinstance(enabled, bool):
            raise ValueError("Pixel operating point enabled must be a boolean.")
        value = payload.get("threshold")
        if enabled and value is None:
            raise ValueError("Enabled pixel operating point must contain a finite threshold.")
        if not enabled and value is not None:
            raise ValueError("Disabled pixel operating point must not contain a threshold.")
        try:
            threshold = float(value) if value is not None else 0.5
        except (TypeError, ValueError) as exc:
            raise ValueError("Pixel operating point threshold must be finite.") from exc
        operating_point = cls(enabled=enabled, threshold=threshold)
        operating_point.validate()
        return operating_point