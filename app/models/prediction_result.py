"""Prediction result models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class PredictionResult:
    """Prediction row used in results and inference."""

    source_path: str
    predicted_label: str
    ground_truth_label: str
    anomaly_score: float
    threshold: float
    original_image: str = ""
    anomaly_map: str = ""
    overlay_image: str = ""
    dataset_role: str = ""
    native_image_score: float | None = None
    native_tile_scores: list[float] = field(default_factory=list)
    score_semantic: str = ""
    continuous_anomaly_map: str = ""
    binary_mask: str = ""
    contour_overlay_image: str = ""
    pixel_threshold: float | None = None
    pixel_threshold_comparator: str = ""
    pixel_threshold_semantic: str = ""
    map_display_normalization: dict[str, Any] = field(default_factory=dict)
    region_metadata: dict[str, Any] = field(default_factory=dict)

    def classification_bucket(self) -> str:
        """Return the result bucket."""
        gt_ng = self.ground_truth_label.upper() == "NG"
        pred_ng = self.predicted_label.upper() == "NG"
        if not gt_ng and not pred_ng:
            return "Correct OK"
        if gt_ng and pred_ng:
            return "Correct NG"
        if gt_ng and not pred_ng:
            return "False OK"
        return "False NG"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result."""
        payload = asdict(self)
        payload["classification_bucket"] = self.classification_bucket()
        payload["correct"] = payload["classification_bucket"] in {"Correct OK", "Correct NG"}
        return payload
