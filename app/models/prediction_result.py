"""Prediction result models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
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
    raw_image_score: float | None = None
    raw_score_semantic: str = ""
    raw_anomaly_map: str = ""
    postprocessed_image_score: float | None = None
    postprocessed_score_semantic: str = ""
    postprocessed_anomaly_map: str = ""
    prediction_contract_version: int = 0
    continuous_anomaly_map: str = ""
    binary_mask: str = ""
    contour_overlay_image: str = ""
    pixel_threshold: float | None = None
    pixel_threshold_comparator: str = ""
    pixel_threshold_semantic: str = ""
    map_display_normalization: dict[str, Any] = field(default_factory=dict)
    region_metadata: dict[str, Any] = field(default_factory=dict)
    timing_metadata: dict[str, Any] = field(default_factory=dict)
    decision_revision_id: str = ""

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

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PredictionResult":
        """Losslessly reconstruct a worker, CSV, or JSON prediction payload."""
        return cls(
            source_path=str(payload.get("source_path", payload.get("image_path", ""))),
            predicted_label=str(payload.get("predicted_label", payload.get("prediction", ""))),
            ground_truth_label=str(payload.get("ground_truth_label", payload.get("ground_truth", ""))),
            anomaly_score=float(payload.get("anomaly_score", payload.get("score", 0.0))),
            threshold=float(payload.get("threshold", 0.0)),
            original_image=str(payload.get("original_image", "")),
            anomaly_map=str(payload.get("anomaly_map", "")),
            overlay_image=str(payload.get("overlay_image", "")),
            dataset_role=str(payload.get("dataset_role", "")),
            native_image_score=_optional_float(payload.get("native_image_score")),
            native_tile_scores=_float_list(payload.get("native_tile_scores")),
            score_semantic=str(payload.get("score_semantic", "")),
            raw_image_score=_optional_float(payload.get("raw_image_score")),
            raw_score_semantic=str(payload.get("raw_score_semantic", "")),
            raw_anomaly_map=str(payload.get("raw_anomaly_map", "")),
            postprocessed_image_score=_optional_float(payload.get("postprocessed_image_score")),
            postprocessed_score_semantic=str(payload.get("postprocessed_score_semantic", "")),
            postprocessed_anomaly_map=str(payload.get("postprocessed_anomaly_map", "")),
            prediction_contract_version=int(payload.get("prediction_contract_version", 0)),
            continuous_anomaly_map=str(payload.get("continuous_anomaly_map", "")),
            binary_mask=str(payload.get("binary_mask", "")),
            contour_overlay_image=str(payload.get("contour_overlay_image", "")),
            pixel_threshold=_optional_float(payload.get("pixel_threshold")),
            pixel_threshold_comparator=str(payload.get("pixel_threshold_comparator", "")),
            pixel_threshold_semantic=str(payload.get("pixel_threshold_semantic", "")),
            map_display_normalization=_mapping(payload.get("map_display_normalization")),
            region_metadata=_mapping(payload.get("region_metadata")),
            timing_metadata=_mapping(payload.get("timing_metadata", payload.get("timing", {}))),
            decision_revision_id=str(payload.get("decision_revision_id", "")),
        )


def _optional_float(value: object) -> float | None:
    return None if value in (None, "") else float(value)


def _float_list(value: object) -> list[float]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, (list, tuple)):
        raise ValueError("Native tile scores must be an array.")
    return [float(item) for item in value]


def _mapping(value: object) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if isinstance(value, str):
        value = json.loads(value)
    return dict(value) if isinstance(value, dict) else {}
