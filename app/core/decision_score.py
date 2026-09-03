"""One semantic-safe image decision-score resolver for every runtime path."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING

from app.core.prediction_contract import POSTPROCESSED_SCORE_SEMANTIC, SUPERADD_NATIVE_IMAGE_SCORE_SEMANTIC
from app.models.preprocessing_config import LEGACY_PREPROCESSING_CONTRACT_VERSION, ResolvedPreprocessingPlan

if TYPE_CHECKING:
    from app.core.preprocessing_pipeline import PreprocessingPipeline, ReconstructedAnomalyMap


@dataclass(frozen=True, slots=True)
class DecisionScore:
    """A finite image score bound to its exact threshold-compatible semantic domain."""

    value: float
    semantic: str
    source: str

    def __post_init__(self) -> None:
        if not isfinite(self.value):
            raise ValueError("Decision score must be finite.")
        if not self.semantic or not self.source:
            raise ValueError("Decision score must declare semantic and source.")


def resolve_decision_score(
    plan: ResolvedPreprocessingPlan | None,
    *,
    postprocessed_image_score: float | None,
    raw_image_score: float | None,
    reconstructed_map: ReconstructedAnomalyMap | None = None,
    preprocessing_pipeline: PreprocessingPipeline | None = None,
) -> DecisionScore:
    """Resolve the one score that may be compared with a saved image threshold.

    The resolver preserves frozen legacy map aggregation, v3 tiled map
    aggregation, native postprocessed non-tiled scores, and SuperADD's raw
    native top-quantile score. Callers must provide the corresponding source
    data; missing data fails closed rather than crossing score domains.
    """
    if plan is None:
        return _native_postprocessed(postprocessed_image_score)
    if plan.preprocessing_contract_version == LEGACY_PREPROCESSING_CONTRACT_VERSION:
        return _reconstructed_map_score(reconstructed_map, preprocessing_pipeline, "legacy_valid_anomaly_map")
    if plan.tiled:
        return _reconstructed_map_score(reconstructed_map, preprocessing_pipeline, "reconstructed_valid_anomaly_map")
    if plan.model_id == "super_add":
        if raw_image_score is None:
            raise ValueError("Non-tiled SuperADD deployment inference must provide its raw native image score.")
        return DecisionScore(float(raw_image_score), SUPERADD_NATIVE_IMAGE_SCORE_SEMANTIC, "superadd_raw_native_image_score")
    return _native_postprocessed(postprocessed_image_score)


def require_matching_score_semantic(score: DecisionScore, expected_semantic: str) -> None:
    """Reject an image decision whose score domain differs from its saved threshold."""
    if not expected_semantic:
        raise ValueError("Deployment decision threshold must declare a score semantic.")
    if score.semantic != expected_semantic:
        raise ValueError(
            "Decision score semantic does not match the deployment threshold: "
            f"score={score.semantic}, threshold={expected_semantic}."
        )


def _native_postprocessed(value: float | None) -> DecisionScore:
    if value is None:
        raise ValueError("Non-tiled deployment inference must provide an Anomalib postprocessed image score.")
    return DecisionScore(float(value), POSTPROCESSED_SCORE_SEMANTIC, "anomalib_postprocessed_image_score")


def _reconstructed_map_score(
    reconstructed_map: ReconstructedAnomalyMap | None,
    preprocessing_pipeline: PreprocessingPipeline | None,
    source: str,
) -> DecisionScore:
    if reconstructed_map is None or preprocessing_pipeline is None:
        raise ValueError("Tiled or legacy deployment inference must provide a reconstructed valid-area anomaly map.")
    return DecisionScore(
        preprocessing_pipeline.score_from_reconstructed_map(reconstructed_map),
        (
            "legacy_v2_valid_map_aggregation_v1"
            if source == "legacy_valid_anomaly_map"
            else "reconstructed_valid_map_aggregation_v3"
        ),
        source,
    )