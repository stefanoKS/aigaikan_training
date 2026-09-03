"""Tests for the single semantic-safe image decision score resolver."""

from __future__ import annotations

import numpy as np
import pytest

from app.core.decision_score import require_matching_score_semantic, resolve_decision_score
from app.core.preprocessing_pipeline import PreprocessingPipeline, ReconstructedAnomalyMap
from app.core.prediction_contract import POSTPROCESSED_SCORE_SEMANTIC, SUPERADD_NATIVE_IMAGE_SCORE_SEMANTIC
from app.models.inspection_region import InspectionRegionConfig
from app.models.preprocessing_config import LEGACY_PREPROCESSING_CONTRACT_VERSION, PreprocessingConfig, TilingConfig


def test_non_tiled_ordinary_model_uses_native_postprocessed_score() -> None:
    plan = PreprocessingConfig().resolve("patchcore", (7, 5))

    score = resolve_decision_score(plan, postprocessed_image_score=0.4, raw_image_score=2.0)

    assert score.value == pytest.approx(0.4)
    assert score.semantic == POSTPROCESSED_SCORE_SEMANTIC
    assert score.source == "anomalib_postprocessed_image_score"


def test_non_tiled_superadd_uses_raw_native_score_without_unit_interval_clamping() -> None:
    plan = PreprocessingConfig().resolve("super_add", (7, 5))

    score = resolve_decision_score(plan, postprocessed_image_score=1.0, raw_image_score=1.7)

    assert score.value == pytest.approx(1.7)
    assert score.semantic == SUPERADD_NATIVE_IMAGE_SCORE_SEMANTIC
    with pytest.raises(ValueError, match="does not match"):
        require_matching_score_semantic(score, POSTPROCESSED_SCORE_SEMANTIC)


@pytest.mark.parametrize("legacy", [False, True])
def test_tiled_and_legacy_scores_use_reconstructed_valid_area_maps(legacy: bool) -> None:
    config = PreprocessingConfig(
        preprocessing_contract_version=LEGACY_PREPROCESSING_CONTRACT_VERSION if legacy else 3,
        tiling=TilingConfig(enabled=not legacy),
    )
    plan = config.resolve("dinomaly_dinov3", (639, 177))
    pipeline = PreprocessingPipeline(InspectionRegionConfig(), plan)
    reconstructed = ReconstructedAnomalyMap(
        anomaly_map=np.full((177, 639), 0.8 if legacy else 0.9, dtype=np.float32),
        valid_mask=np.ones((177, 639), dtype=bool),
    )

    score = resolve_decision_score(
        plan,
        postprocessed_image_score=0.1,
        raw_image_score=None,
        reconstructed_map=reconstructed,
        preprocessing_pipeline=pipeline,
    )

    assert score.value == pytest.approx(0.8 if legacy else 0.9)
    assert score.source in {"legacy_valid_anomaly_map", "reconstructed_valid_anomaly_map"}


def test_superadd_missing_raw_native_score_fails_closed() -> None:
    plan = PreprocessingConfig().resolve("super_add", (7, 5))

    with pytest.raises(ValueError, match="raw native"):
        resolve_decision_score(plan, postprocessed_image_score=1.0, raw_image_score=None)