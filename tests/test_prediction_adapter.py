"""Tests for strict conversion of Anomalib prediction structures."""

from __future__ import annotations

from math import nan

import numpy as np
import pytest

from app.core.prediction_adapter import (
    ANOMALIB_POSTPROCESSED_SCORE_SEMANTIC,
    LEGACY_VALID_MAP_SCORE_SEMANTIC,
    PreprocessedPredictionAccumulator,
    iter_anomalib_predictions,
    iter_preprocessed_predictions,
)
from app.core.preprocessing_pipeline import PreprocessingPipeline
from app.models.inspection_region import InspectionRegionConfig
from app.models.preprocessing_config import LEGACY_PREPROCESSING_CONTRACT_VERSION, PreprocessingConfig, TilingConfig


def test_prediction_adapter_requires_one_finite_score_per_image() -> None:
    output = [{"image_path": ["first.png", "second.png"], "pred_score": [0.1, 0.9]}]

    predictions = list(iter_anomalib_predictions(output))

    assert [prediction.score for prediction in predictions] == [0.1, 0.9]


def test_prediction_adapter_rejects_nonfinite_or_mismatched_output() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        list(iter_anomalib_predictions([{"image_path": ["first.png"], "pred_score": [nan]}]))
    with pytest.raises(ValueError, match="one image path and score"):
        list(iter_anomalib_predictions([{"image_path": ["first.png"], "pred_score": []}]))


def test_preprocessing_v3_adapter_preserves_the_native_full_image_score(tmp_path) -> None:
    source_path = (tmp_path / "source.png").resolve()
    staged_path = (tmp_path / "staged.png").resolve()
    pipeline = PreprocessingPipeline(
        InspectionRegionConfig(),
        PreprocessingConfig().resolve("dinomaly_dinov3", (639, 177)),
    )
    anomaly_map = np.zeros((192, 640), dtype=np.float32)
    anomaly_map[176, 638] = 0.7
    anomaly_map[191, 639] = 1.0
    output = {"image_path": [str(staged_path)], "pred_score": [1.0], "anomaly_map": [anomaly_map]}

    predictions = list(
        iter_preprocessed_predictions(
            output,
            {staged_path: source_path},
            {staged_path: pipeline.plan.tiles[0]},
            pipeline,
        )
    )

    assert len(predictions) == 1
    assert predictions[0].source_path == source_path
    assert predictions[0].score == pytest.approx(1.0)
    assert predictions[0].native_image_score == pytest.approx(1.0)
    assert predictions[0].score_semantic == ANOMALIB_POSTPROCESSED_SCORE_SEMANTIC
    assert predictions[0].anomaly_map.shape == (177, 639)


def test_preprocessing_v2_adapter_preserves_legacy_valid_map_score_semantics(tmp_path) -> None:
    source_path = (tmp_path / "source.png").resolve()
    staged_path = (tmp_path / "staged.png").resolve()
    pipeline = PreprocessingPipeline(
        InspectionRegionConfig(),
        PreprocessingConfig(preprocessing_contract_version=LEGACY_PREPROCESSING_CONTRACT_VERSION).resolve(
            "dinomaly_dinov3", (639, 177)
        ),
    )
    anomaly_map = np.zeros((192, 640), dtype=np.float32)
    anomaly_map[176, 638] = 0.7
    anomaly_map[191, 639] = 1.0

    prediction = next(
        iter_preprocessed_predictions(
            {"image_path": [str(staged_path)], "pred_score": [1.0], "anomaly_map": [anomaly_map]},
            {staged_path: source_path},
            {staged_path: pipeline.plan.tiles[0]},
            pipeline,
        )
    )

    assert prediction.score == pytest.approx(0.7)
    assert prediction.native_image_score is None
    assert prediction.score_semantic == LEGACY_VALID_MAP_SCORE_SEMANTIC


def test_preprocessing_v2_adapter_reconstructs_tiled_source_predictions(tmp_path) -> None:
    source_path = (tmp_path / "source.png").resolve()
    staged_paths = tuple((tmp_path / f"tile_{index}.png").resolve() for index in range(3))
    pipeline = PreprocessingPipeline(
        InspectionRegionConfig(),
        PreprocessingConfig(tiling=TilingConfig(enabled=True)).resolve("dinomaly_dinov3", (639, 177)),
    )
    output = {
        "image_path": [str(path) for path in staged_paths],
        "pred_score": [0.1, 0.2, 0.3],
        "anomaly_map": [np.full((192, 320), index + 1, dtype=np.float32) for index in range(3)],
    }

    predictions = list(
        iter_preprocessed_predictions(
            output,
            {path: source_path for path in staged_paths},
            {path: pipeline.plan.tiles[index] for index, path in enumerate(staged_paths)},
            pipeline,
        )
    )

    assert len(predictions) == 1
    assert predictions[0].score == 0.3
    assert predictions[0].native_tile_scores == (0.1, 0.2, 0.3)
    assert predictions[0].anomaly_map[20, 10] == 1
    assert predictions[0].anomaly_map[20, 200] == 2
    assert predictions[0].anomaly_map[20, 500] == 3


def test_preprocessed_prediction_accumulator_streams_a_source_after_its_last_tile(tmp_path) -> None:
    source_path = (tmp_path / "source.png").resolve()
    staged_paths = tuple((tmp_path / f"tile_{index}.png").resolve() for index in range(3))
    pipeline = PreprocessingPipeline(
        InspectionRegionConfig(),
        PreprocessingConfig(tiling=TilingConfig(enabled=True)).resolve("dinomaly_dinov3", (639, 177)),
    )
    accumulator = PreprocessedPredictionAccumulator(
        {path: source_path for path in staged_paths},
        {path: pipeline.plan.tiles[index] for index, path in enumerate(staged_paths)},
        pipeline,
    )

    first_batch = {
        "image_path": [str(staged_paths[0]), str(staged_paths[1])],
        "pred_score": [0.1, 0.2],
        "anomaly_map": [
            np.full((192, 320), 1, dtype=np.float32),
            np.full((192, 320), 2, dtype=np.float32),
        ],
    }
    final_batch = {
        "image_path": [str(staged_paths[2])],
        "pred_score": [0.3],
        "anomaly_map": [np.full((192, 320), 3, dtype=np.float32)],
    }

    assert list(accumulator.add_batch(first_batch)) == []
    streamed = list(accumulator.add_batch(final_batch))

    assert len(streamed) == 1
    assert streamed[0].source_path == source_path
    assert streamed[0].score == pytest.approx(0.3)
    accumulator.finalize()