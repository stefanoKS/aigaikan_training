"""Strict conversion of Anomalib prediction batches into application values."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from app.core.preprocessing_pipeline import PreprocessingPipeline
from app.models.preprocessing_config import LEGACY_PREPROCESSING_CONTRACT_VERSION, PreprocessingTile

ANOMALIB_POSTPROCESSED_SCORE_SEMANTIC = "anomalib_postprocessed_pred_score_v1"
LEGACY_VALID_MAP_SCORE_SEMANTIC = "legacy_v2_valid_map_aggregation_v1"
NATIVE_TILE_SCORE_SEMANTIC_PREFIX = "native_tile_score_aggregation"

@dataclass(frozen=True, slots=True)
class AnomalibPrediction:
    """One score and anomaly map returned by Anomalib for a source image."""

    image_path: Path
    score: float
    anomaly_map: Any
    score_semantic: str = ANOMALIB_POSTPROCESSED_SCORE_SEMANTIC


@dataclass(frozen=True, slots=True)
class PreprocessedAnomalibPrediction:
    """One source-image prediction reconstructed from one or more v2 model inputs."""

    source_path: Path
    score: float
    anomaly_map: Any
    staged_paths: tuple[Path, ...]
    native_image_score: float | None
    native_tile_scores: tuple[float, ...]
    score_semantic: str


class PreprocessedPredictionAccumulator:
    """Accumulate streamed tile batches until each source image is complete."""

    def __init__(
        self,
        source_path_by_staged_path: dict[Path, Path],
        preprocessing_tile_by_staged_path: dict[Path, PreprocessingTile],
        preprocessing_pipeline: PreprocessingPipeline,
    ) -> None:
        self._source_path_by_staged_path = source_path_by_staged_path
        self._preprocessing_tile_by_staged_path = preprocessing_tile_by_staged_path
        self._preprocessing_pipeline = preprocessing_pipeline
        self._tile_predictions_by_source: dict[Path, dict[int, AnomalibPrediction]] = {}
        self._staged_paths_by_source: dict[Path, dict[int, Path]] = {}
        self._completed_sources: set[Path] = set()
        self._expected_tile_indexes = tuple(tile.index for tile in preprocessing_pipeline.plan.tiles)

    def add_batch(self, output: Any) -> Iterator[PreprocessedAnomalibPrediction]:
        """Yield source predictions completed by one model prediction batch."""
        completed_sources: list[Path] = []
        for prediction in iter_anomalib_predictions(output):
            source_path = self._source_path_by_staged_path.get(prediction.image_path)
            tile = self._preprocessing_tile_by_staged_path.get(prediction.image_path)
            if source_path is None or tile is None:
                raise ValueError(f"Anomalib prediction path is outside the preprocessing-v2 staged inputs: {prediction.image_path}")
            if source_path in self._completed_sources:
                raise ValueError(f"Anomalib returned more than one completed prediction for: {source_path}")
            tile_predictions = self._tile_predictions_by_source.setdefault(source_path, {})
            if tile.index in tile_predictions:
                raise ValueError(f"Anomalib returned more than one prediction for preprocessing tile {tile.index}: {source_path}")
            tile_predictions[tile.index] = prediction
            self._staged_paths_by_source.setdefault(source_path, {})[tile.index] = prediction.image_path
            if tuple(sorted(tile_predictions)) == self._expected_tile_indexes:
                completed_sources.append(source_path)

        for source_path in completed_sources:
            self._completed_sources.add(source_path)
            yield self._build_prediction(source_path)

    def finalize(self) -> None:
        """Reject an inference run that ends before every received source is complete."""
        if not self._tile_predictions_by_source:
            return
        source_path, tile_predictions = next(iter(self._tile_predictions_by_source.items()))
        missing_indexes = sorted(set(self._expected_tile_indexes) - set(tile_predictions))
        raise ValueError(f"Anomalib did not return preprocessing tiles {missing_indexes} for: {source_path}")

    def _build_prediction(self, source_path: Path) -> PreprocessedAnomalibPrediction:
        tile_predictions = self._tile_predictions_by_source.pop(source_path)
        staged_paths = self._staged_paths_by_source.pop(source_path)
        reconstructed = self._preprocessing_pipeline.reconstruct_anomaly_maps(
            tile_predictions[index].anomaly_map for index in self._expected_tile_indexes
        )
        native_tile_scores = tuple(tile_predictions[index].score for index in self._expected_tile_indexes)
        if self._preprocessing_pipeline.plan.preprocessing_contract_version == LEGACY_PREPROCESSING_CONTRACT_VERSION:
            score = self._preprocessing_pipeline.score_from_reconstructed_map(reconstructed)
            native_image_score = None
            score_semantic = LEGACY_VALID_MAP_SCORE_SEMANTIC
        elif len(native_tile_scores) == 1:
            score = native_tile_scores[0]
            native_image_score = score
            score_semantic = ANOMALIB_POSTPROCESSED_SCORE_SEMANTIC
        else:
            score = self._preprocessing_pipeline.aggregate_tile_scores(native_tile_scores)
            native_image_score = None
            score_semantic = f"{NATIVE_TILE_SCORE_SEMANTIC_PREFIX}_{self._preprocessing_pipeline.plan.score_aggregation.value}_v1"
        return PreprocessedAnomalibPrediction(
            source_path=source_path,
            score=score,
            anomaly_map=reconstructed.anomaly_map,
            staged_paths=tuple(staged_paths[index] for index in self._expected_tile_indexes),
            native_image_score=native_image_score,
            native_tile_scores=native_tile_scores,
            score_semantic=score_semantic,
        )


def iter_anomalib_predictions(output: Any) -> Iterator[AnomalibPrediction]:
    """Yield finite, path-addressable predictions or fail instead of guessing values."""
    for batch in _prediction_batches(output):
        paths = _as_list(_batch_value(batch, "image_path"))
        scores = _as_list(_batch_value(batch, "pred_score"))
        anomaly_maps = _as_list(_batch_value(batch, "anomaly_map"))
        if not paths or len(paths) != len(scores):
            raise ValueError("Anomalib prediction output must contain one image path and score per image.")
        for index, raw_path in enumerate(paths):
            score = _as_float(scores[index])
            if not isfinite(score):
                raise ValueError(f"Anomalib returned a non-finite score for {raw_path}")
            yield AnomalibPrediction(
                image_path=Path(str(raw_path)).expanduser().resolve(),
                score=score,
                anomaly_map=anomaly_maps[index] if index < len(anomaly_maps) else None,
            )


def iter_preprocessed_predictions(
    output: Any,
    source_path_by_staged_path: dict[Path, Path],
    preprocessing_tile_by_staged_path: dict[Path, PreprocessingTile],
    preprocessing_pipeline: PreprocessingPipeline,
) -> Iterator[PreprocessedAnomalibPrediction]:
    """Reconstruct source scores and maps from preprocessing-v2 full images or tiles."""
    grouped: dict[Path, dict[int, AnomalibPrediction]] = {}
    staged_paths_by_source: dict[Path, dict[int, Path]] = {}
    for prediction in iter_anomalib_predictions(output):
        source_path = source_path_by_staged_path.get(prediction.image_path)
        tile = preprocessing_tile_by_staged_path.get(prediction.image_path)
        if source_path is None or tile is None:
            raise ValueError(f"Anomalib prediction path is outside the preprocessing-v2 staged inputs: {prediction.image_path}")
        if tile.index in grouped.setdefault(source_path, {}):
            raise ValueError(f"Anomalib returned more than one prediction for preprocessing tile {tile.index}: {source_path}")
        grouped[source_path][tile.index] = prediction
        staged_paths_by_source.setdefault(source_path, {})[tile.index] = prediction.image_path
    expected_indexes = tuple(tile.index for tile in preprocessing_pipeline.plan.tiles)
    for source_path in sorted(grouped, key=lambda path: str(path).casefold()):
        tile_predictions = grouped[source_path]
        if tuple(sorted(tile_predictions)) != expected_indexes:
            raise ValueError(f"Anomalib did not return every preprocessing tile for: {source_path}")
        reconstructed = preprocessing_pipeline.reconstruct_anomaly_maps(
            tile_predictions[index].anomaly_map for index in expected_indexes
        )
        native_tile_scores = tuple(tile_predictions[index].score for index in expected_indexes)
        if preprocessing_pipeline.plan.preprocessing_contract_version == LEGACY_PREPROCESSING_CONTRACT_VERSION:
            score = preprocessing_pipeline.score_from_reconstructed_map(reconstructed)
            native_image_score = None
            score_semantic = LEGACY_VALID_MAP_SCORE_SEMANTIC
        elif len(native_tile_scores) == 1:
            score = native_tile_scores[0]
            native_image_score = score
            score_semantic = ANOMALIB_POSTPROCESSED_SCORE_SEMANTIC
        else:
            score = preprocessing_pipeline.aggregate_tile_scores(native_tile_scores)
            native_image_score = None
            score_semantic = f"{NATIVE_TILE_SCORE_SEMANTIC_PREFIX}_{preprocessing_pipeline.plan.score_aggregation.value}_v1"
        yield PreprocessedAnomalibPrediction(
            source_path=source_path,
            score=score,
            anomaly_map=reconstructed.anomaly_map,
            staged_paths=tuple(staged_paths_by_source[source_path][index] for index in expected_indexes),
            native_image_score=native_image_score,
            native_tile_scores=native_tile_scores,
            score_semantic=score_semantic,
        )


def _prediction_batches(output: Any) -> Iterable[Any]:
    if output is None:
        return ()
    if isinstance(output, (list, tuple)):
        return tuple(item for batch in output for item in batch) if output and isinstance(output[0], list) else output
    return (output,)


def _batch_value(batch: Any, name: str) -> Any:
    if isinstance(batch, dict):
        return batch.get(name)
    return getattr(batch, name, None)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if getattr(value, "ndim", 0) == 0:
            return [value]
        return list(value)
    return [value]


def _as_float(value: Any) -> float:
    if hasattr(value, "item"):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Anomalib returned a non-numeric prediction score: {value!r}") from exc