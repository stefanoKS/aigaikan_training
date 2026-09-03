"""Strict conversion of Anomalib prediction batches into application values."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.core.prediction_contract import (
    POSTPROCESSED_SCORE_SEMANTIC,
    RAW_SCORE_SEMANTIC,
    SUPERADD_NATIVE_IMAGE_SCORE_SEMANTIC,
    validate_postprocessed_values,
)
from app.core.preprocessing_pipeline import PreprocessingPipeline
from app.models.preprocessing_config import LEGACY_PREPROCESSING_CONTRACT_VERSION, PreprocessingTile

ANOMALIB_POSTPROCESSED_SCORE_SEMANTIC = POSTPROCESSED_SCORE_SEMANTIC
LEGACY_VALID_MAP_SCORE_SEMANTIC = "legacy_v2_valid_map_aggregation_v1"
RECONSTRUCTED_VALID_MAP_SCORE_SEMANTIC = "reconstructed_valid_map_aggregation_v3"
NATIVE_TILE_SCORE_SEMANTIC_PREFIX = "native_tile_score_aggregation"

@dataclass(frozen=True, slots=True)
class AnomalibPrediction:
    """One score and anomaly map returned by Anomalib for a source image."""

    image_path: Path
    score: float
    anomaly_map: Any
    score_semantic: str = ANOMALIB_POSTPROCESSED_SCORE_SEMANTIC
    raw_image_score: float | None = None
    raw_anomaly_map: Any = None
    postprocessed_image_score: float | None = None
    postprocessed_score_semantic: str = ANOMALIB_POSTPROCESSED_SCORE_SEMANTIC
    postprocessed_anomaly_map: Any = None


@dataclass(frozen=True, slots=True)
class PostprocessedPredictionBatch:
    """A prediction batch whose native postprocessor has run exactly once."""

    output: Any
    raw_scores: tuple[Any, ...]
    raw_anomaly_maps: tuple[Any, ...]


@dataclass(slots=True)
class _PostprocessingSnapshot:
    """Raw values captured for one concrete callback output."""

    token: str
    raw_scores: tuple[Any, ...]
    raw_anomaly_maps: tuple[Any, ...]


class ExplicitPredictionPostProcessor:
    """Capture raw output and apply Anomalib postprocessing once regardless of callback order."""

    def __init__(self, post_processor: Any) -> None:
        if post_processor is None:
            raise ValueError("Inference model does not provide an Anomalib postprocessor.")
        if not bool(getattr(post_processor, "enable_normalization", False)):
            raise ValueError("Inference requires the checkpoint's enabled Anomalib postprocessor normalization.")
        self._post_processor = post_processor
        self._original_post_process_batch = post_processor.post_process_batch
        self._snapshot_token = uuid4().hex
        post_processor.post_process_batch = self._post_process_once

    def postprocess(self, output: Any) -> PostprocessedPredictionBatch:
        """Return one verified postprocessed callback output with its preprocessor snapshot."""
        snapshot = self._post_process_once(output)
        return PostprocessedPredictionBatch(output, snapshot.raw_scores, snapshot.raw_anomaly_maps)

    def postprocess_output(self, output: Any) -> Any:
        """Wrap a returned Engine.predict tree with raw snapshots and verified postprocessed batches."""
        if isinstance(output, list):
            return [self.postprocess_output(item) for item in output]
        if isinstance(output, tuple):
            return tuple(self.postprocess_output(item) for item in output)
        return self.postprocess(output)

    def close(self) -> None:
        """Restore the native callback implementation and release per-batch guards."""
        self._post_processor.post_process_batch = self._original_post_process_batch

    def _post_process_once(self, output: Any) -> _PostprocessingSnapshot:
        snapshot = _get_postprocessing_snapshot(output)
        if snapshot is not None:
            if snapshot.token == self._snapshot_token:
                return snapshot
            raise RuntimeError("Prediction output is already managed by another Anomalib postprocessor wrapper.")
        raw_scores = tuple(_as_list(_batch_value(output, "pred_score")))
        raw_anomaly_maps = tuple(_as_list(_batch_value(output, "anomaly_map")))
        snapshot = _PostprocessingSnapshot(self._snapshot_token, raw_scores, raw_anomaly_maps)
        _set_postprocessing_snapshot(output, snapshot)
        try:
            self._original_post_process_batch(output)
        except Exception:
            _clear_postprocessing_snapshot(output, snapshot)
            raise
        return snapshot


def explicitly_postprocessed_predict(engine: Any, model: Any, **kwargs: Any) -> Any:
    """Run prediction with one explicit native postprocessing invocation per returned batch.

    Lightweight test doubles that do not model Anomalib components retain their existing direct-output behavior.
    """
    post_processor = getattr(model, "post_processor", None)
    if post_processor is None:
        return engine.predict(model=model, **kwargs)
    processor = ExplicitPredictionPostProcessor(post_processor)
    try:
        output = engine.predict(model=model, **kwargs)
        return processor.postprocess_output(output)
    finally:
        processor.close()


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
    valid_roi_mask: Any
    postprocessed_image_score: float | None
    postprocessed_score_semantic: str
    raw_image_score: float | None = None
    raw_anomaly_map: Any = None


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
        native_tile_scores = tuple(
            tile_predictions[index].postprocessed_image_score
            if tile_predictions[index].postprocessed_image_score is not None
            else tile_predictions[index].score
            for index in self._expected_tile_indexes
        )
        raw_maps = tuple(tile_predictions[index].raw_anomaly_map for index in self._expected_tile_indexes)
        raw_anomaly_map = (
            self._preprocessing_pipeline.reconstruct_anomaly_maps(raw_maps).anomaly_map
            if all(anomaly_map is not None for anomaly_map in raw_maps)
            else None
        )
        raw_scores = tuple(tile_predictions[index].raw_image_score for index in self._expected_tile_indexes)
        postprocessed_image_score = score = native_tile_scores[0]
        postprocessed_score_semantic = ANOMALIB_POSTPROCESSED_SCORE_SEMANTIC
        if self._preprocessing_pipeline.plan.preprocessing_contract_version == LEGACY_PREPROCESSING_CONTRACT_VERSION:
            score = self._preprocessing_pipeline.score_from_reconstructed_map(reconstructed)
            native_image_score = None
            score_semantic = LEGACY_VALID_MAP_SCORE_SEMANTIC
            postprocessed_image_score = score
        elif self._preprocessing_pipeline.plan.tiled:
            score = self._preprocessing_pipeline.score_from_reconstructed_map(reconstructed)
            native_image_score = None
            score_semantic = RECONSTRUCTED_VALID_MAP_SCORE_SEMANTIC
            postprocessed_image_score = score
        elif self._preprocessing_pipeline.plan.model_id == "super_add":
            if len(raw_scores) != 1 or raw_scores[0] is None:
                raise ValueError("SuperADD prediction is missing its native top-quantile image score.")
            score = raw_scores[0]
            native_image_score = native_tile_scores[0]
            score_semantic = SUPERADD_NATIVE_IMAGE_SCORE_SEMANTIC
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
            valid_roi_mask=reconstructed.valid_mask,
            postprocessed_image_score=postprocessed_image_score,
            postprocessed_score_semantic=postprocessed_score_semantic,
            raw_image_score=raw_scores[0] if len(raw_scores) == 1 else None,
            raw_anomaly_map=raw_anomaly_map,
        )


def iter_anomalib_predictions(output: Any) -> Iterator[AnomalibPrediction]:
    """Yield finite, path-addressable predictions or fail instead of guessing values."""
    for batch in _prediction_batches(output):
        paths = _as_list(_batch_value(batch, "image_path"))
        scores = _as_list(_batch_value(batch, "pred_score"))
        anomaly_maps = _as_list(_batch_value(batch, "anomaly_map"))
        raw_scores = list(batch.raw_scores) if isinstance(batch, PostprocessedPredictionBatch) else []
        raw_anomaly_maps = list(batch.raw_anomaly_maps) if isinstance(batch, PostprocessedPredictionBatch) else []
        if not paths or len(paths) != len(scores):
            raise ValueError("Anomalib prediction output must contain one image path and score per image.")
        for index, raw_path in enumerate(paths):
            score = _as_float(scores[index])
            if not isfinite(score):
                raise ValueError(f"Anomalib returned a non-finite score for {raw_path}")
            anomaly_map = anomaly_maps[index] if index < len(anomaly_maps) else None
            if isinstance(batch, PostprocessedPredictionBatch):
                validate_postprocessed_values(score, anomaly_map)
            yield AnomalibPrediction(
                image_path=Path(str(raw_path)).expanduser().resolve(),
                score=score,
                anomaly_map=anomaly_map,
                raw_image_score=_as_float(raw_scores[index]) if index < len(raw_scores) else None,
                raw_anomaly_map=raw_anomaly_maps[index] if index < len(raw_anomaly_maps) else None,
                postprocessed_image_score=score,
                postprocessed_anomaly_map=anomaly_map,
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
        native_tile_scores = tuple(
            tile_predictions[index].postprocessed_image_score
            if tile_predictions[index].postprocessed_image_score is not None
            else tile_predictions[index].score
            for index in expected_indexes
        )
        raw_maps = tuple(tile_predictions[index].raw_anomaly_map for index in expected_indexes)
        raw_anomaly_map = (
            preprocessing_pipeline.reconstruct_anomaly_maps(raw_maps).anomaly_map
            if all(anomaly_map is not None for anomaly_map in raw_maps)
            else None
        )
        raw_scores = tuple(tile_predictions[index].raw_image_score for index in expected_indexes)
        postprocessed_image_score = score = native_tile_scores[0]
        postprocessed_score_semantic = ANOMALIB_POSTPROCESSED_SCORE_SEMANTIC
        if preprocessing_pipeline.plan.preprocessing_contract_version == LEGACY_PREPROCESSING_CONTRACT_VERSION:
            score = preprocessing_pipeline.score_from_reconstructed_map(reconstructed)
            native_image_score = None
            score_semantic = LEGACY_VALID_MAP_SCORE_SEMANTIC
            postprocessed_image_score = score
        elif preprocessing_pipeline.plan.tiled:
            score = preprocessing_pipeline.score_from_reconstructed_map(reconstructed)
            native_image_score = None
            score_semantic = RECONSTRUCTED_VALID_MAP_SCORE_SEMANTIC
            postprocessed_image_score = score
        elif preprocessing_pipeline.plan.model_id == "super_add":
            if len(raw_scores) != 1 or raw_scores[0] is None:
                raise ValueError("SuperADD prediction is missing its native top-quantile image score.")
            score = raw_scores[0]
            native_image_score = native_tile_scores[0]
            score_semantic = SUPERADD_NATIVE_IMAGE_SCORE_SEMANTIC
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
            valid_roi_mask=reconstructed.valid_mask,
            postprocessed_image_score=postprocessed_image_score,
            postprocessed_score_semantic=postprocessed_score_semantic,
            raw_image_score=raw_scores[0] if len(raw_scores) == 1 else None,
            raw_anomaly_map=raw_anomaly_map,
        )


def _prediction_batches(output: Any) -> Iterable[Any]:
    if output is None:
        return ()
    if isinstance(output, (list, tuple)):
        return tuple(item for batch in output for item in batch) if output and isinstance(output[0], list) else output
    return (output,)


_POSTPROCESSING_SNAPSHOT_ATTRIBUTE = "_aigaikan_postprocessing_snapshot"


def _get_postprocessing_snapshot(output: Any) -> _PostprocessingSnapshot | None:
    snapshot = (
        output.get(_POSTPROCESSING_SNAPSHOT_ATTRIBUTE)
        if isinstance(output, dict)
        else getattr(output, _POSTPROCESSING_SNAPSHOT_ATTRIBUTE, None)
    )
    return snapshot if isinstance(snapshot, _PostprocessingSnapshot) else None


def _set_postprocessing_snapshot(output: Any, snapshot: _PostprocessingSnapshot) -> None:
    if isinstance(output, dict):
        output[_POSTPROCESSING_SNAPSHOT_ATTRIBUTE] = snapshot
        return
    try:
        setattr(output, _POSTPROCESSING_SNAPSHOT_ATTRIBUTE, snapshot)
    except (AttributeError, TypeError) as exc:
        raise TypeError("Anomalib prediction output must support application postprocessing state.") from exc


def _clear_postprocessing_snapshot(output: Any, snapshot: _PostprocessingSnapshot) -> None:
    if _get_postprocessing_snapshot(output) is not snapshot:
        return
    if isinstance(output, dict):
        output.pop(_POSTPROCESSING_SNAPSHOT_ATTRIBUTE, None)
        return
    delattr(output, _POSTPROCESSING_SNAPSHOT_ATTRIBUTE)


def _batch_value(batch: Any, name: str) -> Any:
    if isinstance(batch, PostprocessedPredictionBatch):
        batch = batch.output
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