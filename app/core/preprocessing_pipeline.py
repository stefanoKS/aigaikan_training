"""Shared image geometry, valid-mask scoring, and tile reconstruction for preprocessing v2."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image

from app.core.image_preprocessor import ImagePreprocessor
from app.core.inspection_region import InspectionRegionProcessor
from app.models.inspection_region import InspectionRegionConfig
from app.models.preprocessing_config import (
    LEGACY_PREPROCESSING_CONTRACT_VERSION,
    PreprocessingConfig,
    PreprocessingTile,
    ResolvedPreprocessingPlan,
    ScoreAggregation,
)


@dataclass(frozen=True, slots=True)
class PreparedImage:
    """One model-ready RGB image and the pixels that originated from the inspected ROI."""

    tile: PreprocessingTile
    image_rgb: np.ndarray
    valid_mask: np.ndarray


@dataclass(frozen=True, slots=True)
class ReconstructedAnomalyMap:
    """Anomaly map in rectified ROI coordinates and its complete valid-pixel coverage."""

    anomaly_map: np.ndarray
    valid_mask: np.ndarray


class PreprocessingPipeline:
    """Apply one resolved v2 contract identically before fitting, prediction, and deployment."""

    def __init__(self, inspection_region: InspectionRegionConfig, plan: ResolvedPreprocessingPlan) -> None:
        plan.validate()
        self.inspection_region = inspection_region
        self.inspection_region_processor = InspectionRegionProcessor(inspection_region)
        self.plan = plan
        self.image_preprocessor = ImagePreprocessor(plan.image_preprocessing)
        if inspection_region.enabled and inspection_region.rectified_size() != plan.rectified_size:
            raise ValueError("Preprocessing plan rectified size does not match the saved inspection ROI.")

    def prepare_path(self, source_path: Path) -> tuple[PreparedImage, ...]:
        """Decode a source file as RGB exactly once, then prepare model-ready inputs."""
        prepared, _rectified = self.prepare_path_with_rectified(source_path)
        return prepared

    def prepare_path_with_rectified(self, source_path: Path) -> tuple[tuple[PreparedImage, ...], np.ndarray]:
        """Decode once and retain the rectified RGB image for an optional visualization."""
        with Image.open(source_path) as image:
            return self.prepare_array_with_rectified(np.asarray(image.convert("RGB")))

    def prepare_mask_path(self, source_path: Path) -> tuple[np.ndarray, ...]:
        """Prepare a ground-truth mask with the same geometry as its model inputs."""
        with Image.open(source_path) as image:
            return self.prepare_mask_array(np.asarray(image.convert("L")))

    def prepare_array(self, image_rgb: np.ndarray) -> tuple[PreparedImage, ...]:
        """Rectify, pad, and optionally tile one RGB image without stretching its valid pixels."""
        prepared, _rectified = self.prepare_array_with_rectified(image_rgb)
        return prepared

    def prepare_array_with_rectified(self, image_rgb: np.ndarray) -> tuple[tuple[PreparedImage, ...], np.ndarray]:
        """Apply the v2 plan and retain its rectified pre-padding RGB image."""
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("Preprocessing v2 requires a three-channel RGB image.")
        rectified = self._rectify(image_rgb)
        preprocessed = self.preprocess_rectified(rectified)
        return tuple(self._prepare_tile(preprocessed, tile) for tile in self.plan.tiles), rectified

    def preprocess_rectified(self, rectified_rgb: np.ndarray) -> np.ndarray:
        """Apply the frozen image profile before any model-specific padding or tiling."""
        self._validate_rectified_size(rectified_rgb)
        return self.image_preprocessor.apply(rectified_rgb)

    def preview_arrays(self, image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return rectified, preprocessed, and fixed-range absolute-difference preview arrays."""
        rectified = self._rectify(image_rgb)
        preprocessed = self.preprocess_rectified(rectified)
        return rectified, preprocessed, self.image_preprocessor.absolute_difference(rectified, preprocessed)

    def prepare_mask_array(self, mask: np.ndarray) -> tuple[np.ndarray, ...]:
        """Rectify, pad, and tile a single-channel mask with nearest-neighbor semantics."""
        if mask.ndim != 2:
            raise ValueError("Preprocessing v2 masks must have exactly two dimensions.")
        rectified = self.inspection_region_processor.apply_mask(np.ascontiguousarray(mask))
        self._validate_rectified_size(rectified)
        outputs: list[np.ndarray] = []
        for tile in self.plan.tiles:
            x, y, width, height = tile.rectified_box
            cropped = np.ascontiguousarray(rectified[y : y + height, x : x + width])
            padded_width, padded_height = tile.padded_size
            padded = np.zeros((padded_height, padded_width), dtype=cropped.dtype)
            padded[:height, :width] = cropped
            output_width, output_height = tile.model_input_size
            if (padded_width, padded_height) != (output_width, output_height):
                padded = cv2.resize(padded, (output_width, output_height), interpolation=cv2.INTER_NEAREST)
            outputs.append(np.ascontiguousarray(padded))
        return tuple(outputs)

    def _rectify(self, image: np.ndarray) -> np.ndarray:
        rectified = self.inspection_region_processor.apply(np.ascontiguousarray(image))
        if not isinstance(rectified, np.ndarray):
            raise TypeError("Inspection ROI processing did not return an array.")
        self._validate_rectified_size(rectified)
        return rectified

    def _validate_rectified_size(self, rectified: np.ndarray) -> None:
        rectified_height, rectified_width = rectified.shape[:2]
        if (rectified_width, rectified_height) != self.plan.rectified_size:
            raise ValueError(
                "Preprocessing input geometry does not match the saved plan: "
                f"expected {self.plan.rectified_size[0]}x{self.plan.rectified_size[1]}, "
                f"received {rectified_width}x{rectified_height}."
            )

    def _prepare_tile(self, rectified: np.ndarray, tile: PreprocessingTile) -> PreparedImage:
        x, y, width, height = tile.rectified_box
        cropped = np.ascontiguousarray(rectified[y : y + height, x : x + width])
        padded_width, padded_height = tile.padded_size
        padded = np.full(
            (padded_height, padded_width, 3),
            self.plan.padding_value_rgb,
            dtype=cropped.dtype,
        )
        padded[:height, :width] = cropped
        valid_mask = np.zeros((padded_height, padded_width), dtype=bool)
        valid_mask[:height, :width] = True
        output_width, output_height = tile.model_input_size
        if (padded_width, padded_height) != (output_width, output_height):
            padded = cv2.resize(padded, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
            valid_mask = cv2.resize(
                valid_mask.astype(np.uint8),
                (output_width, output_height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        self._validate_tile_mask(tile, valid_mask)
        return PreparedImage(tile=tile, image_rgb=np.ascontiguousarray(padded), valid_mask=np.ascontiguousarray(valid_mask))

    @staticmethod
    def _validate_tile_mask(tile: PreprocessingTile, valid_mask: np.ndarray) -> None:
        x, y, width, height = tile.valid_box
        expected = np.zeros(valid_mask.shape, dtype=bool)
        expected[y : y + height, x : x + width] = True
        if not np.array_equal(valid_mask, expected):
            raise ValueError("Preprocessing valid-pixel mask does not match the saved tile geometry.")

    def score_from_anomaly_map(self, anomaly_map: Any, tile_index: int = 0) -> float:
        """Aggregate only valid anomaly-map pixels using the contract's persisted strategy."""
        tile = self.plan.tiles[tile_index]
        prepared_mask = self._prepared_mask(tile)
        values = self._map_at_model_resolution(anomaly_map, tile)
        valid_values = values[prepared_mask]
        if valid_values.size == 0 or not np.isfinite(valid_values).all():
            raise ValueError("Anomaly map contains no finite valid pixels.")
        return self._aggregate_values(valid_values)

    def aggregate_tile_scores(self, scores: Iterable[float]) -> float:
        """Aggregate one score per tile with the persisted image/part strategy."""
        values = np.asarray(tuple(float(score) for score in scores), dtype=np.float64)
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError("Tile scores must contain at least one finite value.")
        return self._aggregate_values(values)

    def reconstruct_anomaly_maps(self, anomaly_maps: Iterable[Any]) -> ReconstructedAnomalyMap:
        """Reassemble tile maps into source coordinates with seam-resistant v3 overlap blending."""
        values = tuple(anomaly_maps)
        if len(values) != len(self.plan.tiles):
            raise ValueError("Tile anomaly-map count does not match the preprocessing plan.")
        rectified_width, rectified_height = self.plan.rectified_size
        reconstructed = np.full((rectified_height, rectified_width), np.nan, dtype=np.float32)
        coverage = np.zeros((rectified_height, rectified_width), dtype=bool)
        weighted_sum = np.zeros((rectified_height, rectified_width), dtype=np.float64)
        weight_sum = np.zeros((rectified_height, rectified_width), dtype=np.float64)
        for tile, anomaly_map in zip(self.plan.tiles, values, strict=True):
            tile_map = self._map_at_model_resolution(anomaly_map, tile)
            padded_width, padded_height = tile.padded_size
            input_width, input_height = tile.model_input_size
            if (input_width, input_height) != (padded_width, padded_height):
                tile_map = cv2.resize(tile_map, (padded_width, padded_height), interpolation=cv2.INTER_LINEAR)
            x, y, width, height = tile.rectified_box
            valid_map = tile_map[:height, :width]
            if not np.isfinite(valid_map).all():
                raise ValueError("Tile anomaly map contains non-finite valid pixels.")
            destination = reconstructed[y : y + height, x : x + width]
            destination_coverage = coverage[y : y + height, x : x + width]
            if self.plan.preprocessing_contract_version == LEGACY_PREPROCESSING_CONTRACT_VERSION:
                destination[~destination_coverage] = valid_map[~destination_coverage]
                destination[destination_coverage] = np.maximum(destination[destination_coverage], valid_map[destination_coverage])
            else:
                weights = self._tile_blend_weights(tile)
                weighted_sum[y : y + height, x : x + width] += valid_map * weights
                weight_sum[y : y + height, x : x + width] += weights
            destination_coverage[:] = True
        if not coverage.all():
            raise ValueError("Tile reconstruction does not cover every rectified ROI pixel.")
        if self.plan.preprocessing_contract_version != LEGACY_PREPROCESSING_CONTRACT_VERSION:
            if (weight_sum[coverage] <= 0).any():
                raise ValueError("Tile reconstruction produced an uncovered blend region.")
            reconstructed[coverage] = (weighted_sum[coverage] / weight_sum[coverage]).astype(np.float32)
        return ReconstructedAnomalyMap(anomaly_map=reconstructed, valid_mask=coverage)

    @staticmethod
    def _tile_blend_weights(tile: PreprocessingTile) -> np.ndarray:
        """Use a deterministic center-weighted feather so tile joins have no ownership discontinuity."""
        _x, _y, width, height = tile.rectified_box
        horizontal = 1.0 - np.abs(((np.arange(width, dtype=np.float64) + 0.5) / width) * 2.0 - 1.0)
        vertical = 1.0 - np.abs(((np.arange(height, dtype=np.float64) + 0.5) / height) * 2.0 - 1.0)
        return np.outer(vertical, horizontal)

    def score_from_reconstructed_map(self, reconstructed: ReconstructedAnomalyMap) -> float:
        """Aggregate the deduplicated reconstructed map for source-image decisions."""
        if reconstructed.anomaly_map.shape != reconstructed.valid_mask.shape:
            raise ValueError("Reconstructed anomaly map and valid mask must share dimensions.")
        return self._aggregate_values(reconstructed.anomaly_map[reconstructed.valid_mask])

    def _prepared_mask(self, tile: PreprocessingTile) -> np.ndarray:
        width, height = tile.model_input_size
        x, y, valid_width, valid_height = tile.valid_box
        mask = np.zeros((height, width), dtype=bool)
        mask[y : y + valid_height, x : x + valid_width] = True
        return mask

    @staticmethod
    def _as_map_array(anomaly_map: Any) -> np.ndarray:
        values = anomaly_map.detach().float().cpu().numpy() if hasattr(anomaly_map, "detach") else np.asarray(anomaly_map)
        while values.ndim > 2:
            values = values[0]
        if values.ndim != 2 or values.size == 0:
            raise ValueError("Anomaly map must contain one non-empty two-dimensional image.")
        return np.ascontiguousarray(values.astype(np.float32))

    def _map_at_model_resolution(self, anomaly_map: Any, tile: PreprocessingTile) -> np.ndarray:
        values = self._as_map_array(anomaly_map)
        output_width, output_height = tile.model_input_size
        if values.shape != (output_height, output_width):
            values = cv2.resize(values, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
        return values

    def _aggregate_values(self, values: np.ndarray) -> float:
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            raise ValueError("Score aggregation requires at least one finite valid anomaly-map value.")
        if self.plan.score_aggregation is ScoreAggregation.MAX:
            return float(finite_values.max())
        count = max(1, ceil(finite_values.size * self.plan.top_k_fraction))
        return float(np.partition(finite_values, finite_values.size - count)[-count:].mean())


def resolve_preprocessing_plan(
    config: PreprocessingConfig,
    inspection_region: InspectionRegionConfig,
    model_id: str,
    source_paths: Iterable[Path],
) -> ResolvedPreprocessingPlan:
    """Resolve a project policy only after every source dimension is verified."""
    paths = tuple(Path(path).expanduser().resolve() for path in source_paths)
    if not paths:
        raise ValueError("Preprocessing v2 requires at least one source image.")
    processor = InspectionRegionProcessor(inspection_region)
    source_sizes: set[tuple[int, int]] = set()
    for path in paths:
        with Image.open(path) as image:
            source_size = image.size
        processor.validate_source_size(*source_size)
        source_sizes.add(source_size)
    if inspection_region.enabled:
        rectified_size = inspection_region.rectified_size()
    else:
        if len(source_sizes) != 1:
            sizes = ", ".join(f"{width}x{height}" for width, height in sorted(source_sizes))
            raise ValueError("Preprocessing without an inspection ROI requires one source resolution; received " f"{sizes}.")
        rectified_size = next(iter(source_sizes))
    return config.resolve(model_id, rectified_size)