"""Reference preprocessing runner and fixed vectors for deployment verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from app.core.image_preprocessor import ImagePreprocessor
from app.core.preprocessing_pipeline import PreprocessingPipeline
from app.models.image_preprocessing import ImagePreprocessingConfig
from app.models.inspection_region import InspectionRegionConfig
from app.models.preprocessing_config import ResolvedPreprocessingPlan

GOLDEN_VECTOR_VERSION = 1


def preprocess_rgb(image_rgb: np.ndarray, profile: ImagePreprocessingConfig) -> np.ndarray:
    """Apply the exact trainer image-operation implementation to one rectified RGB image."""
    return ImagePreprocessor(profile).apply(image_rgb)


def prepare_model_inputs(
    source_rgb: np.ndarray,
    inspection_region: InspectionRegionConfig,
    plan: ResolvedPreprocessingPlan,
) -> tuple[np.ndarray, ...]:
    """Reproduce trainer ROI, image operations, and model-alignment padding for one raw source image."""
    return tuple(prepared.image_rgb for prepared in PreprocessingPipeline(inspection_region, plan).prepare_array(source_rgb))


def prepare_torch_model_inputs(
    source_rgb: np.ndarray,
    inspection_region: InspectionRegionConfig,
    plan: ResolvedPreprocessingPlan,
) -> tuple[object, ...]:
    """Return trainer-equivalent CHW uint8 tensors before Anomalib normalization."""
    import torch

    return tuple(
        torch.from_numpy(np.ascontiguousarray(np.moveaxis(values, -1, 0)))
        for values in prepare_model_inputs(source_rgb, inspection_region, plan)
    )


def verify_golden_vectors(path: Path) -> None:
    """Reject a profile implementation that differs from checked-in fixed input/output vectors."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("version") != GOLDEN_VECTOR_VERSION:
        raise ValueError("Unsupported preprocessing golden-vector version.")
    vectors = payload.get("vectors")
    if not isinstance(vectors, list) or not vectors:
        raise ValueError("Preprocessing golden vectors must contain at least one vector.")
    for vector in vectors:
        if not isinstance(vector, Mapping):
            raise ValueError("Preprocessing golden vector must be an object.")
        profile = ImagePreprocessingConfig.from_dict(vector.get("profile"))
        tolerance = int(vector.get("tolerance", 0))
        if "resolved_plan" not in vector:
            expected = np.asarray(vector.get("expected_output_rgb"), dtype=np.uint8)
            source = np.asarray(vector.get("input_rgb"), dtype=np.uint8)
            _verify_output(vector, preprocess_rgb(source, profile), expected, tolerance)
            continue
        inspection_region = InspectionRegionConfig.from_dict(vector.get("inspection_region"))
        plan = ResolvedPreprocessingPlan.from_dict(vector.get("resolved_plan"))
        if plan.image_preprocessing != profile:
            raise ValueError("Preprocessing golden vector profile does not match its resolved plan.")
        source = np.asarray(vector.get("input_raw_rgb"), dtype=np.uint8)
        expected_tiles = vector.get("expected_model_inputs_rgb")
        if not isinstance(expected_tiles, list):
            raise ValueError("Full preprocessing golden vector must contain expected model inputs.")
        actual_tiles = prepare_model_inputs(source, inspection_region, plan)
        if len(actual_tiles) != len(expected_tiles):
            raise ValueError("Preprocessing golden vector model-input count differs from the expected output.")
        for index, (actual, expected) in enumerate(zip(actual_tiles, expected_tiles, strict=True)):
            _verify_output(vector, actual, np.asarray(expected, dtype=np.uint8), tolerance, index)


def _verify_output(
    vector: Mapping[str, object],
    actual: np.ndarray,
    expected: np.ndarray,
    tolerance: int,
    tile_index: int | None = None,
) -> None:
    if actual.shape != expected.shape:
        raise ValueError("Preprocessing golden vector output dimensions differ from the expected output.")
    maximum_delta = int(np.abs(actual.astype(np.int16) - expected.astype(np.int16)).max(initial=0))
    if maximum_delta > tolerance:
        suffix = "" if tile_index is None else f" tile {tile_index}"
        raise ValueError(
            f"Preprocessing golden vector '{vector.get('id', 'unnamed')}'{suffix} failed: "
            f"maximum channel delta {maximum_delta} exceeds tolerance {tolerance}."
        )