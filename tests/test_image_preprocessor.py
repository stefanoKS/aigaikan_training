"""Tests for shared deterministic ROI image preprocessing operations."""

from __future__ import annotations

import cv2
import numpy as np

from app.core.image_preprocessor import ImagePreprocessor
from app.core.inspection_region import InspectionRegionProcessor
from app.core.dataset_manifest import EffectiveSplit, stage_effective_split
from app.core.preprocessing_pipeline import PreprocessingPipeline
from app.models.image_preprocessing import (
    BorderMode,
    ColorMode,
    ImagePreprocessingConfig,
    MorphologyOperation,
    SmoothingFilter,
)
from app.models.inspection_region import InspectionRegionConfig
from app.models.preprocessing_config import PreprocessingConfig
from app.models.dataset_config import DatasetConfig, DatasetRole
from app.workers.inference_worker import _stage_preprocessed_inputs


def test_legacy_none_profile_preserves_rgb_pixels_exactly() -> None:
    source = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)

    processed = ImagePreprocessor(ImagePreprocessingConfig()).apply(source)

    assert np.array_equal(processed, source)


def test_grayscale_profile_replicates_one_documented_luminance_channel() -> None:
    source = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)

    processed = ImagePreprocessor(
        ImagePreprocessingConfig(profile_id="gray-v1", color_mode=ColorMode.GRAYSCALE_REPLICATED_RGB)
    ).apply(source)

    expected = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    assert np.array_equal(processed[:, :, 0], expected)
    assert np.array_equal(processed[:, :, 0], processed[:, :, 1])
    assert np.array_equal(processed[:, :, 1], processed[:, :, 2])


def test_disk_opening_uses_an_elliptical_kernel_after_gaussian_smoothing() -> None:
    profile = ImagePreprocessingConfig(
        profile_id="surface-v1",
        color_mode=ColorMode.GRAYSCALE_REPLICATED_RGB,
        smoothing_filter=SmoothingFilter.GAUSSIAN_BLUR,
        gaussian_sigma=1.0,
        gaussian_kernel_size=3,
        smoothing_border_mode=BorderMode.REFLECT,
        morphology_operation=MorphologyOperation.DISK_OPENING,
        disk_radius=2,
        morphology_border_mode=BorderMode.REPLICATE,
    )
    source = np.zeros((9, 9, 3), dtype=np.uint8)
    source[4, 4] = (255, 255, 255)

    processed = ImagePreprocessor(profile).apply(source)
    kernel = ImagePreprocessor.disk_kernel(2)

    grayscale = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (3, 3), sigmaX=1.0, sigmaY=1.0, borderType=cv2.BORDER_REFLECT_101)
    expected = cv2.dilate(
        cv2.erode(blurred, kernel, borderType=cv2.BORDER_REPLICATE),
        kernel,
        borderType=cv2.BORDER_REPLICATE,
    )
    assert kernel[0, 0] == 0
    assert np.array_equal(processed[:, :, 0], expected)
    assert np.array_equal(processed[:, :, 0], processed[:, :, 2])


def test_pipeline_preview_and_training_tiles_share_the_same_preprocessed_pixels() -> None:
    profile = ImagePreprocessingConfig(
        profile_id="gaussian-v1",
        color_mode=ColorMode.GRAYSCALE_REPLICATED_RGB,
        smoothing_filter=SmoothingFilter.GAUSSIAN_BLUR,
        gaussian_sigma=1.0,
        gaussian_kernel_size=3,
    )
    pipeline = PreprocessingPipeline(
        InspectionRegionConfig(),
        PreprocessingConfig(image_preprocessing=profile).resolve("patchcore", (7, 5)),
    )
    source = np.arange(7 * 5 * 3, dtype=np.uint8).reshape(5, 7, 3)

    rectified, preview, difference = pipeline.preview_arrays(source)
    prepared = pipeline.prepare_array(source)[0]

    assert np.array_equal(prepared.image_rgb[:5, :7], preview)
    assert np.array_equal(difference, cv2.absdiff(rectified, preview))


def test_pipeline_orders_roi_then_operations_then_padding_with_dynamic_quad_geometry() -> None:
    inspection_region = InspectionRegionConfig(
        enabled=True,
        source_width=20,
        source_height=12,
        points_px=((2, 1), (17, 1), (17, 10), (2, 10)),
    )
    profile = ImagePreprocessingConfig(
        profile_id="gray-v1",
        color_mode=ColorMode.GRAYSCALE_REPLICATED_RGB,
    )
    pipeline = PreprocessingPipeline(
        inspection_region,
        PreprocessingConfig(image_preprocessing=profile).resolve("dinomaly_dinov3", inspection_region.rectified_size()),
    )
    source = np.zeros((12, 20, 3), dtype=np.uint8)
    source[1:11, 2:18] = (255, 0, 0)

    prepared, rectified = pipeline.prepare_array_with_rectified(source)
    expected_rectified = InspectionRegionProcessor(inspection_region).apply(source)
    expected_preprocessed = ImagePreprocessor(profile).apply(expected_rectified)

    assert rectified.shape[1::-1] == inspection_region.rectified_size()
    assert np.array_equal(rectified, expected_rectified)
    assert np.array_equal(prepared[0].image_rgb[: rectified.shape[0], : rectified.shape[1]], expected_preprocessed)
    assert not prepared[0].valid_mask[-1, -1]


def test_training_calibration_final_test_and_interactive_inference_stage_identical_preprocessed_tensors(tmp_path) -> None:
    from PIL import Image

    profile = ImagePreprocessingConfig(
        profile_id="gray-v1",
        color_mode=ColorMode.GRAYSCALE_REPLICATED_RGB,
    )
    pipeline = PreprocessingPipeline(
        InspectionRegionConfig(),
        PreprocessingConfig(image_preprocessing=profile).resolve("patchcore", (9, 7)),
    )
    source_paths = []
    for role, color in (("training", (10, 20, 30)), ("validation", (40, 50, 60)), ("final", (70, 80, 90))):
        path = tmp_path / f"{role}.png"
        Image.new("RGB", (9, 7), color).save(path)
        source_paths.append(path.resolve())
    training_path, validation_path, final_path = source_paths
    split = EffectiveSplit(
        training_ok=(training_path,),
        validation_ok=(validation_path,),
        validation_ng=(),
        final_test_ok=(final_path,),
        final_test_ng=(),
        seed=42,
    )
    staged = stage_effective_split(split, DatasetConfig(), tmp_path / "snapshot", pipeline)

    for staged_path, source_path in staged.source_path_by_staged_path.items():
        with Image.open(staged_path) as image:
            staged_tensor = np.asarray(image.convert("RGB"))
        assert np.array_equal(staged_tensor, pipeline.prepare_path(source_path)[0].image_rgb)

    prepared_directory, mappings, _tiles, _previews = _stage_preprocessed_inputs(
        (training_path,), pipeline, tmp_path / "interactive"
    )
    interactive_path = next(prepared_directory.glob("*.png"))
    with Image.open(interactive_path) as image:
        interactive_tensor = np.asarray(image.convert("RGB"))
    assert mappings[interactive_path] == training_path
    assert np.array_equal(interactive_tensor, pipeline.prepare_path(training_path)[0].image_rgb)