"""Tests for deterministic image preprocessing profile metadata."""

from __future__ import annotations

import pytest

from app.models.image_preprocessing import (
    BorderMode,
    ColorMode,
    ImagePreprocessingConfig,
    MorphologyOperation,
    PreprocessingPreset,
    SmoothingFilter,
)


def test_legacy_none_profile_has_no_operations_and_round_trips() -> None:
    profile = ImagePreprocessingConfig()

    assert profile.is_legacy_none
    assert profile.to_dict()["operations"] == []
    assert ImagePreprocessingConfig.from_dict(profile.to_dict()) == profile


def test_gaussian_disk_profile_serializes_its_explicit_operation_order_and_physical_size() -> None:
    profile = ImagePreprocessingConfig(
        profile_id="surface-v1",
        color_mode=ColorMode.GRAYSCALE_REPLICATED_RGB,
        smoothing_filter=SmoothingFilter.GAUSSIAN_BLUR,
        gaussian_sigma=1.0,
        gaussian_kernel_size=7,
        smoothing_border_mode=BorderMode.REFLECT,
        morphology_operation=MorphologyOperation.DISK_OPENING,
        disk_radius=2,
        disk_iterations=1,
        morphology_border_mode=BorderMode.REPLICATE,
        pixels_per_millimetre=10.0,
    )

    payload = profile.to_dict()

    assert [operation["type"] for operation in payload["operations"]] == [
        "grayscale",
        "gaussian_blur",
        "disk_morphological_opening",
    ]
    assert payload["operations"][-1]["diameter"] == 5
    assert payload["operations"][-1]["diameter_mm"] == pytest.approx(0.5)
    assert ImagePreprocessingConfig.from_dict(payload) == profile


@pytest.mark.parametrize(
    "profile, message",
    [
        (ImagePreprocessingConfig(box_kernel_width=2), "Box blur kernel width"),
        (ImagePreprocessingConfig(gaussian_sigma=0), "Gaussian blur sigma"),
        (ImagePreprocessingConfig(median_kernel_size=2), "Median blur kernel size"),
        (
            ImagePreprocessingConfig(morphology_operation=MorphologyOperation.DISK_OPENING),
            "requires grayscale",
        ),
    ],
)
def test_invalid_image_preprocessing_parameters_are_rejected(profile: ImagePreprocessingConfig, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        profile.validate()


def test_presets_populate_explicit_controls_without_persisting_a_preset_only() -> None:
    profile = ImagePreprocessingConfig.from_preset(PreprocessingPreset.GRAYSCALE_GAUSSIAN_DISK_OPENING)

    assert profile.color_mode is ColorMode.GRAYSCALE_REPLICATED_RGB
    assert profile.smoothing_filter is SmoothingFilter.GAUSSIAN_BLUR
    assert profile.morphology_operation is MorphologyOperation.DISK_OPENING
    assert "preset" not in profile.to_dict()


def test_metadata_rejects_multiple_smoothing_operations() -> None:
    payload = ImagePreprocessingConfig(
        profile_id="gray-v1",
        color_mode=ColorMode.GRAYSCALE_REPLICATED_RGB,
        smoothing_filter=SmoothingFilter.BOX_BLUR,
    ).to_dict()
    payload["operations"].insert(
        2,
        {"type": "gaussian_blur", "sigma": 1.0, "kernel_size": 3, "border_mode": "reflect"},
    )

    with pytest.raises(ValueError, match="at most one smoothing"):
        ImagePreprocessingConfig.from_dict(payload)


@pytest.mark.parametrize("smoothing_filter", [SmoothingFilter.BOX_BLUR, SmoothingFilter.GAUSSIAN_BLUR, SmoothingFilter.MEDIAN_BLUR])
def test_all_smoothing_filters_round_trip_through_explicit_metadata(smoothing_filter: SmoothingFilter) -> None:
    profile = ImagePreprocessingConfig(
        profile_id=f"{smoothing_filter.value}-v1",
        smoothing_filter=smoothing_filter,
        gaussian_kernel_size=7 if smoothing_filter is SmoothingFilter.GAUSSIAN_BLUR else None,
    )

    assert ImagePreprocessingConfig.from_dict(profile.to_dict()) == profile