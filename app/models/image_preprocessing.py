"""Versioned deterministic image operations applied after ROI rectification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import ceil, isfinite
from typing import Any, Mapping

IMAGE_PREPROCESSING_SCHEMA_VERSION = 1
LEGACY_NONE_PROFILE_ID = "legacy_none_v1"
CUSTOM_PROFILE_ID = "custom_v1"
LUMINANCE_STANDARD = "itu_r_bt601_full_range"
INPUT_COLOR_ORDER = "RGB"
INPUT_DTYPE = "uint8"
INPUT_RANGE = "0_255"
OPERATION_ORDER = "roi_then_image_operations_then_padding"


class ColorMode(StrEnum):
    """Supported color transforms before model alignment padding."""

    PRESERVE_RGB = "preserve_rgb"
    GRAYSCALE_REPLICATED_RGB = "grayscale_replicated_rgb"


class SmoothingFilter(StrEnum):
    """Supported deterministic smoothing filters."""

    NONE = "none"
    BOX_BLUR = "box_blur"
    GAUSSIAN_BLUR = "gaussian_blur"
    MEDIAN_BLUR = "median_blur"


class MorphologyOperation(StrEnum):
    """Supported grayscale morphology operations."""

    NONE = "none"
    DISK_OPENING = "disk_morphological_opening"


class BorderMode(StrEnum):
    """Curated OpenCV-compatible border extensions."""

    REFLECT = "reflect"
    REPLICATE = "replicate"


class PreprocessingPreset(StrEnum):
    """Convenience values which populate explicit profile operations."""

    NONE = "none"
    GRAYSCALE_ONLY = "grayscale_only"
    GRAYSCALE_GAUSSIAN = "grayscale_gaussian"
    GRAYSCALE_MEDIAN = "grayscale_median"
    GRAYSCALE_DISK_OPENING = "grayscale_disk_opening"
    GRAYSCALE_GAUSSIAN_DISK_OPENING = "grayscale_gaussian_disk_opening"


@dataclass(frozen=True, slots=True)
class ImagePreprocessingConfig:
    """Frozen, RGB-in/RGB-out deterministic image operation profile.

    All operations run on the perspective-rectified ROI before model-specific
    alignment and padding. The default reproduces historical behavior exactly.
    """

    schema_version: int = IMAGE_PREPROCESSING_SCHEMA_VERSION
    profile_id: str = LEGACY_NONE_PROFILE_ID
    color_mode: ColorMode = ColorMode.PRESERVE_RGB
    smoothing_filter: SmoothingFilter = SmoothingFilter.NONE
    box_kernel_width: int = 3
    box_kernel_height: int = 3
    gaussian_sigma: float = 1.0
    gaussian_kernel_size: int | None = None
    median_kernel_size: int = 3
    smoothing_border_mode: BorderMode = BorderMode.REFLECT
    morphology_operation: MorphologyOperation = MorphologyOperation.NONE
    disk_radius: int = 2
    disk_iterations: int = 1
    morphology_border_mode: BorderMode = BorderMode.REFLECT
    expected_maximum_fiber_thickness_px: float | None = None
    expected_minimum_defect_diameter_px: float | None = None
    pixels_per_millimetre: float | None = None

    @property
    def disk_diameter(self) -> int:
        """Return the saved disk diameter in rectified-image pixels."""
        return 2 * self.disk_radius + 1

    @property
    def resolved_gaussian_kernel_size(self) -> int:
        """Return an odd Gaussian support width when automatic sizing is selected."""
        return self.gaussian_kernel_size or (2 * ceil(3 * self.gaussian_sigma) + 1)

    @property
    def is_legacy_none(self) -> bool:
        """Return whether this profile has no image transform beyond historical behavior."""
        return (
            self.profile_id == LEGACY_NONE_PROFILE_ID
            and self.color_mode is ColorMode.PRESERVE_RGB
            and self.smoothing_filter is SmoothingFilter.NONE
            and self.morphology_operation is MorphologyOperation.NONE
        )

    def validate(self) -> None:
        """Reject invalid or ambiguous image-transform metadata."""
        if self.schema_version != IMAGE_PREPROCESSING_SCHEMA_VERSION:
            raise ValueError(f"Unsupported image preprocessing schema version: {self.schema_version}")
        if not self.profile_id:
            raise ValueError("Image preprocessing profile ID must not be empty.")
        if self.box_kernel_width <= 0 or self.box_kernel_width % 2 == 0:
            raise ValueError("Box blur kernel width must be positive and odd.")
        if self.box_kernel_height <= 0 or self.box_kernel_height % 2 == 0:
            raise ValueError("Box blur kernel height must be positive and odd.")
        if not isfinite(self.gaussian_sigma) or self.gaussian_sigma <= 0:
            raise ValueError("Gaussian blur sigma must be positive and finite.")
        if self.gaussian_kernel_size is not None and (
            self.gaussian_kernel_size <= 0 or self.gaussian_kernel_size % 2 == 0
        ):
            raise ValueError("Gaussian blur kernel size must be positive and odd when specified.")
        if self.median_kernel_size <= 0 or self.median_kernel_size % 2 == 0:
            raise ValueError("Median blur kernel size must be positive and odd.")
        if self.morphology_operation is MorphologyOperation.DISK_OPENING:
            if self.color_mode is not ColorMode.GRAYSCALE_REPLICATED_RGB:
                raise ValueError("Disk morphological opening requires grayscale replicated RGB mode.")
            if self.disk_radius <= 0:
                raise ValueError("Disk opening radius must be positive.")
            if self.disk_iterations <= 0:
                raise ValueError("Disk opening iterations must be positive.")
        for name, value in (
            ("Expected maximum fiber thickness", self.expected_maximum_fiber_thickness_px),
            ("Expected minimum defect diameter", self.expected_minimum_defect_diameter_px),
            ("Pixels per millimetre", self.pixels_per_millimetre),
        ):
            if value is not None and (not isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive and finite when provided.")

    def warnings(self) -> tuple[str, ...]:
        """Return evidence-based advisory messages without inferring defect geometry."""
        messages: list[str] = []
        minimum_defect = self.expected_minimum_defect_diameter_px
        if minimum_defect is not None and self.smoothing_filter is SmoothingFilter.GAUSSIAN_BLUR:
            if self.resolved_gaussian_kernel_size >= minimum_defect:
                messages.append("Gaussian support approaches or exceeds the configured smallest-defect diameter.")
        if minimum_defect is not None and self.morphology_operation is MorphologyOperation.DISK_OPENING:
            if self.disk_diameter >= minimum_defect:
                messages.append("Disk opening diameter is not smaller than the configured smallest-defect diameter.")
        return tuple(messages)

    def to_dict(self) -> dict[str, object]:
        """Serialize the full reproducible profile rather than a preset label."""
        self.validate()
        operations: list[dict[str, object]] = []
        if self.color_mode is ColorMode.GRAYSCALE_REPLICATED_RGB:
            operations.append(
                {
                    "type": "grayscale",
                    "luminance_standard": LUMINANCE_STANDARD,
                    "output_channels": 3,
                    "channel_replication": True,
                }
            )
        if self.smoothing_filter is SmoothingFilter.BOX_BLUR:
            operations.append(
                {
                    "type": self.smoothing_filter.value,
                    "kernel_width": self.box_kernel_width,
                    "kernel_height": self.box_kernel_height,
                    "border_mode": self.smoothing_border_mode.value,
                }
            )
        elif self.smoothing_filter is SmoothingFilter.GAUSSIAN_BLUR:
            operations.append(
                {
                    "type": self.smoothing_filter.value,
                    "sigma": self.gaussian_sigma,
                    "kernel_size": self.gaussian_kernel_size if self.gaussian_kernel_size is not None else "automatic",
                    "border_mode": self.smoothing_border_mode.value,
                }
            )
        elif self.smoothing_filter is SmoothingFilter.MEDIAN_BLUR:
            operations.append(
                {
                    "type": self.smoothing_filter.value,
                    "kernel_size": self.median_kernel_size,
                    "border_mode": self.smoothing_border_mode.value,
                }
            )
        if self.morphology_operation is MorphologyOperation.DISK_OPENING:
            operation: dict[str, object] = {
                "type": self.morphology_operation.value,
                "radius": self.disk_radius,
                "diameter": self.disk_diameter,
                "iterations": self.disk_iterations,
                "border_mode": self.morphology_border_mode.value,
            }
            if self.pixels_per_millimetre is not None:
                operation["radius_mm"] = self.disk_radius / self.pixels_per_millimetre
                operation["diameter_mm"] = self.disk_diameter / self.pixels_per_millimetre
            operations.append(operation)
        guidance = {
            "expected_maximum_fiber_thickness_px": self.expected_maximum_fiber_thickness_px,
            "expected_minimum_defect_diameter_px": self.expected_minimum_defect_diameter_px,
            "pixels_per_millimetre": self.pixels_per_millimetre,
        }
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "input_color_order": INPUT_COLOR_ORDER,
            "input_dtype": INPUT_DTYPE,
            "input_range": INPUT_RANGE,
            "output_color_order": INPUT_COLOR_ORDER,
            "output_dtype": INPUT_DTYPE,
            "output_range": INPUT_RANGE,
            "operation_order": OPERATION_ORDER,
            "operations": operations,
            "guidance": guidance,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "ImagePreprocessingConfig":
        """Read a supported profile and reject reordered or unrecognized operations."""
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise ValueError("Image preprocessing profile must be a JSON object.")
        if payload.get("schema_version") != IMAGE_PREPROCESSING_SCHEMA_VERSION:
            raise ValueError(f"Unsupported image preprocessing schema version: {payload.get('schema_version')}")
        if (
            payload.get("input_color_order") != INPUT_COLOR_ORDER
            or payload.get("output_color_order") != INPUT_COLOR_ORDER
            or payload.get("input_dtype") != INPUT_DTYPE
            or payload.get("output_dtype") != INPUT_DTYPE
            or payload.get("input_range") != INPUT_RANGE
            or payload.get("output_range") != INPUT_RANGE
            or payload.get("operation_order") != OPERATION_ORDER
        ):
            raise ValueError("Unsupported image preprocessing RGB, numeric, or operation-order contract.")
        operations = payload.get("operations")
        if not isinstance(operations, list):
            raise ValueError("Image preprocessing operations must be an array.")
        color_mode = ColorMode.PRESERVE_RGB
        smoothing_filter = SmoothingFilter.NONE
        morphology_operation = MorphologyOperation.NONE
        box_kernel_width = box_kernel_height = 3
        gaussian_sigma = 1.0
        gaussian_kernel_size: int | None = None
        median_kernel_size = 3
        smoothing_border_mode = BorderMode.REFLECT
        disk_radius = 2
        disk_iterations = 1
        morphology_border_mode = BorderMode.REFLECT
        expected_order = ["grayscale", "box_blur", "gaussian_blur", "median_blur", "disk_morphological_opening"]
        last_order = -1
        for operation in operations:
            if not isinstance(operation, Mapping) or not isinstance(operation.get("type"), str):
                raise ValueError("Each image preprocessing operation must declare a type.")
            operation_type = str(operation["type"])
            try:
                order = expected_order.index(operation_type)
            except ValueError as exc:
                raise ValueError(f"Unsupported image preprocessing operation: {operation_type}") from exc
            if order <= last_order:
                raise ValueError("Image preprocessing operations must use deterministic grayscale, smoothing, disk order.")
            last_order = order
            if operation_type == "grayscale":
                if (
                    operation.get("luminance_standard") != LUMINANCE_STANDARD
                    or operation.get("output_channels") != 3
                    or operation.get("channel_replication") is not True
                ):
                    raise ValueError("Unsupported grayscale preprocessing contract.")
                color_mode = ColorMode.GRAYSCALE_REPLICATED_RGB
            elif operation_type == SmoothingFilter.BOX_BLUR.value:
                if smoothing_filter is not SmoothingFilter.NONE:
                    raise ValueError("Image preprocessing supports at most one smoothing operation.")
                smoothing_filter = SmoothingFilter.BOX_BLUR
                box_kernel_width = _positive_int(operation.get("kernel_width"), "Box blur kernel width")
                box_kernel_height = _positive_int(operation.get("kernel_height"), "Box blur kernel height")
                smoothing_border_mode = BorderMode(operation.get("border_mode"))
            elif operation_type == SmoothingFilter.GAUSSIAN_BLUR.value:
                if smoothing_filter is not SmoothingFilter.NONE:
                    raise ValueError("Image preprocessing supports at most one smoothing operation.")
                smoothing_filter = SmoothingFilter.GAUSSIAN_BLUR
                gaussian_sigma = _positive_float(operation.get("sigma"), "Gaussian blur sigma")
                raw_kernel_size = operation.get("kernel_size")
                gaussian_kernel_size = None if raw_kernel_size == "automatic" else _positive_int(
                    raw_kernel_size, "Gaussian blur kernel size"
                )
                smoothing_border_mode = BorderMode(operation.get("border_mode"))
            elif operation_type == SmoothingFilter.MEDIAN_BLUR.value:
                if smoothing_filter is not SmoothingFilter.NONE:
                    raise ValueError("Image preprocessing supports at most one smoothing operation.")
                smoothing_filter = SmoothingFilter.MEDIAN_BLUR
                median_kernel_size = _positive_int(operation.get("kernel_size"), "Median blur kernel size")
                smoothing_border_mode = BorderMode(operation.get("border_mode"))
            else:
                morphology_operation = MorphologyOperation.DISK_OPENING
                disk_radius = _positive_int(operation.get("radius"), "Disk opening radius")
                if operation.get("diameter") != 2 * disk_radius + 1:
                    raise ValueError("Disk opening diameter must equal 2 * radius + 1.")
                disk_iterations = _positive_int(operation.get("iterations"), "Disk opening iterations")
                morphology_border_mode = BorderMode(operation.get("border_mode"))
        guidance = payload.get("guidance", {})
        if not isinstance(guidance, Mapping):
            raise ValueError("Image preprocessing guidance must be an object.")
        result = cls(
            schema_version=int(payload["schema_version"]),
            profile_id=str(payload.get("profile_id", CUSTOM_PROFILE_ID)),
            color_mode=color_mode,
            smoothing_filter=smoothing_filter,
            box_kernel_width=box_kernel_width,
            box_kernel_height=box_kernel_height,
            gaussian_sigma=gaussian_sigma,
            gaussian_kernel_size=gaussian_kernel_size,
            median_kernel_size=median_kernel_size,
            smoothing_border_mode=smoothing_border_mode,
            morphology_operation=morphology_operation,
            disk_radius=disk_radius,
            disk_iterations=disk_iterations,
            morphology_border_mode=morphology_border_mode,
            expected_maximum_fiber_thickness_px=_optional_positive_float(
                guidance.get("expected_maximum_fiber_thickness_px"), "Expected maximum fiber thickness"
            ),
            expected_minimum_defect_diameter_px=_optional_positive_float(
                guidance.get("expected_minimum_defect_diameter_px"), "Expected minimum defect diameter"
            ),
            pixels_per_millimetre=_optional_positive_float(guidance.get("pixels_per_millimetre"), "Pixels per millimetre"),
        )
        result.validate()
        return result

    @classmethod
    def from_preset(cls, preset: PreprocessingPreset) -> "ImagePreprocessingConfig":
        """Return explicit deterministic operations for a UI convenience preset."""
        if preset is PreprocessingPreset.NONE:
            return cls()
        values: dict[str, object] = {
            "profile_id": CUSTOM_PROFILE_ID,
            "color_mode": ColorMode.GRAYSCALE_REPLICATED_RGB,
        }
        if preset in {PreprocessingPreset.GRAYSCALE_GAUSSIAN, PreprocessingPreset.GRAYSCALE_GAUSSIAN_DISK_OPENING}:
            values["smoothing_filter"] = SmoothingFilter.GAUSSIAN_BLUR
        elif preset is PreprocessingPreset.GRAYSCALE_MEDIAN:
            values["smoothing_filter"] = SmoothingFilter.MEDIAN_BLUR
        if preset in {PreprocessingPreset.GRAYSCALE_DISK_OPENING, PreprocessingPreset.GRAYSCALE_GAUSSIAN_DISK_OPENING}:
            values["morphology_operation"] = MorphologyOperation.DISK_OPENING
        return cls(**values)


def _positive_int(value: object, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if result <= 0 or result % 2 == 0 and "kernel" in name.casefold():
        raise ValueError(f"{name} must be positive and odd.")
    return result


def _positive_float(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite.") from exc
    if not isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite.")
    return result


def _optional_positive_float(value: object, name: str) -> float | None:
    return None if value is None else _positive_float(value, name)