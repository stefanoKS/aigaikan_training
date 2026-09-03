"""Pure deterministic RGB image operations used by preview, training, and inference."""

from __future__ import annotations

from typing import Final

import cv2
import numpy as np

from app.models.image_preprocessing import (
    BorderMode,
    ColorMode,
    ImagePreprocessingConfig,
    MorphologyOperation,
    SmoothingFilter,
)

_BORDER_TYPES: Final[dict[BorderMode, int]] = {
    BorderMode.REFLECT: cv2.BORDER_REFLECT_101,
    BorderMode.REPLICATE: cv2.BORDER_REPLICATE,
}


class ImagePreprocessor:
    """Apply one frozen profile to a rectified RGB ``uint8`` image.

    Input and output arrays use RGB channel order, dtype ``uint8``, and the
    closed numeric range $[0, 255]$. This class deliberately has no UI or file
    system concerns so every runtime path calls the same operations.
    """

    def __init__(self, config: ImagePreprocessingConfig) -> None:
        config.validate()
        self.config = config

    def apply(self, rectified_rgb: np.ndarray) -> np.ndarray:
        """Return a same-size three-channel RGB image after configured operations."""
        values = self._validated_rgb(rectified_rgb)
        result = values.copy()
        if self.config.color_mode is ColorMode.GRAYSCALE_REPLICATED_RGB:
            grayscale = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
            result = np.repeat(grayscale[..., np.newaxis], 3, axis=2)
        result = self._apply_smoothing(result)
        if self.config.morphology_operation is MorphologyOperation.DISK_OPENING:
            result = self._apply_disk_opening(result)
        if result.shape != values.shape or result.dtype != np.uint8:
            raise RuntimeError("Image preprocessing must preserve RGB dimensions and uint8 output.")
        if not np.isfinite(result).all():
            raise RuntimeError("Image preprocessing produced non-finite values.")
        return np.ascontiguousarray(result)

    def absolute_difference(self, original_rectified_rgb: np.ndarray, preprocessed_rgb: np.ndarray) -> np.ndarray:
        """Return fixed-range per-channel absolute difference for preview only."""
        original = self._validated_rgb(original_rectified_rgb)
        processed = self._validated_rgb(preprocessed_rgb)
        if original.shape != processed.shape:
            raise ValueError("Preprocessed image dimensions must match the rectified ROI.")
        return cv2.absdiff(original, processed)

    @staticmethod
    def disk_kernel(radius: int) -> np.ndarray:
        """Return the exact elliptical kernel used for disk morphological opening."""
        if radius <= 0:
            raise ValueError("Disk opening radius must be positive.")
        diameter = 2 * radius + 1
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))

    def _apply_smoothing(self, image_rgb: np.ndarray) -> np.ndarray:
        border_type = _BORDER_TYPES[self.config.smoothing_border_mode]
        if self.config.smoothing_filter is SmoothingFilter.NONE:
            return image_rgb
        if self.config.smoothing_filter is SmoothingFilter.BOX_BLUR:
            return cv2.blur(
                image_rgb,
                (self.config.box_kernel_width, self.config.box_kernel_height),
                borderType=border_type,
            )
        if self.config.smoothing_filter is SmoothingFilter.GAUSSIAN_BLUR:
            size = self.config.resolved_gaussian_kernel_size
            return cv2.GaussianBlur(
                image_rgb,
                (size, size),
                sigmaX=self.config.gaussian_sigma,
                sigmaY=self.config.gaussian_sigma,
                borderType=border_type,
            )
        radius = self.config.median_kernel_size // 2
        padded = cv2.copyMakeBorder(image_rgb, radius, radius, radius, radius, border_type)
        filtered = cv2.medianBlur(padded, self.config.median_kernel_size)
        return filtered[radius:-radius, radius:-radius]

    def _apply_disk_opening(self, image_rgb: np.ndarray) -> np.ndarray:
        if self.config.color_mode is not ColorMode.GRAYSCALE_REPLICATED_RGB:
            raise RuntimeError("Disk morphological opening requires grayscale replicated RGB input.")
        grayscale = image_rgb[:, :, 0]
        kernel = self.disk_kernel(self.config.disk_radius)
        border_type = _BORDER_TYPES[self.config.morphology_border_mode]
        eroded = cv2.erode(
            grayscale,
            kernel,
            iterations=self.config.disk_iterations,
            borderType=border_type,
        )
        opened = cv2.dilate(
            eroded,
            kernel,
            iterations=self.config.disk_iterations,
            borderType=border_type,
        )
        return np.repeat(opened[..., np.newaxis], 3, axis=2)

    @staticmethod
    def _validated_rgb(image_rgb: np.ndarray) -> np.ndarray:
        values = np.asarray(image_rgb)
        if values.ndim != 3 or values.shape[2] != 3 or values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError("Image preprocessing requires one non-empty RGB image.")
        if values.dtype != np.uint8:
            raise ValueError("Image preprocessing requires uint8 RGB values in the range [0, 255].")
        return np.ascontiguousarray(values)