"""Persist reproducible anomaly-map artifacts without changing model values."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.core.inspection_region import inspection_region_hash
from app.core.threshold_contract import PIXEL_THRESHOLD_COMPARATOR, PIXEL_THRESHOLD_SEMANTIC
from app.models.inspection_region import InspectionRegionConfig

DISPLAY_NORMALIZATION_VERSION = "fixed_unit_interval_v1"


@dataclass(frozen=True, slots=True)
class PredictionArtifacts:
    """Paths and rendering metadata produced from one continuous anomaly map."""

    continuous_anomaly_map: str = ""
    raw_anomaly_map: str = ""
    heatmap_image: str = ""
    overlay_image: str = ""
    binary_mask: str = ""
    contour_overlay_image: str = ""
    display_normalization: dict[str, object] | None = None
    pixel_threshold: float | None = None
    pixel_threshold_comparator: str = ""
    pixel_threshold_semantic: str = ""


def inspection_region_metadata(config: InspectionRegionConfig) -> dict[str, object]:
    """Describe the coordinate system shared by persisted map artifacts."""
    rectified_width, rectified_height = config.rectified_size()
    return {
        "coordinate_space": "rectified_roi" if config.enabled else "source_image",
        "roi_contract_version": config.roi_contract_version,
        "roi_hash": inspection_region_hash(config),
        "enabled": config.enabled,
        "type": config.region_type,
        "source_size": [config.source_width, config.source_height],
        "rectified_size": [rectified_width, rectified_height],
    }


def save_prediction_artifacts(
    source_path: Path,
    anomaly_map: Any,
    output_directory: Path,
    index: int,
    *,
    rectified_image: np.ndarray | None = None,
    pixel_threshold: float | None = None,
    valid_roi_mask: Any = None,
    raw_anomaly_map: Any = None,
) -> PredictionArtifacts:
    """Save postprocessed map data and stable display artifacts without altering model values."""
    if anomaly_map is None:
        return PredictionArtifacts()
    values = _as_map_array(anomaly_map)
    valid_mask = _valid_roi_mask(valid_roi_mask, values.shape)
    output_directory.mkdir(parents=True, exist_ok=True)
    continuous_path = (output_directory / f"{index:04d}_continuous_anomaly_map.npz").resolve()
    np.savez_compressed(continuous_path, anomaly_map=values, valid_roi_mask=valid_mask)
    raw_map_path = ""
    if raw_anomaly_map is not None:
        raw_values = _as_map_array(raw_anomaly_map)
        if raw_values.shape != values.shape:
            raise ValueError("Raw and postprocessed anomaly maps must share dimensions.")
        raw_path = (output_directory / f"{index:04d}_raw_anomaly_map.npz").resolve()
        np.savez_compressed(raw_path, anomaly_map=raw_values, valid_roi_mask=valid_mask)
        raw_map_path = str(raw_path)

    finite_values = values[np.logical_and(valid_mask, np.isfinite(values))]
    if finite_values.size == 0:
        return PredictionArtifacts(
            continuous_anomaly_map=str(continuous_path),
            display_normalization={
                "version": DISPLAY_NORMALIZATION_VERSION,
                "status": "no_finite_valid_values",
                "coordinate_space": "continuous_anomaly_map",
            },
        )
    normalization: dict[str, object] = {
        "version": DISPLAY_NORMALIZATION_VERSION,
        "minimum": 0.0,
        "maximum": 1.0,
        "transform": "fixed_unit_interval",
        "coordinate_space": "continuous_anomaly_map",
        "invalid_pixels": "transparent",
    }
    normalized = _normalize_for_display(values, valid_mask)
    heatmap = _heatmap(normalized, valid_mask)
    original = _source_image(source_path, rectified_image)
    heatmap_image = Image.fromarray(heatmap, "RGBA").resize(original.size, Image.Resampling.BILINEAR)
    heatmap_path = (output_directory / f"{index:04d}_heatmap.png").resolve()
    overlay_path = (output_directory / f"{index:04d}_overlay.png").resolve()
    heatmap_image.save(heatmap_path)
    overlay_heatmap = heatmap_image.copy()
    overlay_alpha = np.asarray(heatmap_image.getchannel("A"), dtype=np.uint16) * 115 // 255
    overlay_heatmap.putalpha(Image.fromarray(overlay_alpha.astype(np.uint8), "L"))
    overlay = Image.alpha_composite(original.convert("RGBA"), overlay_heatmap).convert("RGB")
    overlay.save(overlay_path)

    binary_mask_path = ""
    contour_overlay_path = ""
    if pixel_threshold is not None:
        if not isfinite(pixel_threshold):
            raise ValueError("Pixel threshold must be finite when creating a binary anomaly mask.")
        binary_mask = np.logical_and(valid_mask, np.logical_and(np.isfinite(values), values >= pixel_threshold)).astype(np.uint8) * 255
        mask_path = (output_directory / f"{index:04d}_binary_mask.png").resolve()
        Image.fromarray(binary_mask, "L").save(mask_path)
        contour_path = (output_directory / f"{index:04d}_contour_overlay.png").resolve()
        _contour_overlay(original, binary_mask).save(contour_path)
        binary_mask_path = str(mask_path)
        contour_overlay_path = str(contour_path)

    return PredictionArtifacts(
        continuous_anomaly_map=str(continuous_path),
        raw_anomaly_map=raw_map_path,
        heatmap_image=str(heatmap_path),
        overlay_image=str(overlay_path),
        binary_mask=binary_mask_path,
        contour_overlay_image=contour_overlay_path,
        display_normalization=normalization,
        pixel_threshold=pixel_threshold,
        pixel_threshold_comparator=PIXEL_THRESHOLD_COMPARATOR if pixel_threshold is not None else "",
        pixel_threshold_semantic=PIXEL_THRESHOLD_SEMANTIC if pixel_threshold is not None else "",
    )


def save_mask_artifacts(
    source_path: Path,
    anomaly_map: Any,
    output_directory: Path,
    index: int,
    *,
    rectified_image: np.ndarray | None = None,
    pixel_threshold: float | None = None,
    valid_roi_mask: Any = None,
) -> PredictionArtifacts:
    """Write only a binary mask and contour when a revision changes pixel policy.

    Continuous maps, heatmaps, overlays, and raw maps remain immutable when an
    operator changes only a decision or pixel threshold.
    """
    if anomaly_map is None or pixel_threshold is None:
        return PredictionArtifacts()
    if not isfinite(pixel_threshold):
        raise ValueError("Pixel threshold must be finite when creating a binary anomaly mask.")
    values = _as_map_array(anomaly_map)
    valid_mask = _valid_roi_mask(valid_roi_mask, values.shape)
    output_directory.mkdir(parents=True, exist_ok=True)
    binary_mask = np.logical_and(valid_mask, np.logical_and(np.isfinite(values), values >= pixel_threshold)).astype(np.uint8) * 255
    mask_path = (output_directory / f"{index:04d}_binary_mask.png").resolve()
    Image.fromarray(binary_mask, "L").save(mask_path)
    original = _source_image(source_path, rectified_image)
    contour_path = (output_directory / f"{index:04d}_contour_overlay.png").resolve()
    _contour_overlay(original, binary_mask).save(contour_path)
    return PredictionArtifacts(
        binary_mask=str(mask_path),
        contour_overlay_image=str(contour_path),
        pixel_threshold=pixel_threshold,
        pixel_threshold_comparator=PIXEL_THRESHOLD_COMPARATOR,
        pixel_threshold_semantic=PIXEL_THRESHOLD_SEMANTIC,
    )


def _as_map_array(anomaly_map: Any) -> np.ndarray:
    values = anomaly_map.detach().cpu().numpy() if hasattr(anomaly_map, "detach") else np.asarray(anomaly_map)
    while values.ndim > 2:
        values = values[0]
    if values.ndim != 2 or values.size == 0:
        raise ValueError("Anomaly map must contain one non-empty two-dimensional image.")
    return np.ascontiguousarray(values)


def _valid_roi_mask(valid_roi_mask: Any, expected_shape: tuple[int, int]) -> np.ndarray:
    if valid_roi_mask is None:
        return np.ones(expected_shape, dtype=bool)
    mask = valid_roi_mask.detach().cpu().numpy() if hasattr(valid_roi_mask, "detach") else np.asarray(valid_roi_mask)
    if mask.shape != expected_shape:
        raise ValueError("Valid ROI mask and anomaly map must share dimensions.")
    if not mask.astype(bool, copy=False).any():
        raise ValueError("Valid ROI mask must contain at least one pixel.")
    return np.ascontiguousarray(mask.astype(bool, copy=False))


def _normalize_for_display(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Clamp only a display copy so each rendered color has one stable meaning across a run."""
    normalized = np.clip(values.astype(np.float32), 0.0, 1.0)
    return np.where(np.logical_and(valid_mask, np.isfinite(normalized)), normalized, 0.0)


def render_fixed_unit_interval_heatmap(anomaly_map: Any, valid_roi_mask: Any = None) -> np.ndarray:
    """Return the standard RGBA heatmap array without changing continuous map values."""
    values = _as_map_array(anomaly_map)
    valid_mask = _valid_roi_mask(valid_roi_mask, values.shape)
    return _heatmap(_normalize_for_display(values, valid_mask), valid_mask)


def _heatmap(normalized: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Render a deterministic blue-yellow-red heatmap from normalized display values."""
    rgb = np.stack(
        (
            normalized,
            1.0 - np.abs(normalized - 0.5) * 2.0,
            1.0 - normalized,
        ),
        axis=-1,
    )
    alpha = valid_mask.astype(np.uint8) * 255
    return np.dstack((np.clip(rgb * 255, 0, 255).astype(np.uint8), alpha))


def _source_image(source_path: Path, rectified_image: np.ndarray | None) -> Image.Image:
    if rectified_image is not None:
        return Image.fromarray(rectified_image, "RGB")
    with Image.open(source_path) as source:
        return source.convert("RGB")


def _contour_overlay(original: Image.Image, binary_mask: np.ndarray) -> Image.Image:
    canvas = np.asarray(original).copy()
    contours, _hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        scale_x = canvas.shape[1] / binary_mask.shape[1]
        scale_y = canvas.shape[0] / binary_mask.shape[0]
        scaled_contours = [
            np.round(contour.astype(np.float32) * (scale_x, scale_y)).astype(np.int32)
            for contour in contours
        ]
        cv2.drawContours(canvas, scaled_contours, -1, (255, 64, 32), thickness=2)
    return Image.fromarray(canvas, "RGB")