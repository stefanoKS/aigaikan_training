"""Tests for stable continuous-map and display artifact persistence."""

from __future__ import annotations

import numpy as np
from PIL import Image

from app.core.prediction_artifacts import DISPLAY_NORMALIZATION_VERSION, save_prediction_artifacts


def test_prediction_artifacts_preserve_raw_values_and_freeze_display_normalization(tmp_path) -> None:
    source_path = tmp_path / "source.png"
    Image.new("RGB", (8, 6), (20, 30, 40)).save(source_path)
    anomaly_map = np.array([[2.0, 3.0], [np.nan, 8.0]], dtype=np.float32)

    artifacts = save_prediction_artifacts(source_path, anomaly_map, tmp_path / "artifacts", 3)

    persisted = np.load(artifacts.continuous_anomaly_map)["anomaly_map"]
    assert np.array_equal(persisted, anomaly_map, equal_nan=True)
    assert artifacts.display_normalization == {
        "version": DISPLAY_NORMALIZATION_VERSION,
        "minimum": 2.0,
        "maximum": 8.0,
        "coordinate_space": "continuous_anomaly_map",
    }
    assert Image.open(artifacts.heatmap_image).size == (8, 6)
    assert Image.open(artifacts.overlay_image).size == (8, 6)


def test_prediction_artifacts_write_binary_and_contour_overlays_only_for_explicit_pixel_threshold(tmp_path) -> None:
    source_path = tmp_path / "source.png"
    Image.new("RGB", (6, 6), (20, 30, 40)).save(source_path)
    anomaly_map = np.array([[0.2, 0.8], [0.4, 0.6]], dtype=np.float32)

    no_mask = save_prediction_artifacts(source_path, anomaly_map, tmp_path / "without-mask", 0)
    artifacts = save_prediction_artifacts(
        source_path,
        anomaly_map,
        tmp_path / "with-mask",
        0,
        pixel_threshold=0.6,
    )

    assert not no_mask.binary_mask
    assert not no_mask.contour_overlay_image
    assert np.array_equal(
        np.asarray(Image.open(artifacts.binary_mask)),
        np.array([[0, 255], [0, 255]], dtype=np.uint8),
    )
    assert Image.open(artifacts.contour_overlay_image).size == (6, 6)