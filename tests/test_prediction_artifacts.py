"""Tests for stable continuous-map and display artifact persistence."""

from __future__ import annotations

import numpy as np
from PIL import Image

from app.core.prediction_artifacts import DISPLAY_NORMALIZATION_VERSION, save_prediction_artifacts


def test_prediction_artifacts_preserve_postprocessed_values_with_stable_display_normalization(tmp_path) -> None:
    source_path = tmp_path / "source.png"
    Image.new("RGB", (8, 6), (20, 30, 40)).save(source_path)
    anomaly_map = np.array([[0.2, 0.3], [np.nan, 0.8]], dtype=np.float32)
    valid_mask = np.array([[True, True], [False, True]])

    artifacts = save_prediction_artifacts(source_path, anomaly_map, tmp_path / "artifacts", 3, valid_roi_mask=valid_mask)

    persisted_artifact = np.load(artifacts.continuous_anomaly_map)
    persisted = persisted_artifact["anomaly_map"]
    assert np.array_equal(persisted, anomaly_map, equal_nan=True)
    assert np.array_equal(persisted_artifact["valid_roi_mask"], valid_mask)
    assert artifacts.display_normalization == {
        "version": DISPLAY_NORMALIZATION_VERSION,
        "minimum": 0.0,
        "maximum": 1.0,
        "transform": "fixed_unit_interval",
        "coordinate_space": "continuous_anomaly_map",
        "invalid_pixels": "transparent",
    }
    assert Image.open(artifacts.heatmap_image).mode == "RGBA"
    assert Image.open(artifacts.heatmap_image).size == (8, 6)
    assert Image.open(artifacts.overlay_image).size == (8, 6)


def test_prediction_artifacts_use_one_color_scale_and_exclude_invalid_mask_pixels(tmp_path) -> None:
    source_path = tmp_path / "source.png"
    Image.new("RGB", (2, 2), (20, 30, 40)).save(source_path)
    first = save_prediction_artifacts(
        source_path,
        np.array([[0.5, 0.0], [1.0, 0.25]], dtype=np.float32),
        tmp_path / "first",
        0,
    )
    second = save_prediction_artifacts(
        source_path,
        np.array([[0.0, 0.5], [1.0, 0.75]], dtype=np.float32),
        tmp_path / "second",
        0,
        pixel_threshold=0.6,
        valid_roi_mask=np.array([[False, True], [True, True]]),
    )

    assert np.asarray(Image.open(first.heatmap_image))[0, 0].tolist() == np.asarray(Image.open(second.heatmap_image))[0, 1].tolist()
    assert np.asarray(Image.open(second.heatmap_image))[0, 0, 3] == 0
    assert np.asarray(Image.open(second.binary_mask))[0, 0] == 0
    assert np.asarray(Image.open(second.overlay_image))[0, 0].tolist() == [20, 30, 40]


def test_prediction_artifacts_persist_raw_map_without_changing_postprocessed_map(tmp_path) -> None:
    source_path = tmp_path / "source.png"
    Image.new("RGB", (2, 2), (20, 30, 40)).save(source_path)
    raw_map = np.array([[3.0, 5.0], [7.0, 9.0]], dtype=np.float32)
    postprocessed_map = raw_map / 10

    artifacts = save_prediction_artifacts(
        source_path,
        postprocessed_map,
        tmp_path / "artifacts",
        0,
        raw_anomaly_map=raw_map,
    )

    assert np.array_equal(np.load(artifacts.raw_anomaly_map)["anomaly_map"], raw_map)
    assert np.array_equal(np.load(artifacts.continuous_anomaly_map)["anomaly_map"], postprocessed_map)


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