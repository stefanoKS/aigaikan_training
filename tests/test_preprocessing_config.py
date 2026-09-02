"""Tests for the immutable preprocessing-v2 geometry contract."""

from __future__ import annotations

import numpy as np
import pytest

from app.core.preprocessing_contract import (
    preprocessing_hash,
    read_preprocessing_config,
    read_resolved_preprocessing_plan,
    resolved_preprocessing_hash,
    write_preprocessing_config,
    write_resolved_preprocessing_plan,
)
from app.core.preprocessing_pipeline import PreprocessingPipeline, resolve_preprocessing_plan
from app.models.inspection_region import InspectionRegionConfig
from app.models.preprocessing_config import PreprocessingConfig, TilingConfig


def test_dinov3_full_roi_plan_preserves_the_reference_extent_on_a_valid_patch_grid() -> None:
    plan = PreprocessingConfig().resolve("dinomaly_dinov3", (639, 177))

    assert plan.patch_size == 16
    assert plan.model_input_size == (640, 192)
    assert plan.tiles[0].rectified_box == (0, 0, 639, 177)
    assert plan.tiles[0].valid_box == (0, 0, 639, 177)
    assert plan.tiles[0].model_input_size[0] % plan.patch_size == 0
    assert plan.tiles[0].model_input_size[1] % plan.patch_size == 0


def test_dinov2_full_roi_plan_uses_the_patch14_grid_without_a_destructive_crop() -> None:
    plan = PreprocessingConfig().resolve("dinomaly_dinov2", (639, 177))

    assert plan.patch_size == 14
    assert plan.model_input_size == (644, 182)
    assert plan.tiles[0].valid_box == (0, 0, 639, 177)


def test_tiled_dinov3_plan_uses_end_aligned_half_overlap_and_serializes_valid_masks() -> None:
    config = PreprocessingConfig(tiling=TilingConfig(enabled=True))
    plan = config.resolve("dinomaly_dinov3", (639, 177))
    payload = plan.to_dict()

    assert [tile.rectified_box for tile in plan.tiles] == [
        (0, 0, 320, 177),
        (160, 0, 320, 177),
        (319, 0, 320, 177),
    ]
    assert plan.model_input_size == (448, 256)
    assert [tile.valid_box for tile in plan.tiles] == [(0, 0, 448, 236)] * 3
    assert payload["tiles"][2]["valid_pixel_mask"] == {
        "encoding": "rectangular_valid_region",
        "size": [448, 256],
        "box": [0, 0, 448, 236],
    }


def test_preprocessing_config_and_resolved_plan_have_stable_content_hashes(tmp_path) -> None:
    config = PreprocessingConfig(tiling=TilingConfig(enabled=True))
    config_path = write_preprocessing_config(tmp_path / "preprocessing.json", config)
    restored_config = read_preprocessing_config(config_path)
    plan = config.resolve("dinomaly_dinov3", (639, 177))
    plan_path = write_resolved_preprocessing_plan(tmp_path / "resolved_preprocessing.json", plan)
    restored_plan = read_resolved_preprocessing_plan(plan_path)

    assert restored_config == config
    assert restored_plan == plan
    assert preprocessing_hash(restored_config) == preprocessing_hash(config)
    assert resolved_preprocessing_hash(restored_plan) == resolved_preprocessing_hash(plan)


def _reference_roi() -> InspectionRegionConfig:
    return InspectionRegionConfig(
        enabled=True,
        source_width=640,
        source_height=178,
        points_px=((0, 0), (639, 0), (639, 177), (0, 177)),
    )


def _marked_reference_image() -> np.ndarray:
    image = np.zeros((178, 640, 3), dtype=np.uint8)
    image[0, 0] = (255, 0, 0)
    image[0, 639] = (0, 255, 0)
    image[177, 639] = (0, 0, 255)
    image[177, 0] = (255, 255, 0)
    image[40, :, 0] = 80
    image[:, 200, 1] = 120
    image[100, :, 2] = 160
    return image


def test_full_roi_pipeline_preserves_endpoints_and_excludes_padding_from_scoring() -> None:
    config = PreprocessingConfig()
    pipeline = PreprocessingPipeline(_reference_roi(), config.resolve("dinomaly_dinov3", (639, 177)))
    prepared = pipeline.prepare_array(_marked_reference_image())[0]
    anomaly_map = np.zeros((192, 640), dtype=np.float32)
    anomaly_map[176, 638] = 0.7
    anomaly_map[191, 639] = 1.0

    assert prepared.image_rgb.shape == (192, 640, 3)
    assert tuple(prepared.image_rgb[0, 0]) == (255, 0, 0)
    assert tuple(prepared.image_rgb[0, 638]) == (0, 255, 0)
    assert tuple(prepared.image_rgb[176, 638]) == (0, 0, 255)
    assert tuple(prepared.image_rgb[176, 0]) == (255, 255, 0)
    assert prepared.valid_mask[:177, :639].all()
    assert not prepared.valid_mask[191, 639]
    assert 639 / 177 == prepared.valid_mask.sum(axis=0).astype(bool).sum() / prepared.valid_mask.sum(axis=1).astype(bool).sum()
    assert pipeline.score_from_anomaly_map(anomaly_map) == pytest.approx(0.7)


def test_tiled_pipeline_reconstructs_every_valid_reference_pixel() -> None:
    config = PreprocessingConfig(tiling=TilingConfig(enabled=True))
    pipeline = PreprocessingPipeline(_reference_roi(), config.resolve("dinomaly_dinov3", (639, 177)))
    prepared = pipeline.prepare_array(_marked_reference_image())
    reconstructed = pipeline.reconstruct_anomaly_maps(
        [np.full(item.valid_mask.shape, index + 1, dtype=np.float32) for index, item in enumerate(prepared)]
    )

    assert len(prepared) == 3
    assert all(item.image_rgb.shape == (256, 448, 3) for item in prepared)
    assert reconstructed.anomaly_map.shape == (177, 639)
    assert reconstructed.valid_mask.all()
    assert reconstructed.anomaly_map[20, 10] == 1
    assert reconstructed.anomaly_map[20, 200] == 2
    assert reconstructed.anomaly_map[20, 500] == 3


def test_mask_preprocessing_uses_the_same_tile_dimensions_without_interpolation() -> None:
    config = PreprocessingConfig(tiling=TilingConfig(enabled=True))
    pipeline = PreprocessingPipeline(_reference_roi(), config.resolve("dinomaly_dinov3", (639, 177)))
    mask = np.zeros((178, 640), dtype=np.uint8)
    mask[40, 200] = 255

    prepared_masks = pipeline.prepare_mask_array(mask)

    assert len(prepared_masks) == 3
    assert all(prepared_mask.shape == (256, 448) for prepared_mask in prepared_masks)
    assert all(set(np.unique(prepared_mask)).issubset({0, 255}) for prepared_mask in prepared_masks)
    assert sum(np.count_nonzero(prepared_mask) for prepared_mask in prepared_masks) > 0


def test_plan_resolution_uses_the_verified_source_size_without_an_enabled_roi(tmp_path) -> None:
    source_path = tmp_path / "source.png"
    from PIL import Image

    Image.new("RGB", (639, 177), (1, 2, 3)).save(source_path)

    plan = resolve_preprocessing_plan(
        PreprocessingConfig(),
        InspectionRegionConfig(),
        "dinomaly_dinov3",
        [source_path],
    )

    assert plan.rectified_size == (639, 177)
    assert plan.model_input_size == (640, 192)