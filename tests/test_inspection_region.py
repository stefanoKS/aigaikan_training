"""Inspection-region geometry and metadata contract tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.core.inspection_region import (
    InspectionRegionProcessor,
    inspection_region_hash,
    read_inspection_region,
    write_inspection_region,
)
from app.models.inspection_region import InspectionRegionConfig, order_quad_points


def _config() -> InspectionRegionConfig:
    return InspectionRegionConfig(
        enabled=True,
        source_width=120,
        source_height=100,
        points_px=((20, 20), (90, 10), (100, 80), (15, 85)),
    )


def test_order_quad_points_canonicalizes_a_click_order() -> None:
    assert order_quad_points(((100, 80), (20, 20), (15, 85), (90, 10))) == ((20, 20), (90, 10), (100, 80), (15, 85))


def test_perspective_rectification_preserves_natural_geometry_deterministically() -> None:
    config = _config()
    processor = InspectionRegionProcessor(config)
    source = np.zeros((100, 120, 3), dtype=np.uint8)
    source[10:86, 15:101] = (12, 200, 40)

    first = processor.apply(source)
    second = processor.apply(source)

    assert config.rectified_size() == (85, 71)
    assert first.shape == (71, 85, 3)
    assert np.array_equal(first, second)
    assert tuple(first[20, 20]) == (12, 200, 40)


def test_metadata_serializes_normalized_pixels_and_hashes_canonically(tmp_path: Path) -> None:
    config = _config()
    path = write_inspection_region(tmp_path / "inspection_region.json", config)

    restored = read_inspection_region(path)
    payload = restored.to_dict()

    assert payload["points_normalized"][0] == [20 / 120, 20 / 100]
    assert payload["rectified_size"] == {"width": 85, "height": 71}
    assert inspection_region_hash(restored) == inspection_region_hash(config)
    assert len(inspection_region_hash(config)) == 64


def test_invalid_self_intersecting_or_resolution_mismatched_roi_is_rejected() -> None:
    crossing = InspectionRegionConfig(
        enabled=True,
        source_width=100,
        source_height=100,
        points_px=((10, 10), (90, 90), (90, 10), (10, 90)),
    )
    with pytest.raises(ValueError, match="self-intersect"):
        InspectionRegionProcessor(crossing)

    processor = InspectionRegionProcessor(_config())
    with pytest.raises(ValueError, match="resolution does not match"):
        processor.apply(np.zeros((101, 120, 3), dtype=np.uint8))