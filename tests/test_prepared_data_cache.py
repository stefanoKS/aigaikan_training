"""Tests for immutable, verified model-ready tile cache entries."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from app.core.prepared_data_cache import PreparedDataCache
from app.core.preprocessing_pipeline import PreprocessingPipeline
from app.models.inspection_region import InspectionRegionConfig
from app.models.preprocessing_config import PreprocessingConfig, TilingConfig


def test_prepared_data_cache_reuses_verified_tiles_and_rebuilds_corruption(tmp_path: Path, monkeypatch) -> None:
    source_path = tmp_path / "source.png"
    Image.new("RGB", (639, 177), (20, 30, 40)).save(source_path)
    pipeline = PreprocessingPipeline(
        InspectionRegionConfig(),
        PreprocessingConfig(tiling=TilingConfig(enabled=True)).resolve("dinomaly_dinov3", (639, 177)),
    )
    cache = PreparedDataCache(tmp_path / "prepared_data_cache", pipeline)
    original_prepare_path = pipeline.prepare_path
    calls = 0

    def track_prepare_path(path: Path):
        nonlocal calls
        calls += 1
        return original_prepare_path(path)

    monkeypatch.setattr(pipeline, "prepare_path", track_prepare_path)
    first = cache.materialize(source_path)
    second = cache.materialize(source_path)

    assert calls == 1
    assert first == second
    assert cache.report().to_dict() == {"version": 1, "hits": 1, "misses": 1, "rebuilt_entries": 0}

    first[0].write_bytes(b"corrupt")
    rebuilt = cache.materialize(source_path)

    assert calls == 2
    assert Image.open(rebuilt[0]).size == pipeline.plan.model_input_size
    assert cache.report().to_dict() == {"version": 1, "hits": 1, "misses": 2, "rebuilt_entries": 1}
    assert cache.clear() == 1