"""Tests for reproducible source-image manifests and effective splits."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from app.core.dataset_manifest import (
    build_dataset_manifest,
    build_effective_split,
    stage_effective_split,
    validate_effective_split,
)
from app.models.dataset_config import DatasetConfig, DatasetRole


def _save_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (640, 480), color).save(path)


def _dataset_config(root: Path) -> DatasetConfig:
    config = DatasetConfig()
    for role in DatasetRole:
        config.folders[role].path = str(root / role.value)
    return config


def test_effective_split_is_deterministic_and_disjoint(tmp_path: Path) -> None:
    config = _dataset_config(tmp_path)
    for index in range(10):
        _save_image(tmp_path / "ok_train" / f"ok_{index}.png", (index + 10, 0, 0))
    for index in range(6):
        _save_image(tmp_path / "ng_test" / f"ng_{index}.png", (0, index + 30, 0))

    first = build_effective_split(config, seed=42)
    second = build_effective_split(config, seed=42)

    assert first == second
    assert first.counts() == {
        "training": {"ok": 8, "ng": 0},
        "validation": {"ok": 1, "ng": 3},
        "final_test": {"ok": 1, "ng": 3},
    }


def test_effective_split_rejects_cross_split_content_duplicates(tmp_path: Path) -> None:
    original = tmp_path / "ok_train" / "shared.png"
    _save_image(original, (1, 2, 3))
    copied = tmp_path / "ok_test" / "duplicate.png"
    copied.parent.mkdir(parents=True)
    copied.write_bytes(original.read_bytes())
    _save_image(tmp_path / "ok_test" / "unique.png", (4, 5, 6))
    for index in range(2):
        _save_image(tmp_path / "ok_train" / f"ok_{index}.png", (index + 10, 0, 0))
        _save_image(tmp_path / "ng_test" / f"ng_{index}.png", (0, index + 20, 0))

    with pytest.raises(ValueError, match="Dataset leakage"):
        build_effective_split(_dataset_config(tmp_path), seed=42)


def test_manifest_records_source_metadata_and_stable_hash(tmp_path: Path) -> None:
    image_path = tmp_path / "dataset" / "ok.png"
    _save_image(image_path, (3, 4, 5))

    manifest = build_dataset_manifest({"training_ok": [image_path]}, project_root=tmp_path)

    record = manifest["records"][0]
    assert record["path"] == "dataset\\ok.png"
    assert record["dataset_role"] == "training_ok"
    assert record["width"] == 640
    assert record["height"] == 480
    assert len(manifest["manifest_sha256"]) == 64


def test_staged_split_copies_source_images_without_mutating_them(tmp_path: Path) -> None:
    config = _dataset_config(tmp_path)
    for index in range(10):
        _save_image(tmp_path / "ok_train" / f"ok_{index}.png", (index + 10, 0, 0))
    for index in range(4):
        _save_image(tmp_path / "ng_test" / f"ng_{index}.png", (0, index + 30, 0))
    split = build_effective_split(config, seed=42)
    source_bytes = {path: path.read_bytes() for paths in split.roles().values() for path in paths}

    staged = stage_effective_split(split, config, tmp_path / "run" / "dataset_snapshot")

    assert staged.training_config.folders[DatasetRole.OK_TRAIN].resolved_path().is_dir()
    assert staged.final_test_config.folders[DatasetRole.NG_TEST].resolved_path().is_dir()
    assert all(staged_path.read_bytes() == source_bytes[source_path] for staged_path, source_path in staged.source_path_by_staged_path.items())
    assert all(path.read_bytes() == source_bytes[path] for path in source_bytes)


def test_normal_only_split_has_held_out_calibration_and_no_ng_snapshot(tmp_path: Path) -> None:
    config = _dataset_config(tmp_path)
    for index in range(10):
        _save_image(tmp_path / "ok_train" / f"ok_{index}.png", (index + 10, 0, 0))

    split = build_effective_split(config, seed=42)
    staged = stage_effective_split(split, config, tmp_path / "run" / "dataset_snapshot")

    assert split.counts() == {
        "training": {"ok": 8, "ng": 0},
        "validation": {"ok": 1, "ng": 0},
        "final_test": {"ok": 1, "ng": 0},
    }
    assert staged.training_config.folders[DatasetRole.NG_TEST].resolved_path() is None
    assert staged.final_test_config.folders[DatasetRole.NG_TEST].resolved_path() is None