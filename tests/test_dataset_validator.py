"""Tests for dataset validation."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from app.core.dataset_validator import DatasetValidator
from app.models.dataset_config import DatasetConfig, DatasetRole


def _config_for(root: Path) -> DatasetConfig:
    config = DatasetConfig()
    for role in DatasetRole:
        config.folders[role].path = str(root / role.value)
    return config


def test_detects_empty_folder(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    for role in DatasetRole:
        (root / role.value).mkdir(parents=True, exist_ok=True)
    config = _config_for(root)
    report = DatasetValidator().validate(config)
    assert any(issue.message == "Folder is empty" for issue in report.errors)


def test_detects_unsupported_and_corrupt_files(synthetic_dataset: Path) -> None:
    (synthetic_dataset / "ok_train" / "note.txt").write_text("not an image", encoding="utf-8")
    (synthetic_dataset / "ok_test" / "broken.png").write_bytes(b"not a valid png")
    config = _config_for(synthetic_dataset)
    report = DatasetValidator().validate(config)
    messages = {issue.message for issue in report.errors}
    assert "Unsupported file type" in messages
    assert "Corrupt image" in messages


def test_valid_dataset_yields_stats(synthetic_dataset: Path) -> None:
    config = _config_for(synthetic_dataset)
    report = DatasetValidator().validate(config)
    assert report.stats["ok_train"]["image_count"] == 2
    assert report.stats["ng_test"]["typical_resolution"] == "64x64"

