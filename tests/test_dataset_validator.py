"""Tests for dataset validation."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from app.core.dataset_validator import DatasetValidator
from app.models.dataset_config import DatasetConfig, DatasetRole
from app.models.inspection_region import InspectionRegionConfig


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


def test_missing_masks_folder_is_optional(synthetic_dataset: Path) -> None:
    config = _config_for(synthetic_dataset)
    config.folders[DatasetRole.MASKS].path = str(synthetic_dataset / "missing_masks")

    report = DatasetValidator().validate(config)

    assert not any(
        issue.role == DatasetRole.MASKS.value and issue.message == "Folder does not exist"
        for issue in report.errors
    )


def test_empty_selected_mask_folder_only_warns(synthetic_dataset: Path) -> None:
    """An empty optional mask folder must not block image-level training."""
    (synthetic_dataset / "masks" / "ng_test.png").unlink()
    config = _config_for(synthetic_dataset)

    report = DatasetValidator().validate(config)

    assert not any(issue.role == DatasetRole.MASKS.value for issue in report.errors)
    assert any(issue.message == "Mask folder is empty; pixel metrics unavailable" for issue in report.warnings)


def test_mask_suffix_and_dimensions_follow_anomalib_folder_contract(synthetic_dataset: Path) -> None:
    """Anomalib accepts an image_mask suffix, while dimensions still need to match."""
    masks_folder = synthetic_dataset / "masks"
    (masks_folder / "ng_test.png").rename(masks_folder / "ng_test_mask.png")
    config = _config_for(synthetic_dataset)

    report = DatasetValidator().validate(config)

    assert not any(issue.message == "Missing optional mask for NG image" for issue in report.warnings)
    assert not any(issue.message == "Mask dimensions do not match the NG image" for issue in report.warnings)


def test_normal_only_dataset_is_valid_with_an_unverified_defect_warning(tmp_path: Path) -> None:
    folder = tmp_path / "ok_train"
    folder.mkdir()
    Image.new("RGB", (64, 64), (20, 30, 40)).save(folder / "one.png")
    Image.new("RGB", (64, 64), (30, 40, 50)).save(folder / "two.png")
    config = DatasetConfig()
    config.folders[DatasetRole.OK_TRAIN].path = str(folder)

    report = DatasetValidator().validate(config)

    assert report.is_valid
    assert any("No genuine NG test data" in issue.message for issue in report.warnings)


def test_validator_recursively_checks_the_same_files_used_by_manifests(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    for role in DatasetRole:
        (root / role.value).mkdir(parents=True)
    for index in range(2):
        Image.new("RGB", (64, 64), (20 + index, 30, 40)).save(root / "ok_train" / f"top_{index}.png")
    nested = root / "ok_train" / "nested"
    nested.mkdir()
    Image.new("RGB", (64, 64), (80, 90, 100)).save(nested / "valid.png")
    (nested / "unsupported.txt").write_text("not an image", encoding="utf-8")
    (nested / "corrupt.png").write_bytes(b"not a valid png")

    config = _config_for(root)
    report = DatasetValidator().validate(config)

    assert report.stats["ok_train"]["image_count"] == 3
    assert {(issue.message, Path(issue.path).name) for issue in report.errors} >= {
        ("Unsupported file type", "unsupported.txt"),
        ("Corrupt image", "corrupt.png"),
    }


def test_validator_marks_explicit_independent_evaluation_as_production_quality(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    for role in DatasetRole:
        (root / role.value).mkdir(parents=True)
    for folder_name in ("ok_train", "ok_validation", "ok_test", "ng_validation", "ng_test"):
        Image.new("RGB", (64, 64), (40, 50, 60)).save(root / folder_name / f"{folder_name}.png")
    Image.new("RGB", (64, 64), (70, 80, 90)).save(root / "ok_train" / "second.png")

    report = DatasetValidator().validate(_config_for(root))

    assert report.stats["evaluation"]["method"] == "independent_explicit"
    assert not any(issue.role == "evaluation" for issue in report.warnings)


def test_enabled_inspection_region_requires_matching_source_resolution(synthetic_dataset: Path) -> None:
    config = _config_for(synthetic_dataset)
    inspection_region = InspectionRegionConfig(
        enabled=True,
        source_width=80,
        source_height=64,
        points_px=((4, 4), (75, 4), (75, 59), (4, 59)),
    )

    report = DatasetValidator().validate(config, inspection_region)

    assert not report.is_valid
    assert any(issue.role == "inspection_region" and "does not match" in issue.message for issue in report.errors)

