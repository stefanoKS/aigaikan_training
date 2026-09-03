"""Tests for project creation and persistence."""

from __future__ import annotations

from pathlib import Path

from app.core.project_manager import ProjectManager
from app.models.inspection_region import InspectionRegionConfig
from app.models.preprocessing_preview import PreprocessingPreviewState, PreviewSource
from app.models.image_preprocessing import ColorMode, ImagePreprocessingConfig
from app.models.preprocessing_config import PreprocessingConfig


def test_create_and_reopen_project_with_unicode_path(project_manager: ProjectManager) -> None:
    project = project_manager.create_project("Project 日本語")
    reopened = project_manager.load_project(Path(project.project_path))
    assert reopened.name == "Project 日本語"
    assert Path(reopened.project_path).name == "Project 日本語"
    assert (Path(reopened.project_path) / "dataset" / "ok_train").exists()
    assert (Path(reopened.project_path) / "inspection_region.json").is_file()


def test_project_persists_its_canonical_inspection_region_sidecar(project_manager: ProjectManager) -> None:
    project = project_manager.create_project("Inspection Region")
    project.inspection_region = InspectionRegionConfig(
        enabled=True,
        source_width=64,
        source_height=64,
        points_px=((4, 4), (59, 4), (59, 59), (4, 59)),
    )
    project_manager.save_project(project)

    restored = project_manager.load_project(project.root_path)

    assert restored.inspection_region == project.inspection_region


def test_project_persists_preview_source_as_draft_state_without_touching_preprocessing_sidecar(
    project_manager: ProjectManager,
) -> None:
    project = project_manager.create_project("Preprocessing Preview")
    project.preprocessing_preview = PreprocessingPreviewState(
        source=PreviewSource.CUSTOM_IMAGE,
        custom_image_path=r"C:\camera\preview.png",
        already_rectified=True,
    )
    project_manager.save_project(project)

    restored = project_manager.load_project(project.root_path)

    assert restored.preprocessing_preview == project.preprocessing_preview
    assert "custom_image_path" not in (project.root_path / "preprocessing.json").read_text(encoding="utf-8")


def test_project_persists_image_preprocessing_profile_across_restart(project_manager: ProjectManager) -> None:
    project = project_manager.create_project("Image Profile")
    project.preprocessing = PreprocessingConfig(
        image_preprocessing=ImagePreprocessingConfig(
            profile_id="gray-v1",
            color_mode=ColorMode.GRAYSCALE_REPLICATED_RGB,
        )
    )
    project_manager.save_project(project)

    restored = project_manager.load_project(project.root_path)

    assert restored.preprocessing == project.preprocessing


def test_create_unique_run_directory(project_manager: ProjectManager) -> None:
    project = project_manager.create_project("RunDir Project")
    first = project_manager.create_run_directory(project, "patchcore")
    second = project_manager.create_run_directory(project, "patchcore")
    assert first != second
    assert "patchcore" in first.name
    assert second.exists()


def test_import_reuses_a_source_already_inside_project_data(project_manager: ProjectManager) -> None:
    """Reselecting project-owned data after clearing must not copy its files again."""
    project = project_manager.create_project("Existing Dataset")
    source = project.root_path / "dataset" / "ng_test"
    destination = project.root_path / "dataset" / "masks"
    image = source / "000_overexposed.png"
    image.write_bytes(b"image")

    assigned_path = project_manager.import_dataset_folder(source, destination, copy_files=True)

    assert assigned_path == source
    assert image.exists()
