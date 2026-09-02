"""Tests for project creation and persistence."""

from __future__ import annotations

from pathlib import Path

from app.core.project_manager import ProjectManager
from app.models.inspection_region import InspectionRegionConfig


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
