"""Tests for project creation and persistence."""

from __future__ import annotations

from pathlib import Path

from app.core.project_manager import ProjectManager


def test_create_and_reopen_project_with_unicode_path(project_manager: ProjectManager) -> None:
    project = project_manager.create_project("Project 日本語")
    reopened = project_manager.load_project(Path(project.project_path))
    assert reopened.name == "Project 日本語"
    assert Path(reopened.project_path).name == "Project 日本語"
    assert (Path(reopened.project_path) / "dataset" / "ok_train").exists()


def test_create_unique_run_directory(project_manager: ProjectManager) -> None:
    project = project_manager.create_project("RunDir Project")
    first = project_manager.create_run_directory(project, "patchcore")
    second = project_manager.create_run_directory(project, "patchcore")
    assert first != second
    assert "patchcore" in first.name
    assert second.exists()
