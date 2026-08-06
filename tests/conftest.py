"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from app.core.project_manager import ProjectManager
from app.models.dataset_config import DatasetRole


@pytest.fixture()
def synthetic_dataset(tmp_path: Path) -> Path:
    """Create a small synthetic dataset."""
    root = tmp_path / "dataset with spaces" / "日本語"
    for relative in ("ok_train", "ok_test", "ng_test", "masks"):
        (root / relative).mkdir(parents=True, exist_ok=True)

    for index in range(2):
        image = Image.new("RGB", (64, 64), color=(255, 255 - index * 20, 0))
        image.save(root / "ok_train" / f"train_{index}.png")
    Image.new("RGB", (64, 64), color=(0, 255, 0)).save(root / "ok_test" / "ok_test.png")
    Image.new("RGB", (64, 64), color=(255, 0, 0)).save(root / "ng_test" / "ng_test.png")
    Image.new("L", (64, 64), color=255).save(root / "masks" / "ng_test.png")
    return root


@pytest.fixture()
def project_manager(tmp_path: Path) -> ProjectManager:
    """Return a project manager rooted at a temp path."""
    return ProjectManager(tmp_path / "projects")

