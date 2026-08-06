"""Project persistence and filesystem management."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from app.models.project_config import ProjectConfig

LOGGER = logging.getLogger(__name__)


class ProjectManager:
    """Manage project creation, loading, and persistence."""

    PROJECTS_ROOT_NAME = "AnomalibProjects"
    PROJECT_FILE_NAME = "project.json"
    REQUIRED_DIRECTORIES = (
        "dataset/ok_train",
        "dataset/ok_test",
        "dataset/ng_test",
        "dataset/masks",
        "runs",
        "exports",
        "logs",
    )

    def __init__(self, default_root: Path) -> None:
        self.default_root = default_root

    def create_project(self, project_name: str, parent_directory: Path | None = None) -> ProjectConfig:
        """Create a new project with the required folder structure."""
        root = (parent_directory or self.default_root).expanduser().resolve()
        project_root = root / project_name
        project_root.mkdir(parents=True, exist_ok=False)
        for relative in self.REQUIRED_DIRECTORIES:
            (project_root / relative).mkdir(parents=True, exist_ok=True)

        project = ProjectConfig(name=project_name, project_path=str(project_root))
        self.save_project(project)
        LOGGER.info("Created project '%s' at %s", project_name, project_root)
        return project

    def load_project(self, project_file_or_directory: Path) -> ProjectConfig:
        """Load a project from its directory or project file."""
        path = project_file_or_directory.expanduser().resolve()
        project_file = path if path.name == self.PROJECT_FILE_NAME else path / self.PROJECT_FILE_NAME
        payload = self._read_project_payload(project_file)
        project = ProjectConfig.from_dict(payload)
        project.mark_opened()
        self.save_project(project)
        LOGGER.info("Loaded project '%s' from %s", project.name, project_file)
        return project

    def save_project(self, project: ProjectConfig) -> Path:
        """Save project metadata using an atomic write."""
        project_root = Path(project.project_path).expanduser().resolve()
        project_root.mkdir(parents=True, exist_ok=True)
        project_file = project_root / self.PROJECT_FILE_NAME
        payload = project.to_dict()
        self._atomic_write_json(project_file, payload)
        return project_file

    def create_run_directory(self, project: ProjectConfig, model_slug: str) -> Path:
        """Create a unique timestamped run directory."""
        runs_root = Path(project.project_path) / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_name = f"{timestamp}_{model_slug.lower()}"
        candidate = runs_root / base_name
        counter = 1
        while candidate.exists():
            counter += 1
            candidate = runs_root / f"{base_name}_{counter}"
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    def import_dataset_folder(self, source: Path, destination: Path, copy_files: bool) -> Path:
        """Copy a dataset folder into the project or keep the original reference."""
        source = source.expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Dataset folder does not exist: {source}")
        if not copy_files:
            return source
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
        return destination

    def _read_project_payload(self, project_file: Path) -> dict[str, Any]:
        try:
            return json.loads(project_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            backup_file = project_file.with_suffix(project_file.suffix + ".bak")
            shutil.copy2(project_file, backup_file)
            LOGGER.exception("Malformed project file backed up to %s", backup_file)
            raise

    def _atomic_write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        with NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as handle:
            handle.write(text)
            temp_path = Path(handle.name)
        temp_path.replace(path)
