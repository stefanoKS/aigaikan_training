"""Dataset import helpers."""

from __future__ import annotations

from pathlib import Path

from app.models.dataset_config import DatasetConfig, DatasetRole, FolderImportMode


class DatasetManager:
    """Manage dataset folder selections."""

    def assign_folder(
        self,
        config: DatasetConfig,
        role: DatasetRole,
        path: Path,
        import_mode: FolderImportMode,
    ) -> None:
        """Assign a folder to a dataset role."""
        folder = config.folders[role]
        folder.path = str(path)
        folder.import_mode = import_mode

    def clear(self, config: DatasetConfig, role: DatasetRole | None = None) -> None:
        """Clear one or all dataset roles."""
        roles = [role] if role else list(config.folders)
        for current_role in roles:
            folder = config.folders[current_role]
            folder.path = ""
            folder.image_count = 0
            folder.invalid_image_count = 0
            folder.typical_resolution = ""
            folder.color_mode = ""
            folder.thumbnail_paths.clear()

