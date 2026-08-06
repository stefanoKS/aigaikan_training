"""Application settings management."""

from __future__ import annotations

import os
from pathlib import Path


class SettingsManager:
    """Small wrapper around QSettings-like behavior for defaults."""

    ORGANIZATION = "AnomalibTrainer"
    APPLICATION = "AnomalibTrainer"

    def app_data_directory(self) -> Path:
        """Return the per-user application data directory."""
        local = os.environ.get("LOCALAPPDATA")
        if local:
            return Path(local) / self.APPLICATION
        return Path.home() / ".anomalib_trainer"

    def default_projects_directory(self) -> Path:
        """Return the default projects directory."""
        documents = Path.home() / "Documents"
        return documents / "AnomalibProjects"

