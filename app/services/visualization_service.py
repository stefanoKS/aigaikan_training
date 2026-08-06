"""Visualization helpers."""

from __future__ import annotations

from pathlib import Path


class VisualizationService:
    """Create paths for generated visualizations."""

    def visualization_directory(self, run_directory: Path) -> Path:
        """Return the visualization folder path."""
        path = run_directory / "visualizations"
        path.mkdir(parents=True, exist_ok=True)
        return path

