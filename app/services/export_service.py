"""Model export abstraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ExportResult:
    """Result of a model export operation."""

    exported_path: Path
    export_format: str


class ExportService:
    """Abstraction for model export."""

    def verify_export(self, path: Path) -> ExportResult:
        """Verify the export exists before claiming success."""
        if not path.exists():
            raise FileNotFoundError(f"Exported model not found: {path}")
        return ExportResult(exported_path=path, export_format=path.suffix.lstrip("."))

