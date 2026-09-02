"""Project metadata models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dataset_config import DatasetConfig
from .inspection_region import InspectionRegionConfig
from .training_config import TrainingConfig

ISO_FORMAT = "%Y-%m-%dT%H:%M:%S.%f%z"


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(tz=timezone.utc)


@dataclass(slots=True)
class RecentProject:
    """Global application recent project entry."""

    name: str
    path: str
    last_opened: str
    status: str = "Not trained"


@dataclass(slots=True)
class ProjectConfig:
    """Full project state."""

    name: str
    project_path: str
    created_at: str = field(default_factory=lambda: utc_now().strftime(ISO_FORMAT))
    last_opened_at: str = field(default_factory=lambda: utc_now().strftime(ISO_FORMAT))
    last_training_status: str = "Not trained"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inspection_region: InspectionRegionConfig = field(default_factory=InspectionRegionConfig)

    @property
    def root_path(self) -> Path:
        """Return project root path."""
        return Path(self.project_path)

    def mark_opened(self) -> None:
        """Update last-opened timestamp."""
        self.last_opened_at = utc_now().strftime(ISO_FORMAT)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the project."""
        payload = asdict(self)
        payload["dataset"] = self.dataset.to_dict()
        payload["training"] = self.training.to_dict()
        payload.pop("inspection_region", None)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProjectConfig":
        """Deserialize the project."""
        return cls(
            name=payload["name"],
            project_path=payload["project_path"],
            created_at=payload.get("created_at", utc_now().strftime(ISO_FORMAT)),
            last_opened_at=payload.get("last_opened_at", utc_now().strftime(ISO_FORMAT)),
            last_training_status=payload.get("last_training_status", "Not trained"),
            dataset=DatasetConfig.from_dict(payload.get("dataset", {})),
            training=TrainingConfig.from_dict(payload.get("training", {})),
            inspection_region=InspectionRegionConfig.from_dict(payload.get("inspection_region", {})),
        )
