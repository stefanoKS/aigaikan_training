"""Dataset configuration models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

SUPPORTED_IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")


class DatasetRole(StrEnum):
    """Supported dataset roles."""

    OK_TRAIN = "ok_train"
    OK_TEST = "ok_test"
    NG_TEST = "ng_test"
    MASKS = "masks"


class FolderImportMode(StrEnum):
    """Import behavior for external folders."""

    COPY = "copy"
    REFERENCE = "reference"


@dataclass(slots=True)
class ImportedFolder:
    """Description of an imported folder."""

    role: DatasetRole
    path: str = ""
    import_mode: FolderImportMode = FolderImportMode.COPY
    image_count: int = 0
    invalid_image_count: int = 0
    typical_resolution: str = ""
    color_mode: str = ""
    thumbnail_paths: list[str] = field(default_factory=list)

    def resolved_path(self) -> Path | None:
        """Return the folder path when configured."""
        return Path(self.path) if self.path else None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the folder."""
        payload = asdict(self)
        payload["role"] = self.role.value
        payload["import_mode"] = self.import_mode.value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ImportedFolder":
        """Deserialize the folder."""
        return cls(
            role=DatasetRole(payload["role"]),
            path=payload.get("path", ""),
            import_mode=FolderImportMode(payload.get("import_mode", FolderImportMode.COPY.value)),
            image_count=int(payload.get("image_count", 0)),
            invalid_image_count=int(payload.get("invalid_image_count", 0)),
            typical_resolution=payload.get("typical_resolution", ""),
            color_mode=payload.get("color_mode", ""),
            thumbnail_paths=list(payload.get("thumbnail_paths", [])),
        )


@dataclass(slots=True)
class DatasetConfig:
    """Project dataset configuration."""

    folders: dict[DatasetRole, ImportedFolder] = field(
        default_factory=lambda: {
            role: ImportedFolder(role=role)
            for role in (DatasetRole.OK_TRAIN, DatasetRole.OK_TEST, DatasetRole.NG_TEST, DatasetRole.MASKS)
        }
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration."""
        return {role.value: folder.to_dict() for role, folder in self.folders.items()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetConfig":
        """Deserialize the configuration."""
        folders = {
            role: ImportedFolder.from_dict(payload.get(role.value, {"role": role.value}))
            for role in (DatasetRole.OK_TRAIN, DatasetRole.OK_TEST, DatasetRole.NG_TEST, DatasetRole.MASKS)
        }
        return cls(folders=folders)
