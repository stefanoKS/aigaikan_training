"""Project-draft state for the Preprocess Images preview source."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping


class PreviewSource(StrEnum):
    """Mutually exclusive image sources used only to render preprocessing previews."""

    PROJECT_GOOD_IMAGES = "project_good_images"
    CUSTOM_IMAGE = "custom_image"
    CUSTOM_FOLDER = "custom_folder"


@dataclass(slots=True)
class PreprocessingPreviewState:
    """Saved UI draft state intentionally excluded from training and deployment metadata."""

    source: PreviewSource = PreviewSource.PROJECT_GOOD_IMAGES
    custom_image_path: str = ""
    custom_folder_path: str = ""
    already_rectified: bool = False
    selected_index: int = 0

    def validate(self) -> None:
        if self.selected_index < 0:
            raise ValueError("Preprocessing preview image index cannot be negative.")
        if self.source is PreviewSource.CUSTOM_IMAGE and not self.custom_image_path:
            raise ValueError("Custom image preview source requires an image path.")
        if self.source is PreviewSource.CUSTOM_FOLDER and not self.custom_folder_path:
            raise ValueError("Custom folder preview source requires a folder path.")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "source": self.source.value,
            "custom_image_path": self.custom_image_path,
            "custom_folder_path": self.custom_folder_path,
            "already_rectified": self.already_rectified,
            "selected_index": self.selected_index,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "PreprocessingPreviewState":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise ValueError("Preprocessing preview state must be a JSON object.")
        result = cls(
            source=PreviewSource(str(payload.get("source", PreviewSource.PROJECT_GOOD_IMAGES.value))),
            custom_image_path=str(payload.get("custom_image_path", "")),
            custom_folder_path=str(payload.get("custom_folder_path", "")),
            already_rectified=bool(payload.get("already_rectified", False)),
            selected_index=int(payload.get("selected_index", 0)),
        )
        result.validate()
        return result