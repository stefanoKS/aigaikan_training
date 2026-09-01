"""Catalog of supported Anomalib 2.5.1 model configurations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ModelInputType(StrEnum):
    """Dataset contract required by an Anomalib model."""

    IMAGE_FOLDER = "image-folder"
    VIDEO = "video"


class ModelExecutionMode(StrEnum):
    """Whether the model learns from the training split."""

    TRAIN = "train"
    EVALUATE = "evaluate"


class ModelSupportLevel(StrEnum):
    """Validation status of a selectable model."""

    SUPPORTED = "supported"
    PRODUCTION_VALIDATED = "production-validated"
    EXPERIMENTAL = "experimental"


@dataclass(frozen=True, slots=True)
class ModelDefinition:
    """Descriptor for a model exported by Anomalib."""

    key: str
    display_name: str
    anomalib_class_name: str | None
    algorithm: str = ""
    model_variant: str = ""
    encoder_family: str = ""
    official_anomalib_implementation: bool = True
    input_type: ModelInputType = ModelInputType.IMAGE_FOLDER
    execution_mode: ModelExecutionMode = ModelExecutionMode.TRAIN
    requirement: str = ""
    supports_export: bool = True
    support_level: ModelSupportLevel = ModelSupportLevel.EXPERIMENTAL

    @property
    def supports_image_folder(self) -> bool:
        """Return whether this project type can run the model."""
        return self.input_type is ModelInputType.IMAGE_FOLDER


class ModelRegistry:
    """Registry for the four image-folder configurations supported by this app."""

    _MODEL_DEFINITIONS = (
        ModelDefinition(
            "patchcore",
            "PatchCore",
            "Patchcore",
            algorithm="PatchCore",
            model_variant="patchcore",
            support_level=ModelSupportLevel.SUPPORTED,
        ),
        ModelDefinition(
            "padim",
            "PaDiM",
            "Padim",
            algorithm="PaDiM",
            model_variant="padim",
            support_level=ModelSupportLevel.SUPPORTED,
        ),
        ModelDefinition(
            "dinomaly_dinov2",
            "Dinomaly (DINOv2)",
            "Dinomaly",
            algorithm="Dinomaly",
            model_variant="dinomaly_dinov2",
            encoder_family="DINOv2",
            support_level=ModelSupportLevel.SUPPORTED,
        ),
        ModelDefinition(
            "dinomaly_dinov3",
            "Dinomaly (DINOv3)",
            "Dinomaly",
            algorithm="Dinomaly",
            model_variant="dinomaly_dinov3",
            encoder_family="DINOv3",
            requirement="Stock Anomalib 2.5.1 Dinomaly with a DINOv3 timm encoder.",
            support_level=ModelSupportLevel.SUPPORTED,
        ),
    )

    def __init__(self) -> None:
        self._models = {definition.key: definition for definition in self._MODEL_DEFINITIONS}
        self._aliases = {
            self._normalize(identifier): definition
            for definition in self._MODEL_DEFINITIONS
            for identifier in (definition.key, definition.display_name, definition.anomalib_class_name)
            if identifier
        }
        self._aliases.update(
            {
                "dinomaly": self._models["dinomaly_dinov2"],
            }
        )

    def all(self) -> list[ModelDefinition]:
        """Return every current Anomalib model definition."""
        return list(self._MODEL_DEFINITIONS)

    def image_folder_models(self) -> list[ModelDefinition]:
        """Return models compatible with this application's folder dataset."""
        return [definition for definition in self._MODEL_DEFINITIONS if definition.supports_image_folder]

    def production_models(self) -> list[ModelDefinition]:
        """Return the models that completed the production validation contract."""
        return [
            definition
            for definition in self.image_folder_models()
            if definition.support_level is ModelSupportLevel.PRODUCTION_VALIDATED
        ]

    def official_anomalib_models(self) -> list[ModelDefinition]:
        """Return definitions backed directly by installed Anomalib classes."""
        return [
            definition
            for definition in self._MODEL_DEFINITIONS
            if definition.official_anomalib_implementation and definition.anomalib_class_name
        ]

    def get(self, identifier: str) -> ModelDefinition:
        """Get a definition by stored key, display name, or Anomalib class name."""
        try:
            return self._aliases[self._normalize(identifier)]
        except KeyError as exc:
            raise ValueError(f"Unsupported Anomalib model: {identifier}") from exc

    @staticmethod
    def _normalize(identifier: str) -> str:
        return "".join(character for character in identifier.lower() if character.isalnum())

