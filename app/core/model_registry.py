"""Catalog of supported Anomalib 2.6.0 model configurations."""

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
    """Registry for image-folder model configurations supported by this app."""

    _MODEL_DEFINITIONS = (
        ModelDefinition(
            "patchcore",
            "PatchCore",
            "Patchcore",
            algorithm="PatchCore",
            model_variant="patchcore",
            support_level=ModelSupportLevel.PRODUCTION_VALIDATED,
        ),
        ModelDefinition(
            "padim",
            "PaDiM",
            "Padim",
            algorithm="PaDiM",
            model_variant="padim",
            support_level=ModelSupportLevel.PRODUCTION_VALIDATED,
        ),
        ModelDefinition(
            "dinomaly_dinov2",
            "Dinomaly (DINOv2)",
            "Dinomaly",
            algorithm="Dinomaly",
            model_variant="dinomaly_dinov2",
            encoder_family="DINOv2",
            support_level=ModelSupportLevel.PRODUCTION_VALIDATED,
        ),
        ModelDefinition(
            "dinomaly_dinov3",
            "Dinomaly (DINOv3)",
            "Dinomaly",
            algorithm="Dinomaly",
            model_variant="dinomaly_dinov3",
            encoder_family="DINOv3",
            requirement="Stock Anomalib 2.6.0 Dinomaly with a DINOv3 timm encoder.",
            support_level=ModelSupportLevel.PRODUCTION_VALIDATED,
        ),
        ModelDefinition(
            "anomaly_dino",
            "AnomalyDINO",
            "AnomalyDINO",
            algorithm="AnomalyDINO",
            model_variant="anomaly_dino",
            encoder_family="DINOv2",
            requirement="Export formats remain unavailable until deployment parity validation is completed.",
            supports_export=False,
            support_level=ModelSupportLevel.SUPPORTED,
        ),
        ModelDefinition(
            "super_add",
            "SuperADD",
            "SuperADD",
            algorithm="SuperADD",
            model_variant="super_add",
            encoder_family="DINOv3",
            requirement="Export formats remain unavailable until deployment parity validation is completed.",
            supports_export=False,
            support_level=ModelSupportLevel.SUPPORTED,
        ),
        ModelDefinition(
            "efficient_ad",
            "EfficientAD",
            "EfficientAd",
            algorithm="EfficientAD",
            model_variant="efficient_ad",
            requirement="Requires Anomalib's ImageNette reference data. Export formats remain unavailable pending parity validation.",
            supports_export=False,
            support_level=ModelSupportLevel.SUPPORTED,
        ),
        ModelDefinition(
            "supersimplenet",
            "SuperSimpleNet",
            "Supersimplenet",
            algorithm="SuperSimpleNet",
            model_variant="supersimplenet",
            requirement="Export formats remain unavailable until deployment parity validation is completed.",
            supports_export=False,
            support_level=ModelSupportLevel.SUPPORTED,
        ),
    )

    def __init__(self) -> None:
        self._models = {definition.key: definition for definition in self._MODEL_DEFINITIONS}
        self._aliases = {
            self._normalize(identifier): definition
            for definition in self._MODEL_DEFINITIONS
            for identifier in (definition.key, definition.display_name)
            if identifier
        }

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
        """Get a definition by one of the app's permanent IDs or display names."""
        try:
            return self._aliases[self._normalize(identifier)]
        except KeyError as exc:
            raise ValueError(f"Unsupported Anomalib model: {identifier}") from exc

    @staticmethod
    def _normalize(identifier: str) -> str:
        return "".join(character for character in identifier.lower() if character.isalnum())

