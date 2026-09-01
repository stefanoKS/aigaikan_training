"""Catalog of Anomalib 2.6 model capabilities."""

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
    """Registry for all concrete models exported by Anomalib 2.6.0.

    The UI manages still images through ``anomalib.data.Folder``. Video models
    remain visible for accurate capability reporting, but cannot be selected
    until a video-project workflow is added.
    """

    _MODEL_DEFINITIONS = (
        ModelDefinition(
            "patchcore",
            "PatchCore",
            "Patchcore",
            algorithm="PatchCore",
            model_variant="patchcore",
            support_level=ModelSupportLevel.PRODUCTION_VALIDATED,
        ),
        ModelDefinition("padim", "PaDiM", "Padim"),
        ModelDefinition("cfa", "CFA", "Cfa"),
        ModelDefinition("cflow", "CFlow", "Cflow"),
        ModelDefinition("csflow", "CS-Flow", "Csflow"),
        ModelDefinition("dfkde", "DFKDE", "Dfkde"),
        ModelDefinition("dfm", "DFM", "Dfm"),
        ModelDefinition("fastflow", "FastFlow", "Fastflow"),
        ModelDefinition("fre", "FRE", "Fre"),
        ModelDefinition("ganomaly", "GANomaly", "Ganomaly"),
        ModelDefinition("reverse_distillation", "Reverse Distillation", "ReverseDistillation"),
        ModelDefinition("stfpm", "STFPM", "Stfpm"),
        ModelDefinition("uflow", "U-Flow", "Uflow"),
        ModelDefinition("draem", "DRAEM", "Draem", requirement="Requires a DTD texture dataset folder."),
        ModelDefinition("dsr", "DSR", "Dsr"),
        ModelDefinition(
            "efficientad",
            "EfficientAD",
            "EfficientAd",
            requirement="Requires an ImageNet/Imagenette folder for teacher normalization.",
        ),
        ModelDefinition("glass", "GLASS", "Glass", requirement="Optionally uses an anomaly-source image folder."),
        ModelDefinition("supersimplenet", "SuperSimpleNet", "Supersimplenet"),
        ModelDefinition("uninet", "UniNet", "UniNet"),
        ModelDefinition("patchflow", "PatchFlow", "Patchflow"),
        ModelDefinition("generalad", "GeneralAD", "GeneralAD"),
        ModelDefinition("l2bt", "L2BT", "L2BT"),
        ModelDefinition("cfm", "CFM", "CFM", requirement="Requires PointMAE weights."),
        ModelDefinition("anomaly_dino", "AnomalyDINO", "AnomalyDINO"),
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
            "Dinomaly (DINOv3, Experimental)",
            None,
            algorithm="Dinomaly",
            model_variant="dinomaly_dinov3",
            encoder_family="DINOv3",
            official_anomalib_implementation=False,
            requirement="Application-side DINOv3 encoder adapter; not stock Anomalib Dinomaly.",
        ),
        ModelDefinition("inpformer", "INP-Former", "InpFormer"),
        ModelDefinition(
            "superadd_dinov3",
            "SuperADD (DINOv3, Experimental)",
            "SuperADD",
            algorithm="SuperADD",
            model_variant="superadd_dinov3",
            encoder_family="DINOv3",
            requirement="Native Anomalib DINOv3 memory-bank comparison model.",
        ),
        ModelDefinition(
            "anomalyvfm",
            "AnomalyVFM",
            "AnomalyVFM",
            execution_mode=ModelExecutionMode.EVALUATE,
            requirement="Zero-shot model; runs evaluation directly.",
        ),
        ModelDefinition(
            "winclip",
            "WinCLIP",
            "WinClip",
            execution_mode=ModelExecutionMode.EVALUATE,
            requirement="Zero-shot/few-shot model; a class name improves prompt quality.",
        ),
        ModelDefinition(
            "vlmad",
            "VLM-AD",
            "VlmAd",
            execution_mode=ModelExecutionMode.EVALUATE,
            requirement="Requires a configured Ollama or cloud VLM service.",
        ),
        ModelDefinition(
            "aivad",
            "AI-VAD",
            "AiVad",
            input_type=ModelInputType.VIDEO,
            requirement="Requires a video dataset project.",
        ),
        ModelDefinition(
            "fuvas",
            "FUVAS",
            "Fuvas",
            input_type=ModelInputType.VIDEO,
            requirement="Requires a video dataset project.",
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
                "superadd": self._models["superadd_dinov3"],
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

