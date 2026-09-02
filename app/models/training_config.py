"""Training configuration models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from math import ceil
from pathlib import Path
from typing import Any

from app.core.dinomaly_encoder_registry import DinomalyEncoderRegistry
from app.core.threshold_calibrator import ThresholdCalibrationConfig, ThresholdMethod


class DeviceMode(StrEnum):
    """Device selection."""

    AUTO = "auto"
    CUDA = "cuda"
    CPU = "cpu"


_SUPPORTED_MODEL_NAMES = frozenset(
    {
        "patchcore",
        "padim",
        "dinomalydinov2",
        "dinomalydinov3",
        "anomalydino",
        "superadd",
        "efficientad",
        "supersimplenet",
    }
)


@dataclass(slots=True)
class TrainingConfig:
    """Persisted model and threshold configuration for one reproducible run."""

    model_name: str = "patchcore"
    device: DeviceMode = DeviceMode.AUTO
    batch_size: int = 8
    max_epochs: int = 1
    target_training_steps: int | None = None
    validation_every_n_epochs: int = 1
    gradient_clip_val: float = 0.0
    accumulate_grad_batches: int = 1
    random_seed: int = 42
    split_seed: int = 42
    num_workers: int = 0
    output_dir: str = ""
    dinomaly_encoder_id: str = ""
    dinomaly_decoder_depth: int = 8
    dinomaly_bottleneck_dropout: float = 0.2
    dinomaly_context_recentering: bool = False
    threshold_method: ThresholdMethod = ThresholdMethod.AUTO
    target_normal_false_reject_rate: float = 0.005
    minimum_required_ng_recall: float | None = None
    maximum_final_test_false_reject_rate: float = 0.005
    minimum_final_test_ok_images: int = 10
    minimum_final_test_ng_images: int = 10

    def __post_init__(self) -> None:
        """Apply defaults that do not depend on the selected dataset size."""
        if self.is_patchcore:
            if self.batch_size > 0:
                self.batch_size = 8
            self.max_epochs = 1
        elif self.is_padim:
            self.max_epochs = 1

    def validate(self) -> None:
        """Validate configuration values."""
        if self._normalized_model_name not in _SUPPORTED_MODEL_NAMES:
            raise ValueError(f"Unsupported production model: {self.model_name}")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.max_epochs <= 0:
            raise ValueError("Maximum epochs must be positive")
        if self.target_training_steps is not None and self.target_training_steps <= 0:
            raise ValueError("Dinomaly training-step override must be positive")
        if self.validation_every_n_epochs <= 0:
            raise ValueError("Validation frequency must be positive")
        if self.gradient_clip_val < 0:
            raise ValueError("Gradient clip value cannot be negative")
        if self.accumulate_grad_batches <= 0:
            raise ValueError("Gradient accumulation must be positive")
        if self.num_workers < 0:
            raise ValueError("Number of workers cannot be negative")
        if self.split_seed < 0:
            raise ValueError("Split seed cannot be negative")
        if self.dinomaly_decoder_depth <= 1:
            raise ValueError("Dinomaly decoder depth must be greater than one")
        if not 0 <= self.dinomaly_bottleneck_dropout < 1:
            raise ValueError("Dinomaly bottleneck dropout must be between 0 and 1")
        if not 0 <= self.maximum_final_test_false_reject_rate <= 1:
            raise ValueError("Maximum final-test false reject rate must be between zero and one")
        if self.minimum_final_test_ok_images <= 0 or self.minimum_final_test_ng_images <= 0:
            raise ValueError("Minimum final-test evidence counts must be positive")
        if self.uses_fixed_one_pass and self.max_epochs != 1:
            raise ValueError("PatchCore and PaDiM use exactly one epoch with their Anomalib trainer arguments")
        if self.is_patchcore and self.batch_size != 8:
            raise ValueError("PatchCore uses a batch size of 8")
        if self.is_dinomaly and (
            self.dinomaly_decoder_depth != 8
            or self.dinomaly_bottleneck_dropout != 0.2
            or self.dinomaly_context_recentering
        ):
            raise ValueError("Dinomaly settings must use the supported stock profile")
        if self.is_dinomaly:
            DinomalyEncoderRegistry().validate_for_family(
                self.dinomaly_encoder_name,
                "DINOv3" if self.is_dinomaly_dinov3 else "DINOv2",
            )
        ThresholdCalibrationConfig(
            method=self.threshold_method,
            target_normal_false_reject_rate=self.target_normal_false_reject_rate,
            minimum_required_ng_recall=self.minimum_required_ng_recall,
        ).validate()

    @property
    def is_patchcore(self) -> bool:
        """Return whether this config represents the production PatchCore path."""
        return self._normalized_model_name == "patchcore"

    @property
    def is_padim(self) -> bool:
        """Return whether this config represents the production PaDiM path."""
        return self._normalized_model_name == "padim"

    @property
    def is_dinomaly(self) -> bool:
        """Return whether this config represents either explicit Dinomaly variant."""
        return self.is_dinomaly_dinov2 or self.is_dinomaly_dinov3

    @property
    def is_dinomaly_dinov2(self) -> bool:
        """Return whether this config selects stock Anomalib Dinomaly."""
        return self._normalized_model_name == "dinomalydinov2"

    @property
    def is_dinomaly_dinov3(self) -> bool:
        """Return whether this config selects stock Dinomaly with a DINOv3 encoder."""
        return self._normalized_model_name == "dinomalydinov3"

    @property
    def uses_fixed_one_pass(self) -> bool:
        """Return whether the selected model builds a memory bank in one pass."""
        return self.is_patchcore or self.is_padim

    @property
    def dinomaly_encoder_name(self) -> str:
        """Return the explicit stock timm encoder selected by the Dinomaly variant."""
        if self.dinomaly_encoder_id:
            return self.dinomaly_encoder_id
        if self.is_dinomaly_dinov3:
            return "vit_base_patch16_dinov3.lvd1689m"
        return "vit_base_patch14_reg4_dinov2"

    def model_profile(self) -> dict[str, object]:
        """Return the fixed model-specific contract persisted with each run."""
        if self.is_patchcore:
            return {
                "backbone": "wide_resnet50_2",
                "layers": ["layer2", "layer3"],
                "coreset_sampling_ratio": 0.1,
                "num_neighbors": 9,
                "batch_size": 8,
                "max_epochs": 1,
                "preprocessing": "anomalib-native",
            }
        if self.is_padim:
            return {
                "backbone": "resnet18",
                "layers": ["layer1", "layer2", "layer3"],
                "max_epochs": 1,
                "preprocessing": "anomalib-native",
            }
        if self._normalized_model_name == "anomalydino":
            return {
                "encoder_name": "vit_small_patch14_dinov2",
                "num_neighbours": 1,
                "sampling_ratio": 0.1,
                "preprocessing": "anomalib-native",
            }
        if self._normalized_model_name == "superadd":
            return {
                "backbone": "vit_huge_plus_patch16_dinov3",
                "patch_size": 448,
                "patch_overlap": 16,
                "preprocessing": "anomalib-native",
            }
        if self._normalized_model_name == "efficientad":
            return {
                "model_size": "small",
                "teacher_out_channels": 384,
                "preprocessing": "anomalib-native",
            }
        if self._normalized_model_name == "supersimplenet":
            return {
                "backbone": "wide_resnet50_2.tv_in1k",
                "layers": ["layer2", "layer3"],
                "preprocessing": "anomalib-native",
            }
        profile: dict[str, object] = {
            "encoder_name": self.dinomaly_encoder_name,
            "decoder_depth": 8,
            "bottleneck_dropout": 0.2,
            "use_context_recentering": False,
            "training": "step-based",
            "max_steps": self.target_training_steps if self.target_training_steps is not None else "auto",
            "preprocessing": "anomalib-native",
        }
        if self.is_dinomaly_dinov3:
            profile["preprocessing"] = {
                "resize_size": [448, 448],
                "center_crop_size": [384, 384],
                "encoder_patch_size": 16,
            }
        return profile

    def resolved_dinomaly_training_steps(self, training_image_count: int) -> int:
        """Return the baseline or explicitly overridden Dinomaly optimizer-step budget."""
        if not self.is_dinomaly:
            raise ValueError("Dinomaly training steps are available only for Dinomaly configurations")
        if self.target_training_steps is not None:
            return self.target_training_steps
        steps_per_epoch = max(ceil(max(training_image_count, 1) / self.batch_size), 1)
        return max(5000, steps_per_epoch)

    def recommended_epochs(self, training_image_count: int) -> int:
        """Return the model-specific epoch recommendation for the selected data volume."""
        if self.uses_fixed_one_pass:
            return 1
        steps_per_epoch = max(ceil(max(training_image_count, 1) / self.batch_size), 1)
        if self.is_dinomaly:
            return min(max(ceil(self.resolved_dinomaly_training_steps(training_image_count) / steps_per_epoch), 1), 10000)
        return self.max_epochs

    def estimated_training_steps(self, training_image_count: int) -> int:
        """Return the number of optimizer steps expected from the persisted configuration."""
        steps_per_epoch = max(ceil(max(training_image_count, 1) / self.batch_size), 1)
        if self.uses_fixed_one_pass:
            return steps_per_epoch
        if self.is_dinomaly:
            return self.resolved_dinomaly_training_steps(training_image_count)
        return steps_per_epoch * self.max_epochs

    def apply_model_defaults(self, training_image_count: int) -> None:
        """Apply only model-required defaults before persisting a configuration."""
        if self.is_patchcore:
            self.batch_size = 8
            self.max_epochs = 1
        elif self.is_padim:
            self.max_epochs = 1
        elif self.is_dinomaly:
            self.dinomaly_decoder_depth = 8
            self.dinomaly_bottleneck_dropout = 0.2
            self.dinomaly_context_recentering = False

    @property
    def _normalized_model_name(self) -> str:
        return "".join(character for character in self.model_name.casefold() if character.isalnum())

    def resolved_output_dir(self, project_path: Path) -> Path:
        """Return the output directory with a project-relative default."""
        if self.output_dir:
            return Path(self.output_dir)
        return project_path / "runs"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration."""
        payload = asdict(self)
        payload["device"] = self.device.value
        payload["threshold_method"] = self.threshold_method.value
        payload["model_profile"] = self.model_profile()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingConfig":
        """Deserialize the configuration."""
        model_name = str(payload.get("model_name", "patchcore"))
        legacy_encoder = str(payload.get("dinomaly_encoder", "vit_base_patch14_reg4_dinov2"))
        normalized_model_name = "".join(character for character in model_name.casefold() if character.isalnum())
        if normalized_model_name == "dinomaly":
            model_name = "dinomaly_dinov3" if "dinov3" in legacy_encoder.casefold() else "dinomaly_dinov2"
        return cls(
            model_name=model_name,
            device=DeviceMode(payload.get("device", DeviceMode.AUTO.value)),
            batch_size=int(payload.get("batch_size", 8)),
            max_epochs=int(payload.get("max_epochs", 1)),
            target_training_steps=(
                int(payload["target_training_steps"])
                if payload.get("target_training_steps") not in (None, 0, "")
                else None
            ),
            validation_every_n_epochs=int(payload.get("validation_every_n_epochs", 1)),
            gradient_clip_val=float(payload.get("gradient_clip_val", 0.0)),
            accumulate_grad_batches=int(payload.get("accumulate_grad_batches", 1)),
            random_seed=int(payload.get("random_seed", 42)),
            split_seed=int(payload.get("split_seed", payload.get("random_seed", 42))),
            num_workers=int(payload.get("num_workers", 0)),
            output_dir=payload.get("output_dir", ""),
            dinomaly_encoder_id=_curated_dinomaly_encoder_id(
                payload.get("dinomaly_encoder_id", payload.get("dinomaly_encoder", ""))
            ),
            dinomaly_decoder_depth=8,
            dinomaly_bottleneck_dropout=0.2,
            dinomaly_context_recentering=False,
            threshold_method=ThresholdMethod(payload.get("threshold_method", ThresholdMethod.AUTO.value)),
            target_normal_false_reject_rate=float(payload.get("target_normal_false_reject_rate", 0.005)),
            minimum_required_ng_recall=(
                float(payload["minimum_required_ng_recall"])
                if payload.get("minimum_required_ng_recall") is not None
                else None
            ),
            maximum_final_test_false_reject_rate=float(payload.get("maximum_final_test_false_reject_rate", 0.005)),
            minimum_final_test_ok_images=int(payload.get("minimum_final_test_ok_images", 10)),
            minimum_final_test_ng_images=int(payload.get("minimum_final_test_ng_images", 10)),
        )


def _curated_dinomaly_encoder_id(value: object) -> str:
    """Retain only persisted identifiers that remain in the curated encoder catalog."""
    identifier = str(value)
    try:
        DinomalyEncoderRegistry().get(identifier)
    except ValueError:
        return ""
    return identifier
