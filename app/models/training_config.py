"""Training configuration models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from math import ceil
from pathlib import Path
import re
from typing import Any

from app.core.threshold_calibrator import ThresholdCalibrationConfig, ThresholdMethod


class DeviceMode(StrEnum):
    """Device selection."""

    AUTO = "auto"
    CUDA = "cuda"
    CPU = "cpu"


@dataclass(slots=True)
class TrainingConfig:
    """Persisted model and threshold configuration for one reproducible run."""

    model_name: str = "patchcore"
    device: DeviceMode = DeviceMode.AUTO
    image_width: int = 280
    image_height: int = 280
    batch_size: int = 8
    max_epochs: int = 1
    target_training_steps: int = 3000
    validation_every_n_epochs: int = 1
    gradient_clip_val: float = 0.0
    accumulate_grad_batches: int = 1
    random_seed: int = 42
    split_seed: int = 42
    coreset_sampling_ratio: float = 0.1
    num_neighbors: int = 9
    num_workers: int = 0
    output_dir: str = ""
    backbone: str = "wide_resnet50_2"
    layers: tuple[str, ...] = ("layer2", "layer3")
    dinomaly_encoder: str = "vit_base_patch14_reg4_dinov2"
    dinomaly_dinov3_encoder: str = "vit_small_patch16_dinov3.lvd1689m"
    dinov3_feature_layers: tuple[int, ...] = ()
    dinomaly_decoder_depth: int = 8
    dinomaly_bottleneck_dropout: float = 0.2
    dinomaly_context_recentering: bool = False
    superadd_encoder: str = "vit_huge_plus_patch16_dinov3.lvd1689m"
    superadd_patch_size: int = 448
    superadd_patch_overlap: int = 16
    threshold_method: ThresholdMethod = ThresholdMethod.AUTO
    target_normal_false_reject_rate: float = 0.005
    minimum_required_ng_recall: float | None = None
    supplemental_data_path: str = ""
    zero_shot_class_name: str = ""

    def validate(self) -> None:
        """Validate configuration values."""
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("Image dimensions must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.max_epochs <= 0:
            raise ValueError("Maximum epochs must be positive")
        if self.target_training_steps < 1000:
            raise ValueError("Dinomaly target training steps must be at least 1000")
        if self.validation_every_n_epochs <= 0:
            raise ValueError("Validation frequency must be positive")
        if self.gradient_clip_val < 0:
            raise ValueError("Gradient clip value cannot be negative")
        if self.accumulate_grad_batches <= 0:
            raise ValueError("Gradient accumulation must be positive")
        if not 0 < self.coreset_sampling_ratio <= 1:
            raise ValueError("Coreset sampling ratio must be between 0 and 1")
        if self.num_neighbors <= 0:
            raise ValueError("Number of nearest neighbors must be positive")
        if self.num_workers < 0:
            raise ValueError("Number of workers cannot be negative")
        if self.split_seed < 0:
            raise ValueError("Split seed cannot be negative")
        if self.dinomaly_decoder_depth <= 1:
            raise ValueError("Dinomaly decoder depth must be greater than one")
        if not 0 <= self.dinomaly_bottleneck_dropout < 1:
            raise ValueError("Dinomaly bottleneck dropout must be between 0 and 1")
        if self.uses_fixed_one_pass and self.max_epochs != 1:
            raise ValueError("PatchCore and SuperADD use exactly one epoch to build their memory banks")
        if self.is_dinomaly_dinov2:
            patch_size = self._encoder_patch_size()
            if self.image_width % patch_size != 0 or self.image_height % patch_size != 0:
                raise ValueError(f"Dinomaly DINOv2 image dimensions must be divisible by {patch_size}")
        if self.is_dinomaly_dinov3 and self.dinov3_feature_layers and min(self.dinov3_feature_layers) < 0:
            raise ValueError("Dinomaly DINOv3 feature layers cannot be negative")
        if self.is_superadd and (self.superadd_patch_size <= 0 or self.superadd_patch_overlap <= 0):
            raise ValueError("SuperADD patch size and overlap must be positive")
        if self.is_superadd and self.superadd_patch_overlap * 2 >= self.superadd_patch_size:
            raise ValueError("SuperADD patch overlap must be less than half of its patch size")
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
    def is_dinomaly(self) -> bool:
        """Return whether this config represents either explicit Dinomaly variant."""
        return self.is_dinomaly_dinov2 or self.is_dinomaly_dinov3

    @property
    def is_dinomaly_dinov2(self) -> bool:
        """Return whether this config selects stock Anomalib Dinomaly."""
        return self._normalized_model_name in {"dinomaly", "dinomalydinov2"}

    @property
    def is_dinomaly_dinov3(self) -> bool:
        """Return whether this config selects the application-side experimental adapter."""
        return self._normalized_model_name == "dinomalydinov3"

    @property
    def is_superadd(self) -> bool:
        """Return whether this config selects Anomalib's native SuperADD algorithm."""
        return self._normalized_model_name in {"superadd", "superadddinov3"}

    @property
    def uses_fixed_one_pass(self) -> bool:
        """Return whether the selected model builds a memory bank in one pass."""
        return self.is_patchcore or self.is_superadd

    @property
    def model_input_size(self) -> tuple[int, int]:
        """Return Anomalib's model input shape as height then width."""
        return self.image_height, self.image_width

    def recommended_epochs(self, training_image_count: int) -> int:
        """Return the model-specific epoch recommendation for the selected data volume."""
        if self.uses_fixed_one_pass:
            return 1
        steps_per_epoch = max(ceil(max(training_image_count, 1) / self.batch_size), 1)
        return min(max(ceil(self.target_training_steps / steps_per_epoch), 1), 10000)

    def estimated_training_steps(self, training_image_count: int) -> int:
        """Return the number of optimizer steps expected from the persisted configuration."""
        steps_per_epoch = max(ceil(max(training_image_count, 1) / self.batch_size), 1)
        return steps_per_epoch if self.uses_fixed_one_pass else steps_per_epoch * self.max_epochs

    def apply_model_defaults(self, training_image_count: int) -> None:
        """Apply only model-required defaults before persisting a configuration."""
        if self.uses_fixed_one_pass:
            self.max_epochs = 1
        elif self.is_dinomaly and self.max_epochs == 1:
            self.max_epochs = self.recommended_epochs(training_image_count)

    @property
    def _normalized_model_name(self) -> str:
        return "".join(character for character in self.model_name.casefold() if character.isalnum())

    def _encoder_patch_size(self) -> int:
        match = re.search(r"patch(\d+)", self.dinomaly_encoder.casefold())
        return int(match.group(1)) if match else 14

    def resolved_output_dir(self, project_path: Path) -> Path:
        """Return the output directory with a project-relative default."""
        if self.output_dir:
            return Path(self.output_dir)
        return project_path / "runs"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration."""
        payload = asdict(self)
        payload["device"] = self.device.value
        payload["layers"] = list(self.layers)
        payload["dinov3_feature_layers"] = list(self.dinov3_feature_layers)
        payload["threshold_method"] = self.threshold_method.value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingConfig":
        """Deserialize the configuration."""
        model_name = str(payload.get("model_name", "patchcore"))
        legacy_encoder = str(payload.get("dinomaly_encoder", "vit_base_patch14_reg4_dinov2"))
        normalized_model_name = "".join(character for character in model_name.casefold() if character.isalnum())
        if normalized_model_name == "dinomaly":
            model_name = "dinomaly_dinov3" if "dinov3" in legacy_encoder.casefold() else "dinomaly_dinov2"
        elif normalized_model_name == "superadd":
            model_name = "superadd_dinov3"
        return cls(
            model_name=model_name,
            device=DeviceMode(payload.get("device", DeviceMode.AUTO.value)),
            image_width=int(payload.get("image_width", 280)),
            image_height=int(payload.get("image_height", 280)),
            batch_size=int(payload.get("batch_size", 8)),
            max_epochs=int(payload.get("max_epochs", 1)),
            target_training_steps=int(payload.get("target_training_steps", 3000)),
            validation_every_n_epochs=int(payload.get("validation_every_n_epochs", 1)),
            gradient_clip_val=float(payload.get("gradient_clip_val", 0.0)),
            accumulate_grad_batches=int(payload.get("accumulate_grad_batches", 1)),
            random_seed=int(payload.get("random_seed", 42)),
            split_seed=int(payload.get("split_seed", payload.get("random_seed", 42))),
            coreset_sampling_ratio=float(payload.get("coreset_sampling_ratio", 0.1)),
            num_neighbors=int(payload.get("num_neighbors", 9)),
            num_workers=int(payload.get("num_workers", 0)),
            output_dir=payload.get("output_dir", ""),
            backbone=payload.get("backbone", "wide_resnet50_2"),
            layers=tuple(payload.get("layers", ["layer2", "layer3"])),
            dinomaly_encoder=legacy_encoder,
            dinomaly_dinov3_encoder=payload.get(
                "dinomaly_dinov3_encoder",
                legacy_encoder
                if normalized_model_name == "dinomaly" and "dinov3" in legacy_encoder.casefold()
                else "vit_small_patch16_dinov3.lvd1689m",
            ),
            dinov3_feature_layers=tuple(int(layer) for layer in payload.get("dinov3_feature_layers", [])),
            dinomaly_decoder_depth=int(payload.get("dinomaly_decoder_depth", 8)),
            dinomaly_bottleneck_dropout=float(payload.get("dinomaly_bottleneck_dropout", 0.2)),
            dinomaly_context_recentering=bool(payload.get("dinomaly_context_recentering", False)),
            superadd_encoder=payload.get("superadd_encoder", "vit_huge_plus_patch16_dinov3.lvd1689m"),
            superadd_patch_size=int(payload.get("superadd_patch_size", 448)),
            superadd_patch_overlap=int(payload.get("superadd_patch_overlap", 16)),
            threshold_method=ThresholdMethod(payload.get("threshold_method", ThresholdMethod.AUTO.value)),
            target_normal_false_reject_rate=float(payload.get("target_normal_false_reject_rate", 0.005)),
            minimum_required_ng_recall=(
                float(payload["minimum_required_ng_recall"])
                if payload.get("minimum_required_ng_recall") is not None
                else None
            ),
            supplemental_data_path=payload.get("supplemental_data_path", ""),
            zero_shot_class_name=payload.get("zero_shot_class_name", ""),
        )
