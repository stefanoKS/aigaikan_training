"""Training configuration models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from math import ceil
from pathlib import Path
import re
from typing import Any


class DeviceMode(StrEnum):
    """Device selection."""

    AUTO = "auto"
    CUDA = "cuda"
    CPU = "cpu"


@dataclass(slots=True)
class TrainingConfig:
    """PatchCore training configuration."""

    model_name: str = "PatchCore"
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
    dinomaly_decoder_depth: int = 8
    dinomaly_bottleneck_dropout: float = 0.2
    dinomaly_context_recentering: bool = False
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
        if self.is_patchcore and self.max_epochs != 1:
            raise ValueError("PatchCore uses exactly one epoch to build its memory bank")
        if self.is_dinomaly:
            patch_size = self._encoder_patch_size()
            if self.image_width % patch_size != 0 or self.image_height % patch_size != 0:
                raise ValueError(f"Dinomaly image dimensions must be divisible by {patch_size}")

    @property
    def is_patchcore(self) -> bool:
        """Return whether this config represents the production PatchCore path."""
        return self.model_name.casefold().replace("-", "") == "patchcore"

    @property
    def is_dinomaly(self) -> bool:
        """Return whether this config represents the production Dinomaly path."""
        return self.model_name.casefold().replace("-", "") == "dinomaly"

    @property
    def model_input_size(self) -> tuple[int, int]:
        """Return Anomalib's model input shape as height then width."""
        return self.image_height, self.image_width

    def recommended_epochs(self, training_image_count: int) -> int:
        """Return the model-specific epoch recommendation for the selected data volume."""
        if self.is_patchcore:
            return 1
        steps_per_epoch = max(ceil(max(training_image_count, 1) / self.batch_size), 1)
        return min(max(ceil(self.target_training_steps / steps_per_epoch), 1), 10000)

    def estimated_training_steps(self, training_image_count: int) -> int:
        """Return the number of optimizer steps expected from the persisted configuration."""
        steps_per_epoch = max(ceil(max(training_image_count, 1) / self.batch_size), 1)
        return steps_per_epoch if self.is_patchcore else steps_per_epoch * self.max_epochs

    def apply_model_defaults(self, training_image_count: int) -> None:
        """Apply only model-required defaults before persisting a configuration."""
        if self.is_patchcore:
            self.max_epochs = 1
        elif self.is_dinomaly and self.max_epochs == 1:
            self.max_epochs = self.recommended_epochs(training_image_count)

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
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingConfig":
        """Deserialize the configuration."""
        return cls(
            model_name=payload.get("model_name", "PatchCore"),
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
            dinomaly_encoder=payload.get("dinomaly_encoder", "vit_base_patch14_reg4_dinov2"),
            dinomaly_decoder_depth=int(payload.get("dinomaly_decoder_depth", 8)),
            dinomaly_bottleneck_dropout=float(payload.get("dinomaly_bottleneck_dropout", 0.2)),
            dinomaly_context_recentering=bool(payload.get("dinomaly_context_recentering", False)),
            supplemental_data_path=payload.get("supplemental_data_path", ""),
            zero_shot_class_name=payload.get("zero_shot_class_name", ""),
        )
