"""Training configuration models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
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
    image_width: int = 256
    image_height: int = 256
    batch_size: int = 8
    max_epochs: int = 1
    validation_every_n_epochs: int = 1
    gradient_clip_val: float = 0.0
    accumulate_grad_batches: int = 1
    random_seed: int = 42
    coreset_sampling_ratio: float = 0.1
    num_neighbors: int = 9
    num_workers: int = 0
    output_dir: str = ""
    backbone: str = "wide_resnet50_2"
    layers: tuple[str, ...] = ("layer2", "layer3")
    dinomaly_encoder: str = "vit_base_patch16_dinov3.lvd1689m"
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
        if self.dinomaly_decoder_depth <= 1:
            raise ValueError("Dinomaly decoder depth must be greater than one")
        if not 0 <= self.dinomaly_bottleneck_dropout < 1:
            raise ValueError("Dinomaly bottleneck dropout must be between 0 and 1")
        if self.model_name.lower() == "dinomaly" and (
            self.image_width % 16 != 0 or self.image_height % 16 != 0
        ):
            raise ValueError("Dinomaly image dimensions must be divisible by 16")

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
            image_width=int(payload.get("image_width", 256)),
            image_height=int(payload.get("image_height", 256)),
            batch_size=int(payload.get("batch_size", 8)),
            max_epochs=int(payload.get("max_epochs", 1)),
            validation_every_n_epochs=int(payload.get("validation_every_n_epochs", 1)),
            gradient_clip_val=float(payload.get("gradient_clip_val", 0.0)),
            accumulate_grad_batches=int(payload.get("accumulate_grad_batches", 1)),
            random_seed=int(payload.get("random_seed", 42)),
            coreset_sampling_ratio=float(payload.get("coreset_sampling_ratio", 0.1)),
            num_neighbors=int(payload.get("num_neighbors", 9)),
            num_workers=int(payload.get("num_workers", 0)),
            output_dir=payload.get("output_dir", ""),
            backbone=payload.get("backbone", "wide_resnet50_2"),
            layers=tuple(payload.get("layers", ["layer2", "layer3"])),
            dinomaly_encoder=payload.get("dinomaly_encoder", "vit_base_patch16_dinov3.lvd1689m"),
            dinomaly_decoder_depth=int(payload.get("dinomaly_decoder_depth", 8)),
            dinomaly_bottleneck_dropout=float(payload.get("dinomaly_bottleneck_dropout", 0.2)),
            dinomaly_context_recentering=bool(payload.get("dinomaly_context_recentering", False)),
            supplemental_data_path=payload.get("supplemental_data_path", ""),
            zero_shot_class_name=payload.get("zero_shot_class_name", ""),
        )
