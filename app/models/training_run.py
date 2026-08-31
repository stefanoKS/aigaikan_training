"""Training run metadata models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .prediction_result import PredictionResult


@dataclass(slots=True)
class WorkerMessage:
    """Single worker message decoded from JSON Lines."""

    type: str
    payload: dict[str, Any]


@dataclass(slots=True)
class TrainingRun:
    """Persisted training run information."""

    run_name: str
    run_dir: str
    model_name: str
    device: str
    run_date: str = ""
    training_duration_seconds: float = 0.0
    evaluation_duration_seconds: float = 0.0
    final_checkpoint_path: str = ""
    final_checkpoint_sha256: str = ""
    dataset_manifest_sha256: str = ""
    quality_status: str = ""
    export_status: str = "Not exported"
    aigaikan_compatibility_status: str = "Not validated"
    metrics: dict[str, float | str | None] = field(default_factory=dict)
    predictions: list[PredictionResult] = field(default_factory=list)
