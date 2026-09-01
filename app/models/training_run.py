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
    calibration_manifest_sha256: str = ""
    final_test_manifest_sha256: str = ""
    evaluation_revision_id: str = ""
    model_variant: str = ""
    encoder_family: str = ""
    threshold_metadata: dict[str, Any] = field(default_factory=dict)
    mean_inference_latency_ms: float | None = None
    p95_inference_latency_ms: float | None = None
    peak_gpu_memory_mb: float | None = None
    quality_status: str = ""
    export_status: str = "Not exported"
    aigaikan_compatibility_status: str = "Not validated"
    metrics: dict[str, float | str | None] = field(default_factory=dict)
    predictions: list[PredictionResult] = field(default_factory=list)
