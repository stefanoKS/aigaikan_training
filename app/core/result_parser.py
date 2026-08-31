"""Worker result parsing and export helpers."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.models.prediction_result import PredictionResult
from app.models.training_run import TrainingRun, WorkerMessage


KNOWN_METRIC_ALIASES = {
    "image_auroc": "Image AUROC",
    "image_AUROC": "Image AUROC",
    "image_f1score": "Image F1",
    "image_F1Score": "Image F1",
    "image_f1": "Image F1",
    "f1_score": "Image F1",
    "precision": "Precision",
    "recall": "Recall",
    "threshold": "Threshold",
}


@dataclass(slots=True)
class ParsedWorkerState:
    """Aggregated worker state."""

    stages: list[str] = field(default_factory=list)
    metrics: dict[str, float | str | None] = field(default_factory=dict)
    result_images: list[str] = field(default_factory=list)
    logs: list[dict[str, str]] = field(default_factory=list)
    completed_result_dir: str = ""
    error: dict[str, str] | None = None


class ResultParser:
    """Parse JSON Lines worker output."""

    def parse_worker_line(self, line: str) -> WorkerMessage:
        """Parse a single worker JSON line."""
        payload = json.loads(line)
        message_type = payload.pop("type")
        return WorkerMessage(type=message_type, payload=payload)

    def collect(self, lines: list[str]) -> ParsedWorkerState:
        """Collect worker lines into a structured state."""
        state = ParsedWorkerState()
        for line in lines:
            message = self.parse_worker_line(line)
            if message.type == "stage":
                state.stages.append(str(message.payload.get("name", "")))
            elif message.type == "metric":
                name = str(message.payload.get("name", ""))
                state.metrics[self.normalize_metric_name(name)] = message.payload.get("value")
            elif message.type == "result_image":
                state.result_images.append(str(message.payload.get("path", "")))
            elif message.type == "log":
                state.logs.append(
                    {
                        "level": str(message.payload.get("level", "info")),
                        "message": str(message.payload.get("message", "")),
                    }
                )
            elif message.type == "completed":
                state.completed_result_dir = str(message.payload.get("result_dir", ""))
            elif message.type == "error":
                state.error = {
                    "message": str(message.payload.get("message", "Training failed")),
                    "details": str(message.payload.get("details", "")),
                }
        return state

    def normalize_metric_name(self, name: str) -> str:
        """Normalize a metric key to a UI-facing name."""
        return KNOWN_METRIC_ALIASES.get(name, name.replace("_", " ").title())

    def read_predictions_csv(self, path: Path) -> list[PredictionResult]:
        """Read prediction rows from a CSV export."""
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            return [
                PredictionResult(
                    source_path=row.get("source_path", ""),
                    predicted_label=row.get("predicted_label", ""),
                    ground_truth_label=row.get("ground_truth_label", ""),
                    anomaly_score=float(row.get("anomaly_score", 0.0)),
                    threshold=float(row.get("threshold", 0.0)),
                    original_image=row.get("original_image", ""),
                    anomaly_map=row.get("anomaly_map", ""),
                    overlay_image=row.get("overlay_image", ""),
                )
                for row in reader
            ]

    def write_training_run(self, path: Path, run: TrainingRun) -> Path:
        """Persist a training run summary for the Results page."""
        path.write_text(json.dumps(asdict(run), indent=2), encoding="utf-8")
        return path

    def read_training_run(self, path: Path) -> TrainingRun:
        """Load a persisted training run summary."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        predictions = [
            PredictionResult(
                source_path=str(item.get("source_path", "")),
                predicted_label=str(item.get("predicted_label", "")),
                ground_truth_label=str(item.get("ground_truth_label", "")),
                anomaly_score=float(item.get("anomaly_score", 0.0)),
                threshold=float(item.get("threshold", 0.0)),
                original_image=str(item.get("original_image", "")),
                anomaly_map=str(item.get("anomaly_map", "")),
                overlay_image=str(item.get("overlay_image", "")),
            )
            for item in payload.get("predictions", [])
            if isinstance(item, dict)
        ]
        metrics = payload.get("metrics", {})
        return TrainingRun(
            run_name=str(payload.get("run_name", "")),
            run_dir=str(payload.get("run_dir", "")),
            model_name=str(payload.get("model_name", "")),
            device=str(payload.get("device", "")),
            run_date=str(payload.get("run_date", "")),
            training_duration_seconds=float(payload.get("training_duration_seconds", 0.0)),
            evaluation_duration_seconds=float(payload.get("evaluation_duration_seconds", 0.0)),
            metrics=metrics if isinstance(metrics, dict) else {},
            predictions=predictions,
        )

    def export_predictions_csv(self, path: Path, predictions: list[PredictionResult]) -> Path:
        """Export prediction rows to a CSV file."""
        fieldnames = [
            "source_path",
            "predicted_label",
            "ground_truth_label",
            "anomaly_score",
            "threshold",
            "original_image",
            "anomaly_map",
            "overlay_image",
            "classification_bucket",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for prediction in predictions:
                writer.writerow(prediction.to_dict())
        return path

