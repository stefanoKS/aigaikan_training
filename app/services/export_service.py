"""Model export abstraction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import json
from pathlib import Path
import re
from typing import Iterable

from app.models.training_config import TrainingConfig
from app.services.anomalib_service import AnomalibService


class ModelExportFormat(StrEnum):
    """Model formats supported by Anomalib's deployment API."""

    OPENVINO = "openvino"
    ONNX = "onnx"
    TORCH = "torch"


@dataclass(slots=True)
class ExportResult:
    """Result of a model export operation."""

    exported_path: Path
    export_format: str


@dataclass(slots=True)
class ModelExportReport:
    """Collected results from exporting one trained model to multiple formats."""

    exported: list[ExportResult]
    failures: dict[str, str]


class ExportService:
    """Abstraction for model export."""

    def __init__(self, anomalib_service: AnomalibService | None = None) -> None:
        self.anomalib_service = anomalib_service or AnomalibService()

    def export_model(
        self,
        run_directory: Path,
        export_directory: Path,
        export_formats: Iterable[ModelExportFormat],
    ) -> ModelExportReport:
        """Export a completed run using its saved configuration and checkpoint."""
        run_directory = run_directory.expanduser().resolve()
        selected_formats = tuple(dict.fromkeys(ModelExportFormat(export_format) for export_format in export_formats))
        if not selected_formats:
            raise ValueError("Select at least one model export format.")

        config = self._load_training_config(run_directory)
        checkpoint_path = self._find_checkpoint(run_directory)
        export_directory = export_directory.expanduser().resolve()
        export_directory.mkdir(parents=True, exist_ok=True)
        components = self.anomalib_service.create_inference_components(config, export_directory)
        exported: list[ExportResult] = []
        failures: dict[str, str] = {}

        for export_format in selected_formats:
            try:
                exported_path = components["engine"].export(
                    model=components["model"],
                    export_type=export_format.value,
                    export_root=export_directory,
                    model_file_name=self.model_file_name(config.model_name, run_directory.name, export_format),
                    input_size=(config.image_height, config.image_width),
                    ckpt_path=checkpoint_path,
                )
                if exported_path is None:
                    raise RuntimeError("Anomalib did not return an exported model path.")
                exported.append(self.verify_export(Path(exported_path), export_format.value))
            except Exception as exc:
                failures[export_format.value] = str(exc)

        return ModelExportReport(exported=exported, failures=failures)

    def verify_export(self, path: Path, export_format: str | None = None) -> ExportResult:
        """Verify the export exists before claiming success."""
        if not path.exists():
            raise FileNotFoundError(f"Exported model not found: {path}")
        return ExportResult(exported_path=path, export_format=export_format or path.suffix.lstrip("."))

    @staticmethod
    def model_file_name(model_name: str, run_name: str, export_format: ModelExportFormat) -> str:
        """Return a stable, file-system-safe name for a model export."""
        model_slug = ExportService._slugify(model_name)
        run_slug = ExportService._slugify(run_name)
        model_suffix = f"_{model_slug}"
        if run_slug.endswith(model_suffix):
            run_slug = run_slug[: -len(model_suffix)]
        return f"{model_slug}_{run_slug}_{export_format.value}"

    @staticmethod
    def _load_training_config(run_directory: Path) -> TrainingConfig:
        config_path = run_directory / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Training configuration not found: {config_path}")
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Training configuration must be a JSON object: {config_path}")
        return TrainingConfig.from_dict(payload)

    @staticmethod
    def _find_checkpoint(run_directory: Path) -> Path:
        checkpoints = list(run_directory.glob("**/weights/lightning/*.ckpt"))
        if not checkpoints:
            raise FileNotFoundError(f"No Lightning checkpoint found in: {run_directory}")
        return max(checkpoints, key=lambda path: path.stat().st_mtime).resolve()

    @staticmethod
    def _slugify(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "model"

