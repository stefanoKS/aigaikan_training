"""Model export abstraction."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
import json
from math import isfinite
from pathlib import Path
import re
import shutil
from typing import Any, Iterable

from app.core.dataset_manifest import sha256_file
from app.core.result_parser import ResultParser
from app.core.run_artifacts import read_canonical_checkpoint, read_persisted_threshold
from app.models.prediction_result import PredictionResult
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
    sha256: str
    validation_report: Path | None = None


@dataclass(slots=True)
class ModelExportReport:
    """Collected results from exporting one trained model to multiple formats."""

    exported: list[ExportResult]
    failures: dict[str, str]
    package_directory: Path | None = None


class ExportService:
    """Abstraction for model export."""

    def __init__(
        self,
        anomalib_service: AnomalibService | None = None,
        deployment_validator: Callable[[Path, str, list[PredictionResult], float], dict[str, object]] | None = None,
    ) -> None:
        self.anomalib_service = anomalib_service or AnomalibService()
        self._deployment_validator = deployment_validator or self._validate_deployment

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
        checkpoint_path = read_canonical_checkpoint(run_directory).path
        decision_threshold = read_persisted_threshold(run_directory)
        final_test_predictions = self._load_final_test_predictions(run_directory)
        export_directory = export_directory.expanduser().resolve()
        package_directory = export_directory / self.package_directory_name(config.model_name, run_directory.name)
        package_directory.mkdir(parents=True, exist_ok=True)
        included_artifacts = self._copy_run_artifacts(run_directory, package_directory)
        components = self.anomalib_service.create_inference_components(config, package_directory)
        exported: list[ExportResult] = []
        failures: dict[str, str] = {}

        for export_format in selected_formats:
            try:
                exported_path = components["engine"].export(
                    model=components["model"],
                    export_type=export_format.value,
                    export_root=package_directory,
                    model_file_name=self.model_file_name(config.model_name, run_directory.name, export_format),
                    input_size=(config.image_height, config.image_width),
                    ckpt_path=checkpoint_path,
                )
                if exported_path is None:
                    raise RuntimeError("Anomalib did not return an exported model path.")
                result = self.verify_export(Path(exported_path), export_format.value)
                validation = self._deployment_validator(
                    result.exported_path,
                    result.export_format,
                    final_test_predictions,
                    decision_threshold,
                )
                result.validation_report = self._write_validation_report(
                    result,
                    validation,
                    decision_threshold,
                )
                exported.append(result)
            except Exception as exc:
                failures[export_format.value] = str(exc)

        self._write_package_manifest(
            package_directory,
            canonical_checkpoint_path=checkpoint_path,
            final_test_predictions=final_test_predictions,
            exported=exported,
            failures=failures,
            included_artifacts=included_artifacts,
        )
        return ModelExportReport(exported=exported, failures=failures, package_directory=package_directory)

    @staticmethod
    def package_directory_name(model_name: str, run_name: str) -> str:
        """Return a stable package folder name for one validated export operation."""
        return f"{ExportService._slugify(model_name)}_{ExportService._slugify(run_name)}_deployment"

    @staticmethod
    def _copy_run_artifacts(run_directory: Path, package_directory: Path) -> dict[str, str]:
        """Copy all immutable records required to audit deployment decisions."""
        required_names = (
            "config.json",
            "environment.json",
            "dataset_manifest.json",
            "run_manifest.json",
            "results.json",
            "predictions.csv",
        )
        copied: dict[str, str] = {}
        for name in required_names:
            source_path = run_directory / name
            if not source_path.is_file() or source_path.stat().st_size == 0:
                raise FileNotFoundError(f"Required run artifact is missing or empty: {source_path}")
            target_path = package_directory / name
            shutil.copy2(source_path, target_path)
            copied[name] = sha256_file(target_path)
        return copied

    @staticmethod
    def _write_package_manifest(
        package_directory: Path,
        *,
        canonical_checkpoint_path: Path,
        final_test_predictions: list[PredictionResult],
        exported: list[ExportResult],
        failures: dict[str, str],
        included_artifacts: dict[str, str],
    ) -> Path:
        """Record deployment package provenance and every verified file digest."""
        payload = {
            "canonical_checkpoint": str(canonical_checkpoint_path),
            "canonical_checkpoint_sha256": sha256_file(canonical_checkpoint_path),
            "final_test_prediction_count": len(final_test_predictions),
            "included_run_artifacts": included_artifacts,
            "exports": [
                {
                    "format": result.export_format,
                    "path": result.exported_path.name,
                    "sha256": result.sha256,
                    "validation_report": result.validation_report.name if result.validation_report else None,
                }
                for result in exported
            ],
            "failures": failures,
        }
        manifest_path = package_directory / "deployment_manifest.json"
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return manifest_path

    @staticmethod
    def _load_final_test_predictions(run_directory: Path) -> list[PredictionResult]:
        results_path = run_directory / "results.json"
        if not results_path.is_file():
            raise FileNotFoundError(f"Final-test results not found: {results_path}")
        predictions = ResultParser().read_training_run(results_path).predictions
        final_test_predictions = [
            prediction
            for prediction in predictions
            if prediction.dataset_role in {"final_test_ok", "final_test_ng"}
        ]
        if not final_test_predictions:
            raise ValueError("The run has no persisted final-test predictions for deployment parity validation.")
        return final_test_predictions

    @staticmethod
    def _write_validation_report(
        result: ExportResult,
        validation: dict[str, object],
        threshold: float,
    ) -> Path:
        report_path = result.exported_path.with_suffix(f"{result.exported_path.suffix}.validation.json")
        report_path.write_text(
            json.dumps(
                {
                    "artifact": str(result.exported_path),
                    "artifact_sha256": result.sha256,
                    "format": result.export_format,
                    "decision_threshold": threshold,
                    "validation": validation,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return report_path

    @staticmethod
    def _validate_deployment(
        exported_path: Path,
        export_format: str,
        expected_predictions: list[PredictionResult],
        threshold: float,
    ) -> dict[str, object]:
        """Reload an export and require exact final-test OK/NG decision parity."""
        if export_format == ModelExportFormat.TORCH.value:
            from anomalib.deploy import TorchInferencer

            inferencer = TorchInferencer(path=exported_path, device="cpu")
        elif export_format in {ModelExportFormat.OPENVINO.value, ModelExportFormat.ONNX.value}:
            from anomalib.deploy import OpenVINOInferencer

            inferencer = OpenVINOInferencer(path=exported_path, device="CPU")
        else:
            raise ValueError(f"No deployment inferencer is configured for {export_format}.")
        mismatches: list[str] = []
        for expected in expected_predictions:
            score = ExportService._deployment_score(inferencer.predict(expected.source_path))
            predicted_label = "NG" if score >= threshold else "OK"
            if predicted_label != expected.predicted_label.upper():
                mismatches.append(expected.source_path)
        if mismatches:
            raise RuntimeError(
                f"Deployment decision parity failed for {len(mismatches)} of {len(expected_predictions)} final-test images."
            )
        return {"status": "PASS", "tested_images": len(expected_predictions), "decision_parity": 1.0}

    @staticmethod
    def _deployment_score(prediction: Any) -> float:
        """Extract one finite image score from an Anomalib deployment prediction."""
        value = prediction.get("pred_score") if isinstance(prediction, dict) else getattr(prediction, "pred_score", None)
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "numel") and value.numel() != 1:
            raise ValueError("Deployment prediction must contain exactly one image score.")
        if hasattr(value, "item"):
            value = value.item()
        try:
            score = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Deployment prediction did not contain a numeric image score.") from exc
        if not isfinite(score):
            raise ValueError("Deployment prediction score must be finite.")
        return score

    def verify_export(self, path: Path, export_format: str | None = None) -> ExportResult:
        """Verify a nonempty, format-complete deployable artifact before reporting success."""
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Exported model not found: {path}")
        resolved_format = export_format or path.suffix.lstrip(".")
        expected_suffixes = {
            ModelExportFormat.OPENVINO.value: ".xml",
            ModelExportFormat.ONNX.value: ".onnx",
            ModelExportFormat.TORCH.value: ".pt",
        }
        expected_suffix = expected_suffixes.get(resolved_format)
        if expected_suffix and path.suffix.lower() != expected_suffix:
            raise ValueError(f"Expected a {expected_suffix} artifact for {resolved_format}, received: {path.name}")
        if resolved_format == ModelExportFormat.OPENVINO.value:
            weights_path = path.with_suffix(".bin")
            if not weights_path.is_file() or weights_path.stat().st_size == 0:
                raise FileNotFoundError(f"OpenVINO weights file not found: {weights_path}")
        return ExportResult(exported_path=path, export_format=resolved_format, sha256=sha256_file(path))

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
    def _slugify(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "model"

