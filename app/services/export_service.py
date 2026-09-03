"""Model export abstraction."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
import json
from math import isfinite
from pathlib import Path
import re
import shutil
from typing import Any, Iterable

from app.core.dataset_manifest import sha256_file
from app.core.model_registry import ModelRegistry, ModelSupportLevel
from app.core.result_parser import ResultParser
from app.core.inspection_region import InspectionRegionProcessor, inspection_region_hash
from app.core.preprocessing_contract import resolved_preprocessing_hash
from app.core.preprocessing_pipeline import PreprocessingPipeline
from app.core.run_artifacts import (
    read_canonical_checkpoint,
    read_persisted_threshold_metadata,
    read_verified_inspection_region,
    read_verified_preprocessing_plan,
)
from app.models.prediction_result import PredictionResult
from app.models.preprocessing_config import LEGACY_PREPROCESSING_CONTRACT_VERSION
from app.models.training_config import TrainingConfig
from app.services.anomalib_service import AnomalibService, REQUIRED_ANOMALIB_VERSION
from app.services.threshold_revision_service import ThresholdRevisionResult, ThresholdRevisionService

DEPLOYMENT_CONTRACT_VERSION = 2
FORMAT_SCORE_TOLERANCES: dict[str, float] = {
    "torch": 1e-4,
    "onnx": 1e-3,
    "openvino": 1e-3,
}
DEFAULT_SCORE_TOLERANCE = FORMAT_SCORE_TOLERANCES["torch"]


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
    validation: dict[str, object] | None = None


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
        deployment_validator: Callable[[Path, str, list[PredictionResult], float, float], dict[str, object]] | None = None,
        score_tolerances: Mapping[str, float] | None = None,
        model_registry: ModelRegistry | None = None,
    ) -> None:
        self.anomalib_service = anomalib_service or AnomalibService()
        self.model_registry = model_registry or ModelRegistry()
        self._has_custom_deployment_validator = deployment_validator is not None
        self._deployment_validator = deployment_validator or self._validate_deployment
        self.score_tolerances = dict(FORMAT_SCORE_TOLERANCES)
        if score_tolerances is not None:
            unknown_formats = set(score_tolerances).difference(self.score_tolerances)
            if unknown_formats:
                raise ValueError(f"Unknown export score tolerance formats: {', '.join(sorted(unknown_formats))}")
            self.score_tolerances.update(score_tolerances)
        if any(not isfinite(tolerance) or tolerance < 0 for tolerance in self.score_tolerances.values()):
            raise ValueError("Export score tolerances must be finite non-negative values.")

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
        definition = self.model_registry.get(config.model_name)
        if not definition.supports_export or definition.support_level is not ModelSupportLevel.TORCH_EXPORT_VALIDATED:
            raise ValueError(
                f"{definition.display_name} export is unavailable until an Anomalib export/reload/parity smoke test passes."
            )
        checkpoint_path = read_canonical_checkpoint(run_directory).path
        active_revision = ThresholdRevisionService.read_active_revision(run_directory)
        threshold_metadata = self._effective_threshold_metadata(
            read_persisted_threshold_metadata(run_directory),
            active_revision,
        )
        inspection_region = read_verified_inspection_region(run_directory)
        preprocessing_plan = read_verified_preprocessing_plan(run_directory)
        preprocessing_pipeline = (
            PreprocessingPipeline(inspection_region, preprocessing_plan) if preprocessing_plan is not None else None
        )
        inspection_processor = InspectionRegionProcessor(inspection_region) if preprocessing_pipeline is None else None
        decision_threshold = float(threshold_metadata["threshold_value"])
        final_test_predictions = self._load_final_test_predictions(
            run_directory,
            active_revision.predictions_path if active_revision is not None else None,
        )
        export_directory = export_directory.expanduser().resolve()
        package_directory = export_directory / self.package_directory_name(config.model_name, run_directory.name)
        package_directory.mkdir(parents=True, exist_ok=True)
        packaged_checkpoint_path, included_artifacts = self._copy_run_artifacts(
            run_directory,
            package_directory,
            checkpoint_path,
            active_revision,
        )
        components = (
            self.anomalib_service.create_inference_components(config, package_directory, preprocessing_plan)
            if preprocessing_plan is not None
            else self.anomalib_service.create_inference_components(config, package_directory)
        )
        exported: list[ExportResult] = []
        failures: dict[str, str] = {}

        for export_format in selected_formats:
            try:
                exported_path = components["engine"].export(
                    model=components["model"],
                    export_type=export_format.value,
                    export_root=package_directory,
                    model_file_name=self.model_file_name(config.model_name, run_directory.name, export_format),
                    input_size=None,
                    ckpt_path=checkpoint_path,
                )
                if exported_path is None:
                    raise RuntimeError("Anomalib did not return an exported model path.")
                result = self.verify_export(Path(exported_path), export_format.value)
                validation = (
                    self._validate_deployment(
                        result.exported_path,
                        result.export_format,
                        final_test_predictions,
                        decision_threshold,
                        self.score_tolerances[result.export_format],
                        inspection_processor,
                        preprocessing_pipeline,
                    )
                    if not self._has_custom_deployment_validator
                    else self._deployment_validator(
                        result.exported_path,
                        result.export_format,
                        final_test_predictions,
                        decision_threshold,
                        self.score_tolerances[result.export_format],
                    )
                )
                result.validation = validation
                result.validation_report = self._write_validation_report(
                    result,
                    validation,
                    threshold_metadata,
                )
                exported.append(result)
            except Exception as exc:
                failures[export_format.value] = str(exc)

        self._write_package_manifest(
            package_directory,
            canonical_checkpoint_path=packaged_checkpoint_path,
            config=config,
            final_test_predictions=final_test_predictions,
            exported=exported,
            failures=failures,
            included_artifacts=included_artifacts,
            threshold_metadata=threshold_metadata,
            score_tolerances=self.score_tolerances,
            inspection_preprocessing={
                "roi_contract_version": inspection_region.roi_contract_version,
                "type": inspection_region.region_type,
                "metadata_file": "inspection_region.json",
                "metadata_sha256": inspection_region_hash(inspection_region),
                "source_size": [inspection_region.source_width, inspection_region.source_height],
                "rectified_size": list(inspection_region.rectified_size()),
            },
            preprocessing_contract=(
                {
                    "preprocessing_contract_version": preprocessing_plan.preprocessing_contract_version,
                    "metadata_file": "preprocessing_plan.json",
                    "metadata_sha256": resolved_preprocessing_hash(preprocessing_plan),
                    "model_id": preprocessing_plan.model_id,
                    "model_input_size": list(preprocessing_plan.model_input_size),
                    "score_aggregation": preprocessing_plan.score_aggregation.value,
                    "tiled": preprocessing_plan.tiled,
                }
                if preprocessing_plan is not None
                else {"legacy": True}
            ),
        )
        return ModelExportReport(exported=exported, failures=failures, package_directory=package_directory)

    @staticmethod
    def package_directory_name(model_name: str, run_name: str) -> str:
        """Return a stable package folder name for one validated export operation."""
        return f"{ExportService._slugify(model_name)}_{ExportService._slugify(run_name)}_deployment"

    @staticmethod
    def _copy_run_artifacts(
        run_directory: Path,
        package_directory: Path,
        canonical_checkpoint_path: Path,
        active_revision: ThresholdRevisionResult | None = None,
    ) -> tuple[Path, dict[str, str]]:
        """Copy all immutable records required to audit deployment decisions."""
        required_names = (
            "config.json",
            "environment.json",
            "dataset_manifest.json",
            "run_manifest.json",
            "results.json",
            "predictions.csv",
            "inspection_region.json",
        )
        optional_names = ("calibration_manifest.json", "final_test_manifest.json", "preprocessing_plan.json")
        copied: dict[str, str] = {}
        for name in (*required_names, *optional_names):
            source_path = run_directory / name
            if name in optional_names and not source_path.is_file():
                continue
            if not source_path.is_file() or source_path.stat().st_size == 0:
                raise FileNotFoundError(f"Required run artifact is missing or empty: {source_path}")
            target_path = package_directory / name
            shutil.copy2(source_path, target_path)
            copied[name] = sha256_file(target_path)
        packaged_checkpoint_path = package_directory / "canonical_checkpoint.ckpt"
        shutil.copy2(canonical_checkpoint_path, packaged_checkpoint_path)
        copied[packaged_checkpoint_path.name] = sha256_file(packaged_checkpoint_path)

        run_manifest_path = package_directory / "run_manifest.json"
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        checkpoint = run_manifest.get("canonical_checkpoint")
        if not isinstance(checkpoint, dict):
            raise ValueError("Copied run manifest does not contain a canonical checkpoint record.")
        checkpoint["path"] = packaged_checkpoint_path.name
        checkpoint["sha256"] = copied[packaged_checkpoint_path.name]
        run_manifest_path.write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        copied[run_manifest_path.name] = sha256_file(run_manifest_path)
        if active_revision is not None:
            revision_directory = package_directory / "threshold_revisions"
            revision_directory.mkdir(exist_ok=True)
            for source_path in (
                run_directory / "active_threshold_revision.json",
                active_revision.revision_path,
                active_revision.predictions_path,
            ):
                relative_path = source_path.relative_to(run_directory)
                target_path = package_directory / relative_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target_path)
                copied[relative_path.as_posix()] = sha256_file(target_path)
        return packaged_checkpoint_path, copied

    @staticmethod
    def _write_package_manifest(
        package_directory: Path,
        *,
        canonical_checkpoint_path: Path,
        config: TrainingConfig,
        final_test_predictions: list[PredictionResult],
        exported: list[ExportResult],
        failures: dict[str, str],
        included_artifacts: dict[str, str],
        threshold_metadata: dict[str, object],
        score_tolerances: Mapping[str, float],
        inspection_preprocessing: Mapping[str, object],
        preprocessing_contract: Mapping[str, object],
    ) -> Path:
        """Record deployment package provenance and every verified file digest."""
        payload = {
            "deployment_contract_version": DEPLOYMENT_CONTRACT_VERSION,
            "anomalib_version": REQUIRED_ANOMALIB_VERSION,
            "model": {
                "id": config.model_name,
                "profile": config.model_profile(),
            },
            "canonical_checkpoint": canonical_checkpoint_path.name,
            "canonical_checkpoint_sha256": sha256_file(canonical_checkpoint_path),
            "final_test_prediction_count": len(final_test_predictions),
            "threshold_metadata": threshold_metadata,
            "inspection_preprocessing": dict(inspection_preprocessing),
            "preprocessing_contract": dict(preprocessing_contract),
            "format_score_tolerances": dict(score_tolerances),
            "included_run_artifacts": included_artifacts,
            "exports": [
                {
                    "format": result.export_format,
                    "path": result.exported_path.name,
                    "sha256": result.sha256,
                    "validation_report": result.validation_report.name if result.validation_report else None,
                    "validation": result.validation,
                }
                for result in exported
            ],
            "failures": failures,
        }
        manifest_path = package_directory / "deployment_manifest.json"
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return manifest_path

    @staticmethod
    def _load_final_test_predictions(
        run_directory: Path,
        revision_predictions_path: Path | None = None,
    ) -> list[PredictionResult]:
        results_path = revision_predictions_path or run_directory / "results.json"
        if not results_path.is_file():
            raise FileNotFoundError(f"Final-test results not found: {results_path}")
        parser = ResultParser()
        predictions = (
            parser.read_predictions_csv(results_path)
            if revision_predictions_path is not None
            else parser.read_training_run(results_path).predictions
        )
        final_test_predictions = [
            prediction
            for prediction in predictions
            if prediction.dataset_role in {"final_test_ok", "final_test_ng"}
        ]
        if not final_test_predictions:
            raise ValueError("The run has no persisted final-test predictions for deployment parity validation.")
        return final_test_predictions

    @staticmethod
    def _effective_threshold_metadata(
        threshold_metadata: Mapping[str, object],
        active_revision: ThresholdRevisionResult | None,
    ) -> dict[str, object]:
        """Use an active immutable revision as the deployment decision contract when selected."""
        effective = dict(threshold_metadata)
        if active_revision is None:
            return effective
        image_operating_point = active_revision.image_operating_point.to_dict()
        effective.update(
            {
                "threshold_value": image_operating_point["threshold"],
                "threshold_raw": image_operating_point["threshold"],
                "threshold_deployed": image_operating_point["threshold"],
                "score_semantic": image_operating_point["score_semantic"],
                "decision_comparator": image_operating_point["comparator"],
                "pixel_operating_point": active_revision.pixel_operating_point.to_dict(),
                "threshold_revision": active_revision.revision_path.stem,
            }
        )
        return effective

    @staticmethod
    def _write_validation_report(
        result: ExportResult,
        validation: dict[str, object],
        threshold_metadata: dict[str, object],
    ) -> Path:
        report_path = result.exported_path.with_suffix(f"{result.exported_path.suffix}.validation.json")
        report_path.write_text(
            json.dumps(
                {
                    "deployment_contract_version": DEPLOYMENT_CONTRACT_VERSION,
                    "artifact": str(result.exported_path),
                    "artifact_sha256": result.sha256,
                    "format": result.export_format,
                    "decision_threshold": threshold_metadata["threshold_value"],
                    "threshold_metadata": threshold_metadata,
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
        score_tolerance: float = DEFAULT_SCORE_TOLERANCE,
        inspection_processor: InspectionRegionProcessor | None = None,
        preprocessing_pipeline: PreprocessingPipeline | None = None,
    ) -> dict[str, object]:
        """Reload an export and require numerical and threshold-decision final-test parity."""
        if export_format == ModelExportFormat.TORCH.value:
            from anomalib.deploy import TorchInferencer

            inferencer = TorchInferencer(path=exported_path, device="cpu")
        elif export_format in {ModelExportFormat.OPENVINO.value, ModelExportFormat.ONNX.value}:
            from anomalib.deploy import OpenVINOInferencer

            inferencer = OpenVINOInferencer(path=exported_path, device="CPU")
        else:
            raise ValueError(f"No deployment inferencer is configured for {export_format}.")
        decision_mismatches: list[str] = []
        score_mismatches: list[str] = []
        score_deltas: list[float] = []
        for expected in expected_predictions:
            if preprocessing_pipeline is not None:
                deployed_predictions = [
                    inferencer.predict(prepared.image_rgb)
                    for prepared in preprocessing_pipeline.prepare_path(Path(expected.source_path))
                ]
                deployed_maps = [
                    ExportService._deployment_anomaly_map(prediction)
                    for prediction in deployed_predictions
                ]
                reconstructed = preprocessing_pipeline.reconstruct_anomaly_maps(deployed_maps)
                score = preprocessing_pipeline.score_from_reconstructed_map(reconstructed)
            else:
                deployment_input: str | Any = expected.source_path
                if inspection_processor is not None:
                    deployment_input = inspection_processor.apply_path(Path(expected.source_path))
                score = ExportService._deployment_score(inferencer.predict(deployment_input))
            expected_score = float(expected.anomaly_score)
            if not isfinite(expected_score):
                raise ValueError(f"Persisted checkpoint score must be finite: {expected.source_path}")
            score_delta = abs(score - expected_score)
            score_deltas.append(score_delta)
            predicted_label = "NG" if score >= threshold else "OK"
            if predicted_label != expected.predicted_label.upper():
                decision_mismatches.append(expected.source_path)
            if score_delta > score_tolerance:
                score_mismatches.append(expected.source_path)
        if decision_mismatches:
            raise RuntimeError(
                "Deployment decision parity failed for "
                f"{len(decision_mismatches)} of {len(expected_predictions)} final-test images."
            )
        if score_mismatches:
            raise RuntimeError(
                "Deployment score parity failed for "
                f"{len(score_mismatches)} of {len(expected_predictions)} final-test images; "
                f"tolerance is {score_tolerance}."
            )
        return {
            "status": "PASS",
            "tested_images": len(expected_predictions),
            "decision_parity": 1.0,
            "score_tolerance": score_tolerance,
            "maximum_score_delta": max(score_deltas, default=0.0),
            "mean_score_delta": sum(score_deltas) / len(score_deltas) if score_deltas else 0.0,
        }

    @staticmethod
    def _deployment_anomaly_map(prediction: Any) -> Any:
        """Extract the required anomaly map used by preprocessing-v2 external score aggregation."""
        value = prediction.get("anomaly_map") if isinstance(prediction, dict) else getattr(prediction, "anomaly_map", None)
        if value is None:
            raise ValueError("Deployment prediction did not contain an anomaly map required by preprocessing v2.")
        return value

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

