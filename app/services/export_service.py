"""Model export abstraction."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import StrEnum
import hashlib
import json
from math import isfinite
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from typing import Any, Iterable

import numpy as np

from app.core.dataset_manifest import sha256_file
from app.core.decision_policy import DecisionPolicy, decision_policy_hash, read_decision_policy, write_decision_policy
from app.core.decision_score import require_matching_score_semantic, resolve_decision_score
from app.core.deployment_package import (
    DEPLOYMENT_CONTRACT_VERSION as TWO_FILE_DEPLOYMENT_CONTRACT_VERSION,
    DEPLOYMENT_METADATA_FILENAME,
    DEPLOYMENT_MODEL_FILENAME,
    DeploymentPackage,
    read_deployment_json,
    superadd_memory_bank_metadata,
    validate_deployment_json,
)
from app.core.superadd_deployment import SuperADDDeploymentAdapter, SuperADDDeploymentInferencer
from app.core.model_registry import ModelRegistry, ModelSupportLevel
from app.core.result_parser import ResultParser
from app.core.inspection_region import InspectionRegionProcessor, inspection_region_hash
from app.core.preprocessing_contract import image_preprocessing_hash, resolved_preprocessing_hash, write_image_preprocessing_config
from app.core.preprocessing_pipeline import PreprocessingPipeline
from app.core.threshold_contract import PixelThresholdOperatingPoint
from app.core.run_artifacts import (
    read_canonical_checkpoint,
    read_persisted_threshold_metadata,
    read_verified_inspection_region,
    read_verified_preprocessing_plan,
)
from app.models.prediction_result import PredictionResult
from app.models.image_preprocessing import ImagePreprocessingConfig
from app.models.preprocessing_config import LEGACY_PREPROCESSING_CONTRACT_VERSION
from app.models.training_config import TrainingConfig
from app.services.anomalib_service import AnomalibService, REQUIRED_ANOMALIB_VERSION
from app.services.threshold_revision_service import ThresholdRevisionResult, ThresholdRevisionService
from app.version import APP_VERSION

DEPLOYMENT_CONTRACT_VERSION = TWO_FILE_DEPLOYMENT_CONTRACT_VERSION
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
        if selected_formats != (ModelExportFormat.TORCH,):
            raise ValueError("The two-file deployment contract supports only one Torch (.pt) export.")

        config = self._load_training_config(run_directory)
        definition = self.model_registry.get(config.model_name)
        if definition.key != "super_add":
            raise ValueError("Torch deployment export is implemented only for SuperADD.")
        if not definition.supports_export or definition.support_level is not ModelSupportLevel.TORCH_EXPORT_VALIDATED:
            raise ValueError(
                f"{definition.display_name} export is unavailable until an Anomalib export/reload/parity smoke test passes."
            )
        checkpoint_path = read_canonical_checkpoint(run_directory).path
        active_revision = ThresholdRevisionService.read_active_revision(run_directory)
        calibrated_threshold_metadata = read_persisted_threshold_metadata(run_directory)
        threshold_metadata = self._effective_threshold_metadata(
            calibrated_threshold_metadata,
            active_revision,
        )
        inspection_region = read_verified_inspection_region(run_directory)
        preprocessing_plan = read_verified_preprocessing_plan(run_directory)
        preprocessing_pipeline = (
            PreprocessingPipeline(inspection_region, preprocessing_plan) if preprocessing_plan is not None else None
        )
        if preprocessing_plan is None:
            raise ValueError("Two-file deployment export requires a verified resolved preprocessing plan from the completed run.")
        if preprocessing_plan.tiled:
            raise ValueError("SuperADD deployment export prohibits external tiling.")
        final_test_predictions = self._load_final_test_predictions(
            run_directory,
            active_revision.predictions_path if active_revision is not None else None,
        )
        export_directory = export_directory.expanduser().resolve()
        package_directory = export_directory / self.package_directory_name(config.model_name, run_directory.name)
        if package_directory.exists():
            raise FileExistsError(f"Deployment directory already exists: {package_directory}")
        export_directory.mkdir(parents=True, exist_ok=True)
        stage = "create deployment staging area"
        try:
            with TemporaryDirectory(prefix="aigaikan-superadd-deployment-", dir=export_directory) as temporary_directory:
                staging_root = Path(temporary_directory)
                stage = "load completed SuperADD checkpoint"
                components = self.anomalib_service.create_inference_components(config, staging_root, preprocessing_plan)
                staged_package = staging_root / "deployment"
                staged_package.mkdir()
                staged_model = staged_package / DEPLOYMENT_MODEL_FILENAME
                stage = "write SuperADD native-score Torch artifact"
                memory_bank = self._write_superadd_torch_artifact(components["model"], checkpoint_path, staged_model)
                policy = self._build_decision_policy(
                    threshold_metadata,
                    calibrated_threshold_metadata,
                    active_revision,
                    model_sha256=sha256_file(staged_model),
                    preprocessing_plan_sha256=resolved_preprocessing_hash(preprocessing_plan),
                )
                stage = "write pending deployment metadata"
                pending_metadata = self._deployment_metadata(
                    run_directory,
                    config,
                    definition,
                    checkpoint_path,
                    inspection_region,
                    preprocessing_plan,
                    policy,
                    {"status": "PENDING"},
                    superadd_memory_bank=memory_bank,
                )
                self._write_deployment_json(staged_package / DEPLOYMENT_METADATA_FILENAME, pending_metadata)
                stage = "reload local SuperADD artifact and verify final-test parity"
                validation_device = "cuda" if components["device"] == "gpu" else "cpu"
                if config.superadd_precision == "float16" and validation_device != "cuda":
                    raise ValueError("SuperADD FP16 deployment validation requires CUDA.")
                validation = self._validate_two_file_deployment(
                    staged_package,
                    final_test_predictions,
                    policy.threshold,
                    self.score_tolerances[ModelExportFormat.TORCH.value],
                    device=validation_device,
                    trust_newly_created_local_artifact=True,
                )
                if validation.get("status") != "PASS":
                    raise RuntimeError("Deployment export/reload parity validation did not pass.")
                stage = "finalize deployment metadata"
                finalized_metadata = self._deployment_metadata(
                    run_directory,
                    config,
                    definition,
                    checkpoint_path,
                    inspection_region,
                    preprocessing_plan,
                    policy,
                    validation,
                    superadd_memory_bank=memory_bank,
                )
                self._write_deployment_json(staged_package / DEPLOYMENT_METADATA_FILENAME, finalized_metadata)
                validate_deployment_json(finalized_metadata, staged_model)
                stage = "publish validated deployment"
                staged_package.replace(package_directory)
        except Exception as exc:
            raise RuntimeError(f"SuperADD deployment export failed during {stage}: {exc}") from exc
        result = ExportResult(
            exported_path=package_directory / DEPLOYMENT_MODEL_FILENAME,
            export_format=ModelExportFormat.TORCH.value,
            sha256=sha256_file(package_directory / DEPLOYMENT_MODEL_FILENAME),
            validation=validation,
        )
        return ModelExportReport(exported=[result], failures={}, package_directory=package_directory)

    @staticmethod
    def _write_superadd_torch_artifact(model: Any, checkpoint_path: Path, destination: Path) -> dict[str, object]:
        """Serialize the completed SuperADD checkpoint behind the native-score adapter."""
        load_from_checkpoint = getattr(model.__class__, "load_from_checkpoint", None)
        if not callable(load_from_checkpoint):
            raise ValueError("SuperADD model class cannot load the completed checkpoint for deployment.")
        try:
            trained_model = load_from_checkpoint(checkpoint_path, map_location="cpu", weights_only=False)
            adapter = SuperADDDeploymentAdapter(trained_model)
            adapter.eval()
            import torch

            torch.save({"model": adapter}, destination)
            return superadd_memory_bank_metadata(adapter)
        except Exception as exc:
            raise ValueError("Could not serialize the completed SuperADD checkpoint as a native-score Torch artifact.") from exc

    def create_deployment_policy_revision(
        self,
        package_directory: Path,
        destination_directory: Path,
        deployment_ng_score_threshold: float,
        operator_note: str = "",
    ) -> Path:
        """Create a new package revision that reuses a verified Torch artifact and changes only decision policy."""
        package_directory = package_directory.expanduser().resolve()
        manifest = read_deployment_json(package_directory / DEPLOYMENT_METADATA_FILENAME)
        model_path = package_directory / DEPLOYMENT_MODEL_FILENAME
        validate_deployment_json(manifest, model_path)
        if not isfinite(deployment_ng_score_threshold):
            raise ValueError("Deployment NG score threshold must be finite.")
        if "\x00" in operator_note:
            raise ValueError("Operator note must not contain NUL characters.")
        next_revision = self._next_deployment_revision_id(destination_directory, package_directory.name)
        revised_directory = destination_directory.expanduser().resolve() / next_revision
        if revised_directory.exists():
            raise FileExistsError(f"Deployment policy revision already exists: {revised_directory}")
        destination_directory = destination_directory.expanduser().resolve()
        destination_directory.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="aigaikan-two-file-revision-", dir=destination_directory) as temporary_directory:
            staged_directory = Path(temporary_directory) / "deployment"
            staged_directory.mkdir()
            staged_model = staged_directory / DEPLOYMENT_MODEL_FILENAME
            shutil.copy2(model_path, staged_model)
            revised = json.loads(json.dumps(manifest))
            deployment = revised["deployment"]
            decision = revised["decision"]
            if not isinstance(deployment, dict) or not isinstance(decision, dict):
                raise ValueError("deployment.json has invalid deployment or decision metadata.")
            deployment["deployment_id"] = next_revision
            deployment["created_at"] = datetime.now(timezone.utc).isoformat()
            decision.update(
                {
                    "threshold": deployment_ng_score_threshold,
                    "threshold_source": "operator_override",
                    "threshold_revision_id": next_revision,
                    "operator_note": operator_note,
                }
            )
            self._write_deployment_json(staged_directory / DEPLOYMENT_METADATA_FILENAME, revised)
            validate_deployment_json(revised, staged_model)
            staged_directory.replace(revised_directory)
        return revised_directory

    @staticmethod
    def package_directory_name(model_name: str, run_name: str) -> str:
        """Return a stable package folder name for one validated export operation."""
        return f"{ExportService._slugify(model_name)}_{ExportService._slugify(run_name)}_deployment"

    @staticmethod
    def _deployment_metadata(
        run_directory: Path,
        config: TrainingConfig,
        definition: Any,
        checkpoint_path: Path,
        inspection_region: Any,
        preprocessing_plan: Any,
        policy: DecisionPolicy,
        validation: Mapping[str, object],
        *,
        superadd_memory_bank: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Embed all non-tensor state required by the two-file raw-pixel deployment boundary."""
        model_sha256 = policy.model_sha256
        model_profile = config.model_profile()
        algorithm = definition.key
        created_at = datetime.now(timezone.utc).isoformat()
        model_metadata: dict[str, object] = {
            "id": preprocessing_plan.model_id,
            "algorithm": algorithm,
            "anomalib_version": REQUIRED_ANOMALIB_VERSION,
            "torch_version": ExportService._runtime_versions()["torch"],
            "profile": model_profile,
            "anomalib_transform_owner": "model.pt",
            "expected_tensor_layout": "NCHW",
            "expected_precision": config.superadd_precision if config.is_super_add else "float32",
        }
        if config.is_super_add:
            if superadd_memory_bank is None:
                raise ValueError("SuperADD deployment metadata requires trained memory-bank state from the serialized artifact.")
            memory_bank = dict(superadd_memory_bank)
            model_metadata.update(
                {
                    "export_adapter": "superadd_native_v1",
                    "output_contract": {
                        "decision_score": "superadd_native_top_quantile_score_v1",
                        "anomaly_map": "continuous_unthresholded",
                    },
                    "backbone": config.superadd_backbone_name,
                    "layers": model_profile["layers"],
                    "precision": config.superadd_precision,
                    "patch_size": model_profile["patch_size"],
                    "patch_overlap": model_profile["patch_overlap"],
                    "score_quantile": model_profile["score_quantile"],
                    "external_tiling": False,
                    "memory_bank": memory_bank,
                }
            )
        return {
            "deployment_contract_version": DEPLOYMENT_CONTRACT_VERSION,
            "deployment": {
                "deployment_id": f"{ExportService._slugify(run_directory.parent.parent.name)}_{ExportService._slugify(config.model_name)}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
                "created_at": created_at,
                "training_run_id": run_directory.name,
                "project_name": run_directory.parent.parent.name,
                "algorithm": algorithm,
                "model_sha256": model_sha256,
            },
            "input": {
                "dtype": "uint8",
                "range": [0, 255],
                "accepted_layouts": ["HW", "HWC"],
                "canonical_color_order": "RGB",
                "color_input_order": "RGB",
                "mono_conversion": "GRAY_TO_RGB",
            },
            "inspection_region": inspection_region.to_dict(),
            "image_preprocessing": preprocessing_plan.image_preprocessing.to_dict(),
            "model_preprocessing": {
                "resolved_plan": preprocessing_plan.to_dict(),
                "input_size": list(preprocessing_plan.model_input_size),
                "padding": list(preprocessing_plan.resolved_padding),
                "alignment": list(preprocessing_plan.model_alignment),
                "patch_size": preprocessing_plan.patch_size,
                "external_tiling": preprocessing_plan.tiled,
                "interpolation": inspection_region.interpolation,
                "anomalib_transform_owner": "model.pt",
                "expected_tensor_layout": "NCHW",
                "expected_precision": model_metadata["expected_precision"],
            },
            "model": model_metadata,
            "decision": {
                "score_semantic": policy.score_semantic,
                "threshold": policy.threshold,
                "comparator": policy.comparator,
                "above_or_equal_label": policy.above_or_equal_label,
                "below_label": policy.below_label,
                "higher_is_more_anomalous": True,
                "threshold_source": policy.source,
                "base_calibrated_threshold": policy.base_calibrated_threshold,
                "threshold_revision_id": policy.revision_id,
                "operator_note": policy.operator_note,
                "pixel_operating_point": policy.pixel_operating_point.to_dict(),
            },
            "validation": dict(validation),
        }

    @staticmethod
    def _write_deployment_json(path: Path, metadata: Mapping[str, object]) -> Path:
        """Atomically write the sole non-tensor deployment artifact."""
        ExportService._atomic_write_json(path, dict(metadata))
        return path

    @staticmethod
    def _validate_two_file_deployment(
        package_directory: Path,
        expected_predictions: list[PredictionResult],
        threshold: float,
        score_tolerance: float,
        *,
        device: str = "cpu",
        trust_newly_created_local_artifact: bool = False,
    ) -> dict[str, object]:
        """Reload model.pt from the two-file package and require score, map, and decision parity."""
        from PIL import Image

        metadata = read_deployment_json(package_directory / DEPLOYMENT_METADATA_FILENAME)
        model = metadata.get("model")
        if not isinstance(model, Mapping) or model.get("algorithm") != "super_add":
            raise ValueError("Two-file export parity validation is implemented only for SuperADD.")
        package = DeploymentPackage.load(
            package_directory,
            lambda model_path: SuperADDDeploymentInferencer.load(
                model_path,
                device=device,
                trust_newly_created_local_artifact=trust_newly_created_local_artifact,
            ),
            device=device,
            require_validation=False,
        )
        score_errors: list[float] = []
        map_errors: list[np.ndarray] = []
        decisions = 0
        for expected in expected_predictions:
            source_path = Path(expected.source_path)
            if not source_path.is_file():
                raise FileNotFoundError(f"Deployment parity source image is missing: {source_path}")
            with Image.open(source_path) as image:
                raw_rgb = np.asarray(image.convert("RGB"))
            actual = package.predict(raw_rgb)
            if actual.score_semantic != expected.score_semantic:
                raise RuntimeError("Deployment score semantic does not match the completed-run prediction.")
            expected_score = float(expected.anomaly_score)
            if not isfinite(expected_score):
                raise ValueError("Completed-run SuperADD decision score must be finite.")
            score_errors.append(abs(actual.decision_score - expected_score))
            if actual.threshold != threshold:
                raise RuntimeError("Deployment did not use the saved active decision threshold.")
            expected_map = ExportService._stored_anomaly_map(expected, require_raw_superadd_map=True)
            if expected_map.shape != actual.anomaly_map.shape:
                raise RuntimeError("Deployment anomaly-map shape does not match the completed-run prediction.")
            map_errors.append(np.abs(expected_map.astype(np.float64) - actual.anomaly_map.astype(np.float64)))
            decisions += int(actual.is_ng == (expected.predicted_label.upper() == "NG"))
        if not score_errors or max(score_errors) > score_tolerance:
            raise RuntimeError("Deployment score parity failed for the two-file package.")
        if decisions != len(expected_predictions):
            raise RuntimeError("Deployment decision parity failed for the two-file package.")
        errors = np.concatenate([values.ravel() for values in map_errors])
        if float(errors.max()) > score_tolerance:
            raise RuntimeError("Deployment continuous anomaly-map parity failed for the two-file package.")
        return {
            "status": "PASS",
            "score_tolerance": score_tolerance,
            "map_tolerance": score_tolerance,
            "max_abs_score_error": max(score_errors),
            "mean_abs_map_error": float(errors.mean()),
            "max_abs_map_error": float(errors.max()),
            "decision_match_rate": decisions / len(expected_predictions),
            "number_of_test_images": len(expected_predictions),
            "artifact": DEPLOYMENT_MODEL_FILENAME,
            "decision_threshold": threshold,
        }

    @staticmethod
    def _stored_anomaly_map(prediction: PredictionResult, *, require_raw_superadd_map: bool = False) -> np.ndarray:
        path_value = prediction.raw_anomaly_map if require_raw_superadd_map else (
            prediction.postprocessed_anomaly_map or prediction.continuous_anomaly_map
        )
        if require_raw_superadd_map and not path_value:
            raise ValueError("Completed-run SuperADD raw continuous anomaly map is missing for deployment parity.")
        path = Path(path_value)
        if not path.is_file():
            raise FileNotFoundError(f"Completed-run anomaly map is missing for deployment parity: {path}")
        with np.load(path, allow_pickle=False) as stored:
            if "anomaly_map" not in stored:
                raise ValueError("Completed-run anomaly map artifact does not contain anomaly_map.")
            values = stored["anomaly_map"]
        if values.ndim != 2 or values.size == 0 or not np.isfinite(values).all():
            raise ValueError("Completed-run anomaly map artifact must contain finite two-dimensional values.")
        return np.ascontiguousarray(values)

    @staticmethod
    def _copy_run_artifacts(
        run_directory: Path,
        package_directory: Path,
        canonical_checkpoint_path: Path,
        active_revision: ThresholdRevisionResult | None = None,
        image_preprocessing: ImagePreprocessingConfig = ImagePreprocessingConfig(),
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
        standalone_profile_path = write_image_preprocessing_config(
            package_directory / "preprocessing.json", image_preprocessing
        )
        copied[standalone_profile_path.name] = sha256_file(standalone_profile_path)
        ExportService._copy_preprocessing_reference_runner(package_directory, copied)

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
    def _copy_preprocessing_reference_runner(package_directory: Path, copied: dict[str, str]) -> None:
        """Copy the canonical runner source instead of maintaining a bundle-only implementation."""
        workspace_root = Path(__file__).resolve().parents[2]
        runner_directory = package_directory / "reference_runner"
        for relative_path in (
            Path("app/__init__.py"),
            Path("app/version.py"),
            Path("app/core/__init__.py"),
            Path("app/core/decision_policy.py"),
            Path("app/core/decision_score.py"),
            Path("app/core/deployment_reference.py"),
            Path("app/core/image_preprocessor.py"),
            Path("app/core/inference_timing.py"),
            Path("app/core/inspection_region.py"),
            Path("app/core/prediction_contract.py"),
            Path("app/core/prediction_artifacts.py"),
            Path("app/core/preprocessing_contract.py"),
            Path("app/core/preprocessing_pipeline.py"),
            Path("app/core/preprocessing_reference.py"),
            Path("app/core/threshold_contract.py"),
            Path("app/models/__init__.py"),
            Path("app/models/image_preprocessing.py"),
            Path("app/models/dataset_config.py"),
            Path("app/models/inspection_region.py"),
            Path("app/models/preprocessing_config.py"),
            Path("scripts/preprocessing_reference_runner.py"),
            Path("scripts/deployment_reference_inference.py"),
            Path("scripts/benchmark_deployment_reference.py"),
            Path("app/resources/preprocessing_golden_vectors.json"),
        ):
            source_path = workspace_root / relative_path
            target_relative = (
                Path("run_preprocessing_reference.py")
                if relative_path == Path("scripts/preprocessing_reference_runner.py")
                else Path("deployment_reference_inference.py")
                if relative_path == Path("scripts/deployment_reference_inference.py")
                else Path("benchmark_deployment_reference.py")
                if relative_path == Path("scripts/benchmark_deployment_reference.py")
                else Path("golden_vectors.json")
                if relative_path == Path("app/resources/preprocessing_golden_vectors.json")
                else relative_path
            )
            target_path = runner_directory / target_relative
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if relative_path == Path("app/models/__init__.py"):
                target_path.write_text("", encoding="utf-8")
            else:
                shutil.copy2(source_path, target_path)
            copied[(Path("reference_runner") / target_relative).as_posix()] = sha256_file(target_path)

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
        decision_policy: DecisionPolicy,
    ) -> Path:
        """Record deployment package provenance and every verified file digest."""
        payload = {
            "deployment_contract_version": DEPLOYMENT_CONTRACT_VERSION,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "anomalib_version": REQUIRED_ANOMALIB_VERSION,
            "trainer_version": APP_VERSION,
            "model": {
                "id": config.model_name,
                "profile": config.model_profile(),
            },
            "torch_export_type": ModelExportFormat.TORCH.value,
            "runtime_versions": ExportService._runtime_versions(),
            "canonical_checkpoint": canonical_checkpoint_path.name,
            "canonical_checkpoint_sha256": sha256_file(canonical_checkpoint_path),
            "final_test_prediction_count": len(final_test_predictions),
            "threshold_metadata": threshold_metadata,
            "inspection_preprocessing": dict(inspection_preprocessing),
            "preprocessing_contract": dict(preprocessing_contract),
            "decision_policy": {
                "file": "decision_policy.json",
                "sha256": decision_policy_hash(decision_policy),
                "threshold": decision_policy.threshold,
                "comparator": decision_policy.comparator,
                "score_semantic": decision_policy.score_semantic,
                "source": decision_policy.source,
                "revision_id": decision_policy.revision_id,
                "operator_note": decision_policy.operator_note,
            },
            "input_contract": {
                "color_order": "RGB",
                "dtype": "uint8",
                "range": "0_255",
                "model_input_size": preprocessing_contract.get("model_input_size"),
                "tiling_enabled": preprocessing_contract.get("tiled", False),
                "tile_aggregation": preprocessing_contract.get("score_aggregation"),
                "padding_policy": preprocessing_contract.get("padding_policy"),
                "padding_value_rgb": preprocessing_contract.get("padding_value_rgb"),
            },
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
        ExportService._atomic_write_json(manifest_path, payload)
        return manifest_path

    @staticmethod
    def _next_deployment_revision_id(destination_directory: Path, base_name: str) -> str:
        destination_directory = destination_directory.expanduser().resolve()
        prefix = f"{base_name}_decision-"
        revisions = [
            int(path.name.removeprefix(prefix))
            for path in destination_directory.glob(f"{prefix}*")
            if path.is_dir() and path.name.removeprefix(prefix).isdigit()
        ] if destination_directory.is_dir() else []
        return f"{prefix}{max(revisions, default=0) + 1:03d}"

    @staticmethod
    def _runtime_versions() -> dict[str, str]:
        import cv2
        import numpy as np
        import torch

        return {
            "anomalib": REQUIRED_ANOMALIB_VERSION,
            "torch": str(torch.__version__),
            "opencv": str(cv2.__version__),
            "numpy": str(np.__version__),
        }

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
        from tempfile import NamedTemporaryFile

        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            temporary_path = Path(handle.name)
        temporary_path.replace(path)

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
    def _build_decision_policy(
        threshold_metadata: Mapping[str, object],
        calibrated_threshold_metadata: Mapping[str, object],
        active_revision: ThresholdRevisionResult | None,
        *,
        model_sha256: str,
        preprocessing_plan_sha256: str,
    ) -> DecisionPolicy:
        """Bind the selected run/revision threshold to the exact exported Torch artifact."""
        try:
            threshold = float(threshold_metadata["threshold_value"])
            base_threshold = float(calibrated_threshold_metadata.get("threshold_raw", threshold))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Deployment decision policy requires finite calibrated and active thresholds.") from exc
        score_semantic = str(threshold_metadata.get("score_semantic", ""))
        if not score_semantic:
            raise ValueError("Deployment decision policy requires a saved decision score semantic.")
        pixel_payload = threshold_metadata.get("pixel_operating_point")
        if not isinstance(pixel_payload, Mapping):
            raise ValueError("Deployment decision policy requires a pixel operating point.")
        return DecisionPolicy(
            threshold=threshold,
            score_semantic=score_semantic,
            source=active_revision.source if active_revision is not None else "calibrated",
            base_calibrated_threshold=base_threshold,
            revision_id=active_revision.revision_path.stem if active_revision is not None else "calibrated",
            model_sha256=model_sha256,
            preprocessing_plan_sha256=preprocessing_plan_sha256,
            pixel_operating_point=PixelThresholdOperatingPoint.from_dict(pixel_payload),
            operator_note=getattr(active_revision, "operator_note", "") if active_revision is not None else "",
        )

    @staticmethod
    def _legacy_preprocessing_plan_hash() -> str:
        return hashlib.sha256(b"legacy_none_v1").hexdigest()

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
        threshold_semantic: str = "",
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
        map_shapes: list[list[int]] = []
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
                map_shapes.append(list(reconstructed.anomaly_map.shape))
                decision_score = resolve_decision_score(
                    preprocessing_pipeline.plan,
                    postprocessed_image_score=(
                        ExportService._deployment_score(deployed_predictions[0]) if len(deployed_predictions) == 1 else None
                    ),
                    raw_image_score=(
                        ExportService._deployment_decision_score(deployed_predictions[0])
                        if preprocessing_pipeline.plan.model_id == "super_add" and len(deployed_predictions) == 1
                        else None
                    ),
                    reconstructed_map=reconstructed,
                    preprocessing_pipeline=preprocessing_pipeline,
                )
            else:
                deployment_input: str | Any = expected.source_path
                if inspection_processor is not None:
                    deployment_input = inspection_processor.apply_path(Path(expected.source_path))
                prediction = inferencer.predict(deployment_input)
                decision_score = resolve_decision_score(
                    None,
                    postprocessed_image_score=ExportService._deployment_score(prediction),
                    raw_image_score=None,
                )
            expected_semantic = threshold_semantic or expected.score_semantic
            if expected_semantic:
                require_matching_score_semantic(decision_score, expected_semantic)
            score = decision_score.value
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
            "reconstructed_map_shapes": map_shapes,
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

    @staticmethod
    def _deployment_decision_score(prediction: Any) -> float | None:
        """Read only the explicit application-level SuperADD decision score from a deployment output."""
        value = prediction.get("decision_score") if isinstance(prediction, dict) else getattr(prediction, "decision_score", None)
        return None if value is None else ExportService._deployment_score({"pred_score": value})

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

