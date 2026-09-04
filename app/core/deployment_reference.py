"""Verified in-memory reference inference for a checksummed Torch deployment package."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Callable, Mapping

import numpy as np

from app.core.decision_policy import DecisionPolicy, decision_policy_hash, read_decision_policy
from app.core.decision_score import DecisionScore, require_matching_score_semantic, resolve_decision_score
from app.core.inference_timing import InferenceTimingRecord, timed_call, timed_model_call
from app.core.inspection_region import InspectionRegionProcessor, read_inspection_region
from app.core.prediction_artifacts import render_fixed_unit_interval_heatmap
from app.core.preprocessing_contract import (
    image_preprocessing_hash,
    read_image_preprocessing_config,
    read_resolved_preprocessing_plan,
    resolved_preprocessing_hash,
)
from app.core.preprocessing_pipeline import PreprocessingPipeline, ReconstructedAnomalyMap
from app.models.preprocessing_config import ResolvedPreprocessingPlan

SUPPORTED_DEPLOYMENT_CONTRACT_VERSIONS = frozenset({2, 3})


@dataclass(frozen=True, slots=True)
class DeploymentInferenceResult:
    """One in-memory deployment decision with continuous map and separated timing phases."""

    score: float
    score_semantic: str
    score_source: str
    predicted_label: str
    continuous_anomaly_map: np.ndarray
    heatmap_rgba: np.ndarray
    binary_mask: np.ndarray | None
    valid_roi_mask: np.ndarray
    timing: InferenceTimingRecord


class TorchDeploymentReferenceInferencer:
    """Load a verified Torch package and run raw RGB arrays without disk staging."""

    def __init__(
        self,
        package_directory: Path,
        manifest: Mapping[str, object],
        policy: DecisionPolicy,
        inspection_processor: InspectionRegionProcessor,
        plan: ResolvedPreprocessingPlan | None,
        inferencer: Any,
        model_load_ms: float,
        device: str,
    ) -> None:
        self.package_directory = package_directory
        self.manifest = dict(manifest)
        self.policy = policy
        self.inspection_processor = inspection_processor
        self.plan = plan
        self.preprocessing_pipeline = PreprocessingPipeline(inspection_processor.config, plan) if plan is not None else None
        self.inferencer = inferencer
        self.model_load_ms = model_load_ms
        self.device = device

    @classmethod
    def load(
        cls,
        package_directory: Path,
        inferencer_factory: Callable[[Path], Any] | None = None,
        device: str = "cpu",
    ) -> "TorchDeploymentReferenceInferencer":
        """Verify every referenced package artifact before loading a Torch inferencer."""
        package_directory = package_directory.expanduser().resolve()
        manifest = read_deployment_manifest(package_directory)
        version = manifest.get("deployment_contract_version")
        if version not in SUPPORTED_DEPLOYMENT_CONTRACT_VERSIONS:
            raise ValueError("Unsupported deployment contract version.")
        if version < 3:
            raise ValueError("Deployment contract lacks decision_policy.json and cannot be used for production reference inference.")
        cls._verify_manifest_artifacts(package_directory, manifest)
        policy = cls._read_verified_policy(package_directory, manifest)
        inspection_path = package_directory / "inspection_region.json"
        inspection_processor = InspectionRegionProcessor(read_inspection_region(inspection_path))
        plan = cls._read_verified_plan(package_directory, manifest, policy)
        model_path = cls._verified_torch_artifact(package_directory, manifest, policy)
        started = perf_counter_ns()
        if inferencer_factory is None:
            from anomalib.deploy import TorchInferencer

            inferencer = TorchInferencer(path=model_path, device=device)
        else:
            inferencer = inferencer_factory(model_path)
        model_load_ms = (perf_counter_ns() - started) / 1_000_000
        return cls(package_directory, manifest, policy, inspection_processor, plan, inferencer, model_load_ms, device)

    def infer_rgb(self, source_rgb: np.ndarray) -> DeploymentInferenceResult:
        """Run raw RGB ``uint8`` input entirely in memory through the frozen package contract."""
        source = self._validated_source_rgb(source_rgb)
        started = perf_counter_ns()
        if self.preprocessing_pipeline is None:
            rectified, roi_ms = timed_call(lambda: self.inspection_processor.apply(source))
            model_inputs = (rectified,)
            image_filter_ms = 0.0
            padding_tiling_ms = 0.0
            valid_mask = np.ones(rectified.shape[:2], dtype=bool)
        else:
            rectified, roi_ms = timed_call(lambda: self.preprocessing_pipeline._rectify(source))
            preprocessed, image_filter_ms = timed_call(lambda: self.preprocessing_pipeline.preprocess_rectified(rectified))
            prepared, padding_tiling_ms = timed_call(
                lambda: tuple(self.preprocessing_pipeline._prepare_tile(preprocessed, tile) for tile in self.plan.tiles)
            )
            model_inputs = tuple(item.image_rgb for item in prepared)
            valid_mask = None
        outputs, model_forward_ms = timed_model_call(
            lambda: tuple(self.inferencer.predict(item) for item in model_inputs), self.device
        )
        decision_score, continuous_map, valid_mask, application_postprocess_ms = self._resolve_outputs(outputs, valid_mask)
        require_matching_score_semantic(decision_score, self.policy.score_semantic)
        binary_mask = (
            np.logical_and(valid_mask, continuous_map >= self.policy.pixel_operating_point.active_threshold).astype(np.uint8) * 255
            if self.policy.pixel_operating_point.active_threshold is not None
            else None
        )
        heatmap = render_fixed_unit_interval_heatmap(continuous_map, valid_mask)
        total_ms = (perf_counter_ns() - started) / 1_000_000
        timing = InferenceTimingRecord(
            input_decode_ms=0.0,
            roi_rectification_ms=roi_ms,
            image_filter_ms=image_filter_ms,
            padding_tiling_ms=padding_tiling_ms,
            preprocess_compute_ms=roi_ms + image_filter_ms + padding_tiling_ms,
            staging_io_ms=0.0,
            host_to_device_ms=None,
            model_forward_ms=model_forward_ms,
            native_postprocess_ms=None,
            application_postprocess_ms=application_postprocess_ms,
            inference_total_ms=model_forward_ms + application_postprocess_ms,
            artifact_io_ms=0.0,
            end_to_end_ms=total_ms,
            model_load_ms=self.model_load_ms,
            device=self.device,
            dtype="uint8_rgb",
            batch_size=1,
            tile_count=len(model_inputs),
            raw_input_size=(source.shape[1], source.shape[0]),
            rectified_size=(rectified.shape[1], rectified.shape[0]),
            model_input_size=(model_inputs[0].shape[1], model_inputs[0].shape[0]),
            warmup_status="not_warmed",
        )
        return DeploymentInferenceResult(
            score=decision_score.value,
            score_semantic=decision_score.semantic,
            score_source=decision_score.source,
            predicted_label="NG" if decision_score.value >= self.policy.threshold else "OK",
            continuous_anomaly_map=continuous_map,
            heatmap_rgba=heatmap,
            binary_mask=binary_mask,
            valid_roi_mask=valid_mask,
            timing=timing,
        )

    def _resolve_outputs(
        self,
        outputs: tuple[Any, ...],
        direct_valid_mask: np.ndarray | None,
    ) -> tuple[DecisionScore, np.ndarray, np.ndarray, float]:
        started = perf_counter_ns()
        maps = tuple(self._anomaly_map(output) for output in outputs)
        if self.preprocessing_pipeline is not None:
            reconstructed = self.preprocessing_pipeline.reconstruct_anomaly_maps(maps)
            decision = resolve_decision_score(
                self.plan,
                postprocessed_image_score=self._score(outputs[0]) if len(outputs) == 1 else None,
                raw_image_score=self._raw_score(outputs[0]) if self.plan and self.plan.model_id == "super_add" else None,
                reconstructed_map=reconstructed,
                preprocessing_pipeline=self.preprocessing_pipeline,
            )
            return decision, reconstructed.anomaly_map, reconstructed.valid_mask, (perf_counter_ns() - started) / 1_000_000
        continuous_map = self._map_array(maps[0])
        decision = resolve_decision_score(None, postprocessed_image_score=self._score(outputs[0]), raw_image_score=None)
        return decision, continuous_map, direct_valid_mask if direct_valid_mask is not None else np.ones(continuous_map.shape, dtype=bool), (perf_counter_ns() - started) / 1_000_000

    @staticmethod
    def _verify_manifest_artifacts(package_directory: Path, manifest: Mapping[str, object]) -> None:
        artifacts = manifest.get("included_run_artifacts")
        if not isinstance(artifacts, Mapping):
            raise ValueError("Deployment manifest is missing required artifact checksums.")
        required = {"config.json", "environment.json", "inspection_region.json", "preprocessing.json", "decision_policy.json"}
        missing = required.difference(artifacts)
        if missing:
            raise ValueError(f"Deployment manifest is missing required artifact checksums: {', '.join(sorted(missing))}")
        for relative, expected_hash in artifacts.items():
            if not isinstance(relative, str) or not isinstance(expected_hash, str):
                raise ValueError("Deployment manifest artifact checksums are invalid.")
            path = package_directory / relative
            if not path.is_file() or _sha256_file(path) != expected_hash:
                raise ValueError(f"Deployment artifact checksum mismatch: {relative}")

    @staticmethod
    def _read_verified_policy(package_directory: Path, manifest: Mapping[str, object]) -> DecisionPolicy:
        policy_metadata = manifest.get("decision_policy")
        if not isinstance(policy_metadata, Mapping) or policy_metadata.get("file") != "decision_policy.json":
            raise ValueError("Deployment manifest is missing a valid decision policy reference.")
        policy_path = package_directory / "decision_policy.json"
        policy = read_decision_policy(policy_path)
        if decision_policy_hash(policy) != policy_metadata.get("sha256"):
            raise ValueError("Deployment decision policy checksum does not match deployment manifest.")
        if (
            policy.threshold != policy_metadata.get("threshold")
            or policy.score_semantic != policy_metadata.get("score_semantic")
            or policy.comparator != policy_metadata.get("comparator")
            or policy.source != policy_metadata.get("source")
            or policy.revision_id != policy_metadata.get("revision_id")
        ):
            raise ValueError("Deployment decision policy disagrees with deployment manifest.")
        return policy

    @staticmethod
    def _read_verified_plan(
        package_directory: Path,
        manifest: Mapping[str, object],
        policy: DecisionPolicy,
    ) -> ResolvedPreprocessingPlan | None:
        contract = manifest.get("preprocessing_contract")
        if not isinstance(contract, Mapping):
            raise ValueError("Deployment manifest is missing preprocessing contract metadata.")
        input_contract = manifest.get("input_contract")
        if not isinstance(input_contract, Mapping):
            raise ValueError("Deployment manifest is missing explicit model input contract metadata.")
        if (
            input_contract.get("color_order") != "RGB"
            or input_contract.get("dtype") != "uint8"
            or input_contract.get("range") != "0_255"
        ):
            raise ValueError("Deployment input contract must declare RGB uint8 values in the range 0_255.")
        profile_file = contract.get("image_preprocessing_file")
        profile_hash = contract.get("image_preprocessing_sha256")
        if profile_file != "preprocessing.json" or not isinstance(profile_hash, str):
            raise ValueError("Deployment preprocessing contract is missing standalone profile metadata.")
        profile = read_image_preprocessing_config(package_directory / profile_file)
        if image_preprocessing_hash(profile) != profile_hash:
            raise ValueError("Deployment image preprocessing profile checksum does not match deployment manifest.")
        if contract.get("legacy"):
            if policy.preprocessing_plan_sha256 != _legacy_plan_hash():
                raise ValueError("Deployment legacy preprocessing policy is not bound to the legacy plan identity.")
            return None
        if contract.get("metadata_file") != "preprocessing_plan.json":
            raise ValueError("Deployment preprocessing contract does not reference preprocessing_plan.json.")
        plan = read_resolved_preprocessing_plan(package_directory / "preprocessing_plan.json")
        plan_hash = resolved_preprocessing_hash(plan)
        if plan_hash != contract.get("metadata_sha256") or plan_hash != policy.preprocessing_plan_sha256:
            raise ValueError("Deployment preprocessing plan checksum does not match decision policy or manifest.")
        if plan.image_preprocessing != profile:
            raise ValueError("Deployment standalone preprocessing profile does not match preprocessing_plan.json.")
        model = manifest.get("model")
        model_id = model.get("id") if isinstance(model, Mapping) else None
        if not isinstance(model_id, str) or _normalize_model_id(model_id) != _normalize_model_id(plan.model_id):
            raise ValueError("Deployment model identifier does not match preprocessing_plan.json.")
        if input_contract.get("model_input_size") != list(plan.model_input_size):
            raise ValueError("Deployment model input dimensions do not match preprocessing_plan.json.")
        return plan

    @staticmethod
    def _verified_torch_artifact(package_directory: Path, manifest: Mapping[str, object], policy: DecisionPolicy) -> Path:
        exports = manifest.get("exports")
        if not isinstance(exports, list):
            raise ValueError("Deployment manifest is missing exported artifact records.")
        torch_export = next((item for item in exports if isinstance(item, Mapping) and item.get("format") == "torch"), None)
        if not isinstance(torch_export, Mapping):
            raise ValueError("Deployment package does not contain a validated Torch artifact.")
        artifact_name = torch_export.get("path")
        expected_hash = torch_export.get("sha256")
        if not isinstance(artifact_name, str) or not isinstance(expected_hash, str):
            raise ValueError("Deployment Torch artifact metadata is invalid.")
        artifact_path = package_directory / artifact_name
        if not artifact_path.is_file() or _sha256_file(artifact_path) != expected_hash:
            raise ValueError("Deployment Torch artifact checksum mismatch.")
        if expected_hash != policy.model_sha256:
            raise ValueError("Deployment decision policy is not bound to the packaged Torch artifact.")
        return artifact_path

    @staticmethod
    def _validated_source_rgb(source_rgb: np.ndarray) -> np.ndarray:
        source = np.asarray(source_rgb)
        if source.dtype != np.uint8 or source.ndim != 3 or source.shape[2] != 3 or source.shape[0] == 0 or source.shape[1] == 0:
            raise ValueError("Deployment reference inference requires a non-empty uint8 RGB array.")
        return np.ascontiguousarray(source)

    @staticmethod
    def _value(output: Any, name: str) -> Any:
        return output.get(name) if isinstance(output, Mapping) else getattr(output, name, None)

    @classmethod
    def _score(cls, output: Any) -> float:
        return cls._finite_scalar(cls._value(output, "pred_score"), "postprocessed image score")

    @classmethod
    def _raw_score(cls, output: Any) -> float | None:
        value = cls._value(output, "decision_score")
        return None if value is None else cls._finite_scalar(value, "SuperADD decision_score")

    @classmethod
    def _anomaly_map(cls, output: Any) -> np.ndarray:
        return cls._map_array(cls._value(output, "anomaly_map"))

    @staticmethod
    def _finite_scalar(value: Any, name: str) -> float:
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "numel") and value.numel() != 1:
            raise ValueError(f"Deployment {name} must contain exactly one value.")
        if hasattr(value, "item"):
            value = value.item()
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Deployment output is missing a numeric {name}.") from exc
        if not isfinite(result):
            raise ValueError(f"Deployment {name} must be finite.")
        return result

    @staticmethod
    def _map_array(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        values = np.asarray(value)
        while values.ndim > 2:
            values = values[0]
        if values.ndim != 2 or values.size == 0 or not np.isfinite(values).all():
            raise ValueError("Deployment output must contain one finite anomaly map.")
        return np.ascontiguousarray(values.astype(np.float32))


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _legacy_plan_hash() -> str:
    import hashlib

    return hashlib.sha256(b"legacy_none_v1").hexdigest()


def read_deployment_manifest(package_directory: Path) -> dict[str, object]:
    """Read supported v2/v3 manifest structure without assigning an unsafe default policy."""
    manifest_path = package_directory.expanduser().resolve() / "deployment_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Deployment manifest is missing: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Deployment manifest is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Deployment manifest must be a JSON object.")
    if payload.get("deployment_contract_version") not in SUPPORTED_DEPLOYMENT_CONTRACT_VERSIONS:
        raise ValueError("Unsupported deployment contract version.")
    return payload


def _normalize_model_id(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())