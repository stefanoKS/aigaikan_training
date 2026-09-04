"""Two-file Torch deployment package contract and raw-pixel reference loader."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.decision_score import DecisionScore, require_matching_score_semantic, resolve_decision_score
from app.core.preprocessing_pipeline import PreprocessingPipeline
from app.core.threshold_contract import PixelThresholdOperatingPoint
from app.models.image_preprocessing import ImagePreprocessingConfig
from app.models.inspection_region import InspectionRegionConfig
from app.models.preprocessing_config import ResolvedPreprocessingPlan

DEPLOYMENT_CONTRACT_VERSION = 1
DEPLOYMENT_MODEL_FILENAME = "model.pt"
DEPLOYMENT_METADATA_FILENAME = "deployment.json"
_REQUIRED_SECTIONS = frozenset(
    {
        "deployment",
        "input",
        "inspection_region",
        "image_preprocessing",
        "model_preprocessing",
        "model",
        "decision",
        "validation",
    }
)


@dataclass(frozen=True, slots=True)
class DeploymentPrediction:
    """One raw-pixel deployment inference result without visualization artifacts."""

    decision_score: float
    threshold: float
    is_ng: bool
    anomaly_map: np.ndarray
    score_semantic: str


class DeploymentPackage:
    """Load only ``model.pt`` and ``deployment.json`` and reproduce the frozen run contract."""

    def __init__(
        self,
        directory: Path,
        metadata: Mapping[str, object],
        pipeline: PreprocessingPipeline,
        inferencer: Any,
    ) -> None:
        self.directory = directory
        self.metadata = dict(metadata)
        self.pipeline = pipeline
        self.inferencer = inferencer

    @classmethod
    def load(
        cls,
        directory: Path,
        inferencer_factory: Callable[[Path], Any] | None = None,
        device: str = "cpu",
        *,
        require_validation: bool = True,
    ) -> "DeploymentPackage":
        """Verify and load a package without consulting its original training directory."""
        package_directory = directory.expanduser().resolve()
        model_path = package_directory / DEPLOYMENT_MODEL_FILENAME
        metadata_path = package_directory / DEPLOYMENT_METADATA_FILENAME
        metadata = read_deployment_json(metadata_path)
        inspection_region, image_preprocessing, plan = validate_deployment_json(
            metadata,
            model_path,
            require_validation=require_validation,
        )
        model = _section(metadata, "model")
        if inferencer_factory is None:
            if str(model.get("algorithm", "")) == "super_add":
                if model.get("expected_precision") == "float16" and device != "cuda":
                    raise ValueError("SuperADD FP16 deployment requires CUDA.")
                from app.core.superadd_deployment import SuperADDDeploymentInferencer

                inferencer = SuperADDDeploymentInferencer.load(model_path, device=device)
            else:
                from anomalib.deploy import TorchInferencer

                inferencer = TorchInferencer(path=model_path, device=device)
        else:
            inferencer = inferencer_factory(model_path)
        if str(model.get("algorithm", "")) == "super_add":
            validate_superadd_memory_bank(inferencer, model)
        pipeline = PreprocessingPipeline(inspection_region, plan)
        return cls(package_directory, metadata, pipeline, inferencer)

    def predict(self, raw_uint8_frame: np.ndarray) -> DeploymentPrediction:
        """Adapt raw Mono8/RGB pixels, then apply the frozen preprocessing and decision contract."""
        source_rgb = adapt_raw_input(raw_uint8_frame, _section(self.metadata, "input"))
        prepared, _rectified = self.pipeline.prepare_array_with_rectified(source_rgb)
        outputs = tuple(self.inferencer.predict(item.image_rgb) for item in prepared)
        maps = tuple(_anomaly_map(output) for output in outputs)
        reconstructed = self.pipeline.reconstruct_anomaly_maps(maps)
        model = _section(self.metadata, "model")
        decision_metadata = _section(self.metadata, "decision")
        algorithm = str(model.get("algorithm", ""))
        if algorithm == "super_add":
            if len(outputs) != 1:
                raise ValueError("SuperADD deployment must not use external tiling.")
            decision = DecisionScore(
                _finite_scalar(_value(outputs[0], "decision_score"), "SuperADD decision_score"),
                str(decision_metadata["score_semantic"]),
                "superadd_native_top_quantile_decision_score",
            )
        else:
            decision = resolve_decision_score(
                self.pipeline.plan,
                postprocessed_image_score=_finite_scalar(_value(outputs[0], "pred_score"), "postprocessed image score")
                if len(outputs) == 1
                else None,
                raw_image_score=None,
                reconstructed_map=reconstructed,
                preprocessing_pipeline=self.pipeline,
            )
        score_semantic = str(decision_metadata["score_semantic"])
        require_matching_score_semantic(decision, score_semantic)
        threshold = _finite_scalar(decision_metadata.get("threshold"), "deployment decision threshold")
        return DeploymentPrediction(
            decision_score=decision.value,
            threshold=threshold,
            is_ng=decision.value >= threshold,
            anomaly_map=reconstructed.anomaly_map,
            score_semantic=decision.semantic,
        )


def read_deployment_json(path: Path) -> dict[str, object]:
    """Read a two-file deployment document without accepting legacy bundle manifests."""
    if not path.is_file():
        raise FileNotFoundError(f"Deployment metadata is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("deployment.json is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("deployment.json must be a JSON object.")
    return payload


def validate_deployment_json(
    metadata: Mapping[str, object],
    model_path: Path,
    *,
    require_validation: bool = True,
) -> tuple[InspectionRegionConfig, ImagePreprocessingConfig, ResolvedPreprocessingPlan]:
    """Validate every deployment-critical field without inferring omitted preprocessing behavior."""
    if metadata.get("deployment_contract_version") != DEPLOYMENT_CONTRACT_VERSION:
        raise ValueError("Unsupported deployment contract version.")
    missing = _REQUIRED_SECTIONS.difference(metadata)
    if missing:
        raise ValueError(f"deployment.json is missing required sections: {', '.join(sorted(missing))}")
    _validate_two_file_directory(model_path.parent)
    deployment = _section(metadata, "deployment")
    expected_hash = str(deployment.get("model_sha256", ""))
    if not _is_sha256(expected_hash):
        raise ValueError("deployment.model_sha256 must be a SHA-256 digest.")
    if not model_path.is_file() or sha256_file(model_path) != expected_hash:
        raise ValueError("model.pt SHA-256 does not match deployment.json.")
    if not str(deployment.get("deployment_id", "")) or not str(deployment.get("training_run_id", "")):
        raise ValueError("deployment metadata must declare deployment and training-run IDs.")
    if not str(deployment.get("algorithm", "")):
        raise ValueError("deployment metadata must declare an algorithm.")
    try:
        datetime.fromisoformat(str(deployment.get("created_at", "")))
    except ValueError as exc:
        raise ValueError("deployment.created_at must be ISO-8601.") from exc

    _validate_input_contract(_section(metadata, "input"))
    inspection_region = InspectionRegionConfig.from_dict(dict(_section(metadata, "inspection_region")))
    image_preprocessing = ImagePreprocessingConfig.from_dict(_section(metadata, "image_preprocessing"))
    model_preprocessing = _section(metadata, "model_preprocessing")
    raw_plan = model_preprocessing.get("resolved_plan")
    if not isinstance(raw_plan, Mapping):
        raise ValueError("model_preprocessing must embed a complete resolved_plan.")
    plan = ResolvedPreprocessingPlan.from_dict(dict(raw_plan))
    if plan.image_preprocessing != image_preprocessing:
        raise ValueError("image_preprocessing does not match model_preprocessing.resolved_plan.")
    if inspection_region.enabled and plan.rectified_size != inspection_region.rectified_size():
        raise ValueError("model_preprocessing rectified size does not match inspection_region.")
    if bool(model_preprocessing.get("external_tiling")) != plan.tiled:
        raise ValueError("model_preprocessing external_tiling does not match the resolved plan.")
    if model_preprocessing.get("input_size") != list(plan.model_input_size):
        raise ValueError("model_preprocessing input_size does not match the resolved plan.")
    if model_preprocessing.get("padding") != list(plan.resolved_padding):
        raise ValueError("model_preprocessing padding does not match the resolved plan.")
    if model_preprocessing.get("alignment") != list(plan.model_alignment):
        raise ValueError("model_preprocessing alignment does not match the resolved plan.")
    if model_preprocessing.get("patch_size") != plan.patch_size:
        raise ValueError("model_preprocessing patch_size does not match the resolved plan.")
    if model_preprocessing.get("interpolation") != inspection_region.interpolation:
        raise ValueError("model_preprocessing interpolation does not match inspection_region.")
    if model_preprocessing.get("anomalib_transform_owner") != "model.pt":
        raise ValueError("model_preprocessing must declare model.pt as the Anomalib transform owner.")
    if model_preprocessing.get("expected_tensor_layout") != "NCHW":
        raise ValueError("model_preprocessing must declare NCHW tensor layout.")

    model = _section(metadata, "model")
    if str(model.get("algorithm", "")) != str(deployment["algorithm"]):
        raise ValueError("model algorithm does not match deployment metadata.")
    if model.get("id") != plan.model_id:
        raise ValueError("model ID does not match model_preprocessing resolved plan.")
    if not isinstance(model.get("profile"), Mapping) or not str(model.get("anomalib_version", "")) or not str(model.get("torch_version", "")):
        raise ValueError("model metadata must contain profile and runtime version information.")
    if model.get("anomalib_transform_owner") != "model.pt" or model.get("expected_tensor_layout") != "NCHW":
        raise ValueError("model metadata must declare model.pt transform ownership and NCHW layout.")
    if model.get("expected_precision") != model_preprocessing.get("expected_precision"):
        raise ValueError("model precision does not match model_preprocessing.")
    if model.get("expected_precision") not in {"float16", "float32"}:
        raise ValueError("model expected_precision must be float16 or float32.")
    if plan.model_id == "super_add":
        _validate_superadd_adapter_contract(model, model_preprocessing)
    _validate_decision(_section(metadata, "decision"), deployment, plan)
    validation = _section(metadata, "validation")
    if require_validation:
        _validate_validation(validation)
    return inspection_region, image_preprocessing, plan


def adapt_raw_input(frame: np.ndarray, input_contract: Mapping[str, object]) -> np.ndarray:
    """Convert only declared raw camera formats into canonical RGB uint8 pixels."""
    _validate_input_contract(input_contract)
    values = np.asarray(frame)
    if values.dtype != np.uint8:
        raise ValueError("Deployment raw input must use uint8 pixels.")
    if values.size == 0 or values.min() < 0 or values.max() > 255:
        raise ValueError("Deployment raw input pixels must be in the range 0..255.")
    if values.ndim == 2:
        if "HW" not in input_contract["accepted_layouts"]:
            raise ValueError("deployment.json does not accept Mono8 HW input.")
        return np.ascontiguousarray(cv2.cvtColor(values, cv2.COLOR_GRAY2RGB))
    if values.ndim != 3 or values.shape[2] != 3 or "HWC" not in input_contract["accepted_layouts"]:
        raise ValueError("Deployment color input must use HWC layout with exactly three channels.")
    if input_contract.get("color_input_order") != "RGB":
        raise ValueError("deployment.json declares an unsupported color input order; RGB is required.")
    return np.ascontiguousarray(values)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of the model file bound to deployment.json."""
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def superadd_memory_bank_metadata(model: Any) -> dict[str, object]:
    """Read the actual registered SuperADD memory-bank tensor without constructing replacement state."""
    implementation = getattr(model, "model", model)
    bank = getattr(implementation, "memory_bank", None)
    if bank is None or not hasattr(bank, "shape"):
        raise ValueError("SuperADD model.pt does not contain a non-empty registered memory bank.")
    shape = tuple(int(value) for value in bank.shape)
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError("SuperADD memory bank must have non-empty [layer, database, feature] dimensions.")
    return {
        "bank_count": shape[0],
        "feature_dimension": shape[2],
        "database_sizes": [shape[1]] * shape[0],
        "dtype": str(getattr(bank, "dtype", "")),
    }


def validate_superadd_memory_bank(inferencer: Any, model_metadata: Mapping[str, object]) -> None:
    """Require loaded SuperADD state to match the deployment metadata exactly before inference."""
    expected = model_metadata.get("memory_bank")
    if not isinstance(expected, Mapping):
        raise ValueError("SuperADD deployment metadata must contain memory-bank dimensions.")
    actual = superadd_memory_bank_metadata(getattr(inferencer, "model", inferencer))
    for key in ("bank_count", "feature_dimension", "database_sizes"):
        if actual[key] != expected.get(key):
            raise ValueError(f"SuperADD memory bank {key} does not match deployment.json.")
    expected_dtype = expected.get("dtype")
    if expected_dtype and actual["dtype"] != expected_dtype:
        raise ValueError("SuperADD memory bank dtype does not match deployment.json.")


def _validate_superadd_adapter_contract(
    model: Mapping[str, object],
    model_preprocessing: Mapping[str, object],
) -> None:
    if model.get("export_adapter") != "superadd_native_v1":
        raise ValueError("SuperADD model.pt must use the superadd_native_v1 deployment adapter.")
    output_contract = model.get("output_contract")
    if output_contract != {
        "decision_score": "superadd_native_top_quantile_score_v1",
        "anomaly_map": "continuous_unthresholded",
    }:
        raise ValueError("SuperADD deployment adapter output contract is invalid.")
    if model.get("external_tiling") is not False or model_preprocessing.get("external_tiling") is not False:
        raise ValueError("SuperADD deployment must prohibit external tiling.")


def _section(metadata: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = metadata.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"deployment.json section {name} must be an object.")
    return value


def _validate_input_contract(contract: Mapping[str, object]) -> None:
    if contract.get("dtype") != "uint8" or contract.get("range") != [0, 255]:
        raise ValueError("deployment input contract must declare uint8 pixels in the range 0..255.")
    layouts = contract.get("accepted_layouts")
    if layouts != ["HW", "HWC"]:
        raise ValueError("deployment input contract must accept exactly HW and HWC layouts.")
    if contract.get("canonical_color_order") != "RGB" or contract.get("color_input_order") != "RGB":
        raise ValueError("deployment input contract must explicitly require RGB color input.")
    if contract.get("mono_conversion") != "GRAY_TO_RGB":
        raise ValueError("deployment input contract must declare GRAY_TO_RGB Mono8 conversion.")


def _validate_decision(decision: Mapping[str, object], deployment: Mapping[str, object], plan: ResolvedPreprocessingPlan) -> None:
    threshold = _finite_scalar(decision.get("threshold"), "deployment decision threshold")
    if decision.get("comparator") != ">=" or decision.get("above_or_equal_label") != "NG" or decision.get("below_label") != "OK":
        raise ValueError("deployment decision must use score >= threshold -> NG and OK below threshold.")
    if decision.get("higher_is_more_anomalous") is not True:
        raise ValueError("deployment decision must declare higher scores as more anomalous.")
    if not str(decision.get("score_semantic", "")) or not str(decision.get("threshold_revision_id", "")):
        raise ValueError("deployment decision must declare score semantic and threshold revision ID.")
    if decision.get("threshold_source") not in {"calibrated", "operator_override"}:
        raise ValueError("deployment decision must declare a supported threshold source.")
    if not isinstance(decision.get("operator_note"), str):
        raise ValueError("deployment decision operator_note must be a string.")
    _finite_scalar(decision.get("base_calibrated_threshold"), "base calibrated threshold")
    pixel_policy = decision.get("pixel_operating_point")
    if not isinstance(pixel_policy, Mapping):
        raise ValueError("deployment decision must contain a separate pixel operating point.")
    PixelThresholdOperatingPoint.from_dict(pixel_policy)
    if plan.model_id == "super_add" and decision.get("score_semantic") != "superadd_native_top_quantile_score_v1":
        raise ValueError("SuperADD deployment decision must preserve the native top-quantile score semantic.")
    if not isfinite(threshold) or str(deployment.get("algorithm", "")) != plan.model_id:
        raise ValueError("deployment decision is inconsistent with the resolved model contract.")


def _validate_validation(validation: Mapping[str, object]) -> None:
    if validation.get("status") != "PASS":
        raise ValueError("deployment.json does not contain a successful export/reload parity validation.")
    if validation.get("artifact") != DEPLOYMENT_MODEL_FILENAME:
        raise ValueError("deployment validation must identify model.pt as the tested artifact.")
    _finite_scalar(validation.get("decision_threshold"), "validation decision_threshold")
    for name in (
        "score_tolerance",
        "map_tolerance",
        "max_abs_score_error",
        "mean_abs_map_error",
        "max_abs_map_error",
        "decision_match_rate",
    ):
        value = _finite_scalar(validation.get(name), f"validation {name}")
        if value < 0:
            raise ValueError(f"validation {name} must be non-negative.")
    if _finite_scalar(validation.get("decision_match_rate"), "validation decision_match_rate") != 1.0:
        raise ValueError("deployment validation decision_match_rate must equal 1.0.")
    count = validation.get("number_of_test_images")
    if not isinstance(count, int) or isinstance(count, bool):
        raise ValueError("validation number_of_test_images must be positive.")
    if count <= 0:
        raise ValueError("validation number_of_test_images must be positive.")


def _validate_two_file_directory(directory: Path) -> None:
    if not directory.is_dir():
        raise FileNotFoundError(f"Deployment directory is missing: {directory}")
    names = {path.name for path in directory.iterdir()}
    if names != {DEPLOYMENT_MODEL_FILENAME, DEPLOYMENT_METADATA_FILENAME}:
        raise ValueError("Deployment directory must contain exactly model.pt and deployment.json.")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value.casefold())


def _value(output: Any, name: str) -> Any:
    return output.get(name) if isinstance(output, Mapping) else getattr(output, name, None)


def _finite_scalar(value: object, name: str) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numel") and value.numel() != 1:
        raise ValueError(f"{name} must contain exactly one value.")
    if hasattr(value, "item"):
        value = value.item()
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite.") from exc
    if not isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _anomaly_map(output: Any) -> np.ndarray:
    value = _value(output, "anomaly_map")
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    values = np.asarray(value)
    while values.ndim > 2:
        values = values[0]
    if values.ndim != 2 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Deployment output must contain one finite anomaly map.")
    return np.ascontiguousarray(values.astype(np.float32))