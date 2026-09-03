"""Immutable training-run artifact helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

from app.core.dataset_manifest import sha256_file
from app.core.inspection_region import inspection_region_hash, read_inspection_region
from app.core.preprocessing_contract import read_resolved_preprocessing_plan, resolved_preprocessing_hash
from app.core.preprocessing_contract import image_preprocessing_hash, read_image_preprocessing_config
from app.core.threshold_contract import PixelThresholdOperatingPoint
from app.models.image_preprocessing import ImagePreprocessingConfig
from app.models.inspection_region import InspectionRegionConfig
from app.models.preprocessing_config import ResolvedPreprocessingPlan


@dataclass(frozen=True, slots=True)
class CanonicalCheckpoint:
    """The one Anomalib-selected checkpoint used by all subsequent run operations."""

    path: Path
    sha256: str


def resolve_canonical_checkpoint(engine: Any) -> CanonicalCheckpoint:
    """Resolve Anomalib's chosen best checkpoint without timestamp heuristics."""
    path_value = getattr(engine, "best_model_path", None)
    if not path_value:
        raise RuntimeError("Anomalib did not report a canonical final checkpoint.")
    path = Path(path_value).expanduser().resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Anomalib reported an invalid canonical checkpoint: {path}")
    return CanonicalCheckpoint(path=path, sha256=sha256_file(path))


def write_run_manifest(
    path: Path,
    *,
    canonical_checkpoint: CanonicalCheckpoint,
    dataset_manifest_sha256: str,
    split_counts: Mapping[str, Mapping[str, int]],
    threshold: float,
    threshold_metadata: Mapping[str, object] | None = None,
    extra: Mapping[str, object] | None = None,
) -> Path:
    """Write the run contract used by testing, inference, export, and deployment."""
    if not isfinite(threshold):
        raise ValueError("The persisted decision threshold must be finite.")
    payload: dict[str, object] = {
        "canonical_checkpoint": {
            "path": str(canonical_checkpoint.path),
            "sha256": canonical_checkpoint.sha256,
        },
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "effective_split_counts": {name: dict(counts) for name, counts in split_counts.items()},
        "threshold": threshold,
    }
    if threshold_metadata:
        metadata = _validated_threshold_metadata(threshold_metadata, threshold)
        payload["threshold_metadata"] = metadata
    if extra:
        payload.update(extra)
    pixel_operating_point = payload.get("pixel_operating_point")
    if pixel_operating_point is not None:
        if not isinstance(pixel_operating_point, Mapping):
            raise ValueError("Pixel operating point must be an object.")
        payload["pixel_operating_point"] = PixelThresholdOperatingPoint.from_dict(pixel_operating_point).to_dict()
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_evaluation_revision(
    run_directory: Path,
    *,
    canonical_checkpoint: CanonicalCheckpoint,
    calibration_manifest_sha256: str,
    final_test_manifest_sha256: str,
    threshold_metadata: Mapping[str, object],
    evaluation_metrics: Mapping[str, object] | None = None,
    revision_id: str | None = None,
    pixel_operating_point: Mapping[str, object] | None = None,
) -> Path:
    """Write an immutable recalibration/evaluation record without altering the model checkpoint."""
    metadata = _validated_threshold_metadata(
        threshold_metadata,
        float(threshold_metadata.get("threshold_value", "nan")),
    )
    revisions_directory = run_directory / "evaluation_revisions"
    revisions_directory.mkdir(parents=True, exist_ok=True)
    revision_id = revision_id or next_evaluation_revision_id(run_directory)
    if not revision_id.startswith("revision-") or not revision_id[9:].isdigit():
        raise ValueError("Evaluation revision ID must use the form revision-NNN.")
    metadata["threshold_revision"] = revision_id
    path = revisions_directory / f"{revision_id}.json"
    payload: dict[str, object] = {
        "revision_id": revision_id,
        "canonical_checkpoint": {
            "path": str(canonical_checkpoint.path),
            "sha256": canonical_checkpoint.sha256,
        },
        "calibration_manifest_sha256": calibration_manifest_sha256,
        "final_test_manifest_sha256": final_test_manifest_sha256,
        "threshold_metadata": metadata,
    }
    if evaluation_metrics:
        payload["evaluation_metrics"] = dict(evaluation_metrics)
    if pixel_operating_point is not None:
        payload["pixel_operating_point"] = PixelThresholdOperatingPoint.from_dict(pixel_operating_point).to_dict()
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def next_evaluation_revision_id(run_directory: Path) -> str:
    """Return the next deterministic revision identifier without writing a record."""
    revisions_directory = run_directory / "evaluation_revisions"
    existing = sorted(revisions_directory.glob("revision-*.json"))
    return f"revision-{len(existing) + 1:03d}"


def extract_decision_threshold(model: Any) -> float:
    """Read Anomalib's calibrated image decision boundary without inventing a fallback."""
    post_processor = getattr(model, "post_processor", None)
    threshold = getattr(post_processor, "image_threshold", None)
    value = getattr(threshold, "value", threshold)
    if hasattr(value, "item"):
        value = value.item()
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = float("nan")
    if isfinite(result):
        return result
    raise RuntimeError("Anomalib did not provide a finite calibrated decision threshold.")


def read_run_manifest(run_directory: Path) -> dict[str, Any]:
    """Load the persisted run contract and reject malformed manifests."""
    manifest_path = run_directory / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Run manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Run manifest must be a JSON object: {manifest_path}")
    return payload


def read_persisted_threshold(run_directory: Path) -> float:
    """Read the calibrated threshold used to approve this specific training run."""
    value = read_persisted_threshold_metadata(run_directory)["threshold_value"]
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Run manifest does not contain a decision threshold.") from exc
    if not isfinite(threshold):
        raise ValueError("Run manifest decision threshold must be finite.")
    return threshold


def read_persisted_threshold_metadata(run_directory: Path) -> dict[str, object]:
    """Read structured threshold evidence, retaining compatibility with legacy run manifests."""
    manifest = read_run_manifest(run_directory)
    metadata = manifest.get("threshold_metadata")
    if isinstance(metadata, dict):
        return _validated_threshold_metadata(metadata, float(metadata.get("threshold_value", "nan")))
    threshold = manifest.get("threshold")
    return _validated_threshold_metadata(
        {
            "threshold_value": threshold,
            "threshold_method": "legacy_anomalib_post_processor",
            "threshold_revision": "legacy",
        },
        float(threshold),
    )


def read_persisted_pixel_operating_point(run_directory: Path) -> PixelThresholdOperatingPoint:
    """Read the explicit mask operating point, leaving legacy runs mask-free."""
    payload = read_run_manifest(run_directory).get("pixel_operating_point")
    if payload is None:
        return PixelThresholdOperatingPoint()
    if not isinstance(payload, Mapping):
        raise ValueError("Run manifest pixel operating point must be an object.")
    return PixelThresholdOperatingPoint.from_dict(payload)


def read_canonical_checkpoint(run_directory: Path) -> CanonicalCheckpoint:
    """Read and verify the persisted canonical checkpoint before downstream work."""
    payload = read_run_manifest(run_directory)
    checkpoint = payload.get("canonical_checkpoint", {})
    persisted_path = Path(str(checkpoint.get("path", ""))).expanduser()
    checkpoint_path = (persisted_path if persisted_path.is_absolute() else run_directory / persisted_path).resolve()
    expected_hash = str(checkpoint.get("sha256", ""))
    if not checkpoint_path.is_file() or not expected_hash:
        raise ValueError("Run manifest does not contain a valid canonical checkpoint.")
    current_hash = sha256_file(checkpoint_path)
    if current_hash != expected_hash:
        raise ValueError("Canonical checkpoint hash does not match run_manifest.json.")
    return CanonicalCheckpoint(path=checkpoint_path, sha256=current_hash)


def read_verified_inspection_region(run_directory: Path) -> InspectionRegionConfig:
    """Load the immutable ROI sidecar and verify it is the contract recorded for this run."""
    manifest = read_run_manifest(run_directory)
    preprocessing = manifest.get("inspection_preprocessing")
    if not isinstance(preprocessing, dict):
        raise ValueError("Run manifest does not contain inspection ROI preprocessing provenance.")
    if preprocessing.get("metadata_file") != "inspection_region.json":
        raise ValueError("Run manifest does not reference the required inspection_region.json metadata file.")
    config = read_inspection_region(run_directory / "inspection_region.json")
    metadata_hash = str(preprocessing.get("metadata_sha256", ""))
    if not metadata_hash or inspection_region_hash(config) != metadata_hash:
        raise ValueError("Inspection ROI metadata hash does not match run_manifest.json.")
    if str(manifest.get("inspection_region_hash", "")) != metadata_hash:
        raise ValueError("Run inspection ROI hash does not match inspection preprocessing metadata.")
    if config.roi_contract_version != preprocessing.get("roi_contract_version"):
        raise ValueError("Inspection ROI contract version does not match run_manifest.json.")
    if preprocessing.get("source_size") != [config.source_width, config.source_height]:
        raise ValueError("Inspection ROI source size does not match run_manifest.json.")
    if preprocessing.get("rectified_size") != list(config.rectified_size()):
        raise ValueError("Inspection ROI rectified size does not match run_manifest.json.")
    return config


def read_verified_preprocessing_plan(run_directory: Path) -> ResolvedPreprocessingPlan | None:
    """Load a v2 plan or return ``None`` for a completed legacy preprocessing-v1 run."""
    manifest = read_run_manifest(run_directory)
    metadata = manifest.get("preprocessing_contract")
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise ValueError("Run manifest preprocessing contract must be an object.")
    if metadata.get("metadata_file") != "preprocessing_plan.json":
        raise ValueError("Run manifest does not reference the required preprocessing_plan.json metadata file.")
    plan = read_resolved_preprocessing_plan(run_directory / "preprocessing_plan.json")
    metadata_hash = str(metadata.get("metadata_sha256", ""))
    if not metadata_hash or resolved_preprocessing_hash(plan) != metadata_hash:
        raise ValueError("Preprocessing plan metadata hash does not match run_manifest.json.")
    if metadata.get("preprocessing_contract_version") != plan.preprocessing_contract_version:
        raise ValueError("Preprocessing contract version does not match run_manifest.json.")
    if metadata.get("model_id") != plan.model_id:
        raise ValueError("Preprocessing model ID does not match run_manifest.json.")
    if metadata.get("model_input_size") != list(plan.model_input_size):
        raise ValueError("Preprocessing model input size does not match run_manifest.json.")
    if metadata.get("score_aggregation") != plan.score_aggregation.value:
        raise ValueError("Preprocessing score aggregation does not match run_manifest.json.")
    if bool(metadata.get("tiled")) != plan.tiled:
        raise ValueError("Preprocessing tiling flag does not match run_manifest.json.")
    embedded_profile = metadata.get("image_preprocessing")
    if embedded_profile is not None and ImagePreprocessingConfig.from_dict(embedded_profile) != plan.image_preprocessing:
        raise ValueError("Embedded image preprocessing profile does not match preprocessing_plan.json.")
    profile_hash = metadata.get("image_preprocessing_sha256")
    if profile_hash is not None:
        if not isinstance(profile_hash, str) or image_preprocessing_hash(plan.image_preprocessing) != profile_hash:
            raise ValueError("Image preprocessing profile hash does not match run_manifest.json.")
        profile_file = metadata.get("image_preprocessing_file")
        if profile_file != "image_preprocessing.json":
            raise ValueError("Run manifest image preprocessing profile filename is invalid.")
        profile = read_image_preprocessing_config(run_directory / profile_file)
        if profile != plan.image_preprocessing:
            raise ValueError("Standalone image preprocessing profile does not match preprocessing_plan.json.")
    return plan


def _validated_threshold_metadata(metadata: Mapping[str, object], expected_threshold: float) -> dict[str, object]:
    """Validate the exact deployed threshold while preserving all calibration provenance."""
    payload = dict(metadata)
    threshold = _finite_threshold_value(payload.get("threshold_value"), "threshold_value")
    raw_threshold = _finite_threshold_value(payload.get("threshold_raw", threshold), "threshold_raw")
    deployed_threshold = _finite_threshold_value(payload.get("threshold_deployed", threshold), "threshold_deployed")
    if deployed_threshold != threshold:
        raise ValueError("Threshold metadata threshold_deployed must match threshold_value.")
    if isfinite(expected_threshold) and deployed_threshold != expected_threshold:
        raise ValueError("Threshold metadata does not match the persisted decision threshold.")
    payload["threshold_value"] = threshold
    payload["threshold_raw"] = raw_threshold
    payload["threshold_deployed"] = deployed_threshold
    score_semantic = payload.get("score_semantic")
    if score_semantic is not None and (not isinstance(score_semantic, str) or not score_semantic):
        raise ValueError("Threshold metadata score_semantic must be a non-empty string when provided.")
    comparator = payload.get("decision_comparator")
    if comparator is not None and comparator != "greater_than_or_equal":
        raise ValueError("Threshold metadata has an unsupported image decision comparator.")
    return payload


def _finite_threshold_value(value: object, name: str) -> float:
    """Parse one persisted threshold value without accepting nonfinite artifacts."""
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Threshold metadata must contain a finite {name}.") from exc
    if not isfinite(threshold):
        raise ValueError(f"Threshold metadata must contain a finite {name}.")
    return threshold