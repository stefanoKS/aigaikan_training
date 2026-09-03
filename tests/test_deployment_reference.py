"""Tests for verified in-memory Torch deployment reference inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from app.core.decision_policy import DecisionPolicy, decision_policy_hash, write_decision_policy
from app.core.deployment_reference import TorchDeploymentReferenceInferencer, read_deployment_manifest
from app.core.inspection_region import write_inspection_region
from app.core.preprocessing_contract import (
    image_preprocessing_hash,
    resolved_preprocessing_hash,
    write_image_preprocessing_config,
    write_resolved_preprocessing_plan,
)
from app.core.threshold_contract import PixelThresholdOperatingPoint
from app.models.inspection_region import InspectionRegionConfig
from app.models.preprocessing_config import PreprocessingConfig
from app.core.prediction_contract import POSTPROCESSED_SCORE_SEMANTIC, SUPERADD_NATIVE_IMAGE_SCORE_SEMANTIC


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package(tmp_path: Path, *, model_id: str = "patchcore", pixel_enabled: bool = False) -> Path:
    package = tmp_path / "package"
    package.mkdir()
    model_path = package / "model.pt"
    model_path.write_bytes(b"torch model")
    roi = InspectionRegionConfig()
    write_inspection_region(package / "inspection_region.json", roi)
    plan = PreprocessingConfig().resolve(model_id, (3, 2))
    write_resolved_preprocessing_plan(package / "preprocessing_plan.json", plan)
    standalone_profile = write_image_preprocessing_config(package / "preprocessing.json", plan.image_preprocessing)
    (package / "config.json").write_text("{}", encoding="utf-8")
    (package / "environment.json").write_text("{}", encoding="utf-8")
    policy = DecisionPolicy(
        threshold=1.5 if model_id == "super_add" else 0.5,
        score_semantic=SUPERADD_NATIVE_IMAGE_SCORE_SEMANTIC if model_id == "super_add" else POSTPROCESSED_SCORE_SEMANTIC,
        source="operator_override",
        base_calibrated_threshold=1.4 if model_id == "super_add" else 0.4,
        revision_id="threshold-003",
        model_sha256=_sha256(model_path),
        preprocessing_plan_sha256=resolved_preprocessing_hash(plan),
        pixel_operating_point=PixelThresholdOperatingPoint(enabled=pixel_enabled, threshold=0.35),
        operator_note="test",
    )
    policy_path = write_decision_policy(package / "decision_policy.json", policy)
    included = {path.name: _sha256(path) for path in (package / "inspection_region.json", package / "preprocessing_plan.json", standalone_profile, package / "config.json", package / "environment.json", policy_path)}
    manifest = {
        "deployment_contract_version": 3,
        "included_run_artifacts": included,
        "decision_policy": {
            "file": policy_path.name,
            "sha256": decision_policy_hash(policy),
            "threshold": policy.threshold,
            "comparator": ">=",
            "score_semantic": policy.score_semantic,
            "source": policy.source,
            "revision_id": policy.revision_id,
        },
        "preprocessing_contract": {
            "metadata_file": "preprocessing_plan.json",
            "metadata_sha256": resolved_preprocessing_hash(plan),
            "image_preprocessing_file": "preprocessing.json",
            "image_preprocessing_sha256": image_preprocessing_hash(plan.image_preprocessing),
        },
        "model": {"id": model_id},
        "input_contract": {
            "color_order": "RGB",
            "dtype": "uint8",
            "range": "0_255",
            "model_input_size": list(plan.model_input_size),
        },
        "exports": [{"format": "torch", "path": model_path.name, "sha256": _sha256(model_path)}],
    }
    (package / "deployment_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return package


def test_reference_inference_uses_frozen_plan_policy_maps_masks_and_timing(tmp_path: Path) -> None:
    package = _package(tmp_path, pixel_enabled=True)

    reference = TorchDeploymentReferenceInferencer.load(
        package,
        lambda _model_path: type("Inferencer", (), {"predict": lambda _self, _input: {"pred_score": 0.4, "anomaly_map": np.full((2, 3), 0.4, dtype=np.float32)}})(),
    )
    result = reference.infer_rgb(np.zeros((2, 3, 3), dtype=np.uint8))

    assert result.score == pytest.approx(0.4)
    assert result.score_semantic == POSTPROCESSED_SCORE_SEMANTIC
    assert result.predicted_label == "OK"
    assert result.continuous_anomaly_map.shape == (2, 3)
    assert result.heatmap_rgba.shape == (2, 3, 4)
    assert result.binary_mask is not None and result.binary_mask.max() == 255
    assert result.timing.model_forward_ms is not None and result.timing.model_forward_ms >= 0
    assert result.timing.end_to_end_ms is not None and result.timing.end_to_end_ms >= 0


def test_reference_inference_preserves_unbounded_superadd_raw_score(tmp_path: Path) -> None:
    package = _package(tmp_path, model_id="super_add")
    map_values = np.full((448, 448), 0.4, dtype=np.float32)

    reference = TorchDeploymentReferenceInferencer.load(
        package,
        lambda _model_path: type(
            "Inferencer",
            (),
            {"predict": lambda _self, _input: {"pred_score": 1.0, "raw_pred_score": 1.7, "anomaly_map": map_values}},
        )(),
    )
    result = reference.infer_rgb(np.zeros((2, 3, 3), dtype=np.uint8))

    assert result.score == pytest.approx(1.7)
    assert result.score_semantic == SUPERADD_NATIVE_IMAGE_SCORE_SEMANTIC
    assert result.predicted_label == "NG"


def test_reference_inference_fails_closed_for_package_checksum_tampering(tmp_path: Path) -> None:
    package = _package(tmp_path)
    (package / "config.json").write_text("tampered", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        TorchDeploymentReferenceInferencer.load(package, lambda _model_path: object())


def test_reference_inference_fails_closed_when_standalone_profile_disagrees_with_plan(tmp_path: Path) -> None:
    package = _package(tmp_path)
    (package / "preprocessing.json").write_text("{}", encoding="utf-8")
    manifest_path = package / "deployment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["included_run_artifacts"]["preprocessing.json"] = _sha256(package / "preprocessing.json")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported image preprocessing schema version"):
        TorchDeploymentReferenceInferencer.load(package, lambda _model_path: object())


def test_reference_inference_rejects_manifest_policy_disagreement(tmp_path: Path) -> None:
    package = _package(tmp_path)
    manifest_path = package / "deployment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["decision_policy"]["source"] = "calibrated"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="disagrees"):
        TorchDeploymentReferenceInferencer.load(package, lambda _model_path: object())


def test_reference_inference_rejects_model_or_input_contract_mismatch(tmp_path: Path) -> None:
    package = _package(tmp_path)
    manifest_path = package / "deployment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["input_contract"]["model_input_size"] = [999, 999]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="input dimensions"):
        TorchDeploymentReferenceInferencer.load(package, lambda _model_path: object())


def test_legacy_deployment_manifest_remains_readable_but_cannot_infer_without_policy(tmp_path: Path) -> None:
    package = tmp_path / "legacy-package"
    package.mkdir()
    (package / "deployment_manifest.json").write_text(json.dumps({"deployment_contract_version": 2}), encoding="utf-8")

    assert read_deployment_manifest(package)["deployment_contract_version"] == 2
    with pytest.raises(ValueError, match="lacks decision_policy"):
        TorchDeploymentReferenceInferencer.load(package, lambda _model_path: object())