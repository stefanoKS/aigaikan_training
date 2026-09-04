"""Two-file deployment package contract tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from app.core.deployment_package import DeploymentPackage, sha256_file
from app.core.prediction_contract import POSTPROCESSED_SCORE_SEMANTIC, SUPERADD_NATIVE_IMAGE_SCORE_SEMANTIC
from app.models.image_preprocessing import ImagePreprocessingConfig
from app.models.inspection_region import InspectionRegionConfig
from app.models.preprocessing_config import PreprocessingConfig
from app.core.threshold_contract import PixelThresholdOperatingPoint


def _package(tmp_path: Path, *, model_id: str = "patchcore") -> Path:
    directory = tmp_path / "deployment"
    directory.mkdir(parents=True)
    model_path = directory / "model.pt"
    model_path.write_bytes(b"portable torch deployment")
    plan = PreprocessingConfig().resolve(model_id, (4, 3))
    decision_semantic = SUPERADD_NATIVE_IMAGE_SCORE_SEMANTIC if model_id == "super_add" else POSTPROCESSED_SCORE_SEMANTIC
    metadata = {
        "deployment_contract_version": 1,
        "deployment": {
            "deployment_id": "run_deployment_001",
            "created_at": "2026-09-04T00:00:00+00:00",
            "training_run_id": "run",
            "algorithm": model_id,
            "model_sha256": sha256_file(model_path),
        },
        "input": {
            "dtype": "uint8",
            "range": [0, 255],
            "accepted_layouts": ["HW", "HWC"],
            "canonical_color_order": "RGB",
            "color_input_order": "RGB",
            "mono_conversion": "GRAY_TO_RGB",
        },
        "inspection_region": InspectionRegionConfig().to_dict(),
        "image_preprocessing": ImagePreprocessingConfig().to_dict(),
        "model_preprocessing": {
            "resolved_plan": plan.to_dict(),
            "input_size": list(plan.model_input_size),
            "padding": list(plan.resolved_padding),
            "alignment": list(plan.model_alignment),
            "patch_size": plan.patch_size,
            "external_tiling": plan.tiled,
            "interpolation": InspectionRegionConfig().interpolation,
            "anomalib_transform_owner": "model.pt",
            "expected_tensor_layout": "NCHW",
            "expected_precision": "float32",
        },
        "model": {
            "id": model_id,
            "algorithm": model_id,
            "anomalib_version": "2.6.0",
            "torch_version": "test",
            "profile": {},
            "anomalib_transform_owner": "model.pt",
            "expected_tensor_layout": "NCHW",
            "expected_precision": "float32",
        } | (
            {
                "memory_bank": {
                    "bank_count": 2,
                    "feature_dimension": 4,
                    "database_sizes": [3, 3],
                    "dtype": "float32",
                }
            }
            if model_id == "super_add"
            else {}
        ),
        "decision": {
            "score_semantic": decision_semantic,
            "threshold": 1.7 if model_id == "super_add" else 0.5,
            "comparator": ">=",
            "above_or_equal_label": "NG",
            "below_label": "OK",
            "higher_is_more_anomalous": True,
            "threshold_source": "operator_override",
            "base_calibrated_threshold": 1.2 if model_id == "super_add" else 0.4,
            "threshold_revision_id": "threshold-003",
            "operator_note": "line setting",
            "pixel_operating_point": PixelThresholdOperatingPoint().to_dict(),
        },
        "validation": {
            "status": "PASS",
            "score_tolerance": 0.0001,
            "map_tolerance": 0.0001,
            "max_abs_score_error": 0.0,
            "mean_abs_map_error": 0.0,
            "max_abs_map_error": 0.0,
            "decision_match_rate": 1.0,
            "number_of_test_images": 1,
            "artifact": "model.pt",
            "decision_threshold": 1.7 if model_id == "super_add" else 0.5,
        },
    }
    (directory / "deployment.json").write_text(json.dumps(metadata), encoding="utf-8")
    return directory


def _inferencer(outputs: list[dict[str, object]], received: list[np.ndarray]):
    class FakeInferencer:
        def predict(self, image: np.ndarray) -> dict[str, object]:
            received.append(image.copy())
            return outputs.pop(0)

    return FakeInferencer()


def _superadd_inferencer(outputs: list[dict[str, object]], received: list[np.ndarray]):
    inferencer = _inferencer(outputs, received)
    inferencer.model = type("SuperADD", (), {"model": type("Core", (), {"memory_bank": np.zeros((2, 3, 4), dtype=np.float32)})()})()
    return inferencer


def test_two_file_package_adapts_mono8_to_rgb_and_decides_from_postprocessed_score(tmp_path: Path) -> None:
    directory = _package(tmp_path)
    received: list[np.ndarray] = []
    package = DeploymentPackage.load(
        directory,
        lambda _model: _inferencer([{"pred_score": 0.5, "anomaly_map": np.full((3, 4), 0.5, dtype=np.float32)}], received),
    )

    result = package.predict(np.array([[0, 127, 255, 10], [20, 30, 40, 50], [60, 70, 80, 90]], dtype=np.uint8))

    assert received[0].shape == (3, 4, 3)
    assert np.array_equal(received[0][:, :, 0], received[0][:, :, 1])
    assert np.array_equal(received[0][:, :, 1], received[0][:, :, 2])
    assert result.decision_score == pytest.approx(0.5)
    assert result.is_ng
    assert result.score_semantic == POSTPROCESSED_SCORE_SEMANTIC


def test_two_file_package_preserves_rgb_uint8_and_rejects_invalid_raw_input(tmp_path: Path) -> None:
    directory = _package(tmp_path)
    received: list[np.ndarray] = []
    color = np.full((3, 4, 3), [10, 20, 30], dtype=np.uint8)
    package = DeploymentPackage.load(
        directory,
        lambda _model: _inferencer([{"pred_score": 0.4, "anomaly_map": np.full((3, 4), 0.4, dtype=np.float32)}], received),
    )

    package.predict(color)

    assert np.array_equal(received[0], color)
    with pytest.raises(ValueError, match="uint8"):
        package.predict(color.astype(np.float32))
    with pytest.raises(ValueError, match="three channels"):
        package.predict(np.zeros((3, 4, 4), dtype=np.uint8))


def test_two_file_package_validates_schema_version_hash_and_no_sidecars(tmp_path: Path) -> None:
    directory = _package(tmp_path)
    metadata_path = directory / "deployment.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("decision")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required sections"):
        DeploymentPackage.load(directory, lambda _model: object())

    directory = _package(tmp_path / "version")
    metadata_path = directory / "deployment.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["deployment_contract_version"] = 2
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported deployment contract version"):
        DeploymentPackage.load(directory, lambda _model: object())

    directory = _package(tmp_path / "input")
    metadata_path = directory / "deployment.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["input"]["range"] = [0, 1]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="range 0..255"):
        DeploymentPackage.load(directory, lambda _model: object())

    directory = _package(tmp_path / "next")
    (directory / "model.pt").write_bytes(b"replaced")
    with pytest.raises(ValueError, match="SHA-256"):
        DeploymentPackage.load(directory, lambda _model: object())

    directory = _package(tmp_path / "third")
    (directory / "unexpected.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly model.pt and deployment.json"):
        DeploymentPackage.load(directory, lambda _model: object())

    directory = _package(tmp_path / "validation")
    metadata_path = directory / "deployment.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["validation"]["number_of_test_images"] = False
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="number_of_test_images must be positive"):
        DeploymentPackage.load(directory, lambda _model: object())

    directory = _package(tmp_path / "incomplete-validation")
    metadata_path = directory / "deployment.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["validation"] = {"status": "PASS"}
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="tested artifact"):
        DeploymentPackage.load(directory, lambda _model: object())


def test_two_file_package_preserves_superadd_native_decision_score_without_aliases(tmp_path: Path) -> None:
    directory = _package(tmp_path, model_id="super_add")
    received: list[np.ndarray] = []
    package = DeploymentPackage.load(
        directory,
        lambda _model: _superadd_inferencer(
            [{"pred_score": 1.0, "decision_score": 1.7, "anomaly_map": np.full((448, 448), 0.4, dtype=np.float32)}],
            received,
        ),
    )

    result = package.predict(np.zeros((3, 4, 3), dtype=np.uint8))

    assert result.decision_score == pytest.approx(1.7)
    assert result.is_ng
    assert result.score_semantic == SUPERADD_NATIVE_IMAGE_SCORE_SEMANTIC

    missing_score = DeploymentPackage.load(
        directory,
        lambda _model: _superadd_inferencer([{"pred_score": 1.0, "raw_pred_score": 1.7, "anomaly_map": np.full((448, 448), 0.4, dtype=np.float32)}], []),
    )
    with pytest.raises(ValueError, match="decision_score"):
        missing_score.predict(np.zeros((3, 4, 3), dtype=np.uint8))


def test_two_file_package_rejects_superadd_memory_bank_metadata_mismatch(tmp_path: Path) -> None:
    directory = _package(tmp_path, model_id="super_add")
    metadata_path = directory / "deployment.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["model"]["memory_bank"]["database_sizes"] = [2, 2]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="database_sizes"):
        DeploymentPackage.load(
            directory,
            lambda _model: _superadd_inferencer(
                [{"decision_score": 1.7, "anomaly_map": np.full((448, 448), 0.4, dtype=np.float32)}],
                [],
            ),
        )


def test_two_file_package_loads_after_only_model_and_metadata_are_copied(tmp_path: Path) -> None:
    source = _package(tmp_path)
    clean = tmp_path / "clean"
    clean.mkdir()
    for name in ("model.pt", "deployment.json"):
        shutil.copy2(source / name, clean / name)
    package = DeploymentPackage.load(
        clean,
        lambda _model: _inferencer([{"pred_score": 0.4, "anomaly_map": np.full((3, 4), 0.4, dtype=np.float32)}], []),
    )

    assert package.predict(np.zeros((3, 4), dtype=np.uint8)).is_ng is False


def test_two_file_package_clean_process_needs_only_the_two_deployment_files(tmp_path: Path) -> None:
    source = _package(tmp_path)
    clean = tmp_path / "offline"
    clean.mkdir()
    for name in ("model.pt", "deployment.json"):
        shutil.copy2(source / name, clean / name)
    script = """
from pathlib import Path
import numpy as np
from app.core.deployment_package import DeploymentPackage
class Inferencer:
    def predict(self, image):
        return {\"pred_score\": 0.4, \"anomaly_map\": np.full((3, 4), 0.4, dtype=np.float32)}
package = DeploymentPackage.load(Path(__import__(\"sys\").argv[1]), lambda _: Inferencer())
result = package.predict(np.zeros((3, 4), dtype=np.uint8))
assert result.is_ng is False and result.decision_score == 0.4
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    result = subprocess.run(
        [sys.executable, "-c", script, str(clean)],
        cwd=clean,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr