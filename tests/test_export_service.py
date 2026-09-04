"""Tests for selectable trained-model export."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import numpy as np
import pytest
import torch

from app.core.dataset_manifest import sha256_file
from app.core.deployment_package import DeploymentPrediction
from app.core.inspection_region import InspectionRegionProcessor, inspection_region_hash, write_inspection_region
from app.core.model_registry import ModelRegistry, ModelSupportLevel
from app.core.preprocessing_contract import resolved_preprocessing_hash, write_resolved_preprocessing_plan
from app.core.preprocessing_pipeline import PreprocessingPipeline
from app.models.inspection_region import InspectionRegionConfig
from app.core.run_artifacts import CanonicalCheckpoint, read_canonical_checkpoint, write_run_manifest
from app.core.result_parser import ResultParser
from app.models.prediction_result import PredictionResult
from app.models.training_config import TrainingConfig
from app.models.preprocessing_config import LEGACY_PREPROCESSING_CONTRACT_VERSION, PreprocessingConfig, TilingConfig
from app.models.training_run import TrainingRun
from app.services.export_service import DEPLOYMENT_CONTRACT_VERSION, FORMAT_SCORE_TOLERANCES, ExportService, ModelExportFormat


class FakeEngine:
    """Write placeholder exported artifacts while recording Anomalib API calls."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def export(self, **kwargs: object) -> Path:
        self.calls.append(kwargs)
        suffix = {"openvino": ".xml", "onnx": ".onnx", "torch": ".pt"}[str(kwargs["export_type"])]
        exported_path = Path(kwargs["export_root"]) / f"{kwargs['model_file_name']}{suffix}"
        exported_path.write_text("exported", encoding="utf-8")
        if kwargs["export_type"] == "openvino":
            exported_path.with_suffix(".bin").write_text("weights", encoding="utf-8")
        return exported_path


class FakeAnomalibService:
    """Provide the inference components required by the export boundary."""

    def __init__(self, engine: FakeEngine) -> None:
        self.engine = engine

    def create_inference_components(
        self,
        config: TrainingConfig,
        output_directory: Path,
        preprocessing_plan: object | None = None,
    ) -> dict[str, object]:
        return {"model": object(), "engine": self.engine}


def test_superadd_memory_bank_metadata_comes_from_completed_checkpoint_state(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "superadd.ckpt"
    bank = torch.zeros((2, 3, 4), dtype=torch.float16)
    torch.save({"state_dict": {"model.model.memory_bank": bank}}, checkpoint_path)

    metadata = ExportService._superadd_memory_bank_metadata_from_checkpoint(checkpoint_path)

    assert metadata == {
        "bank_count": 2,
        "feature_dimension": 4,
        "database_sizes": [3, 3],
        "dtype": "torch.float16",
    }

    for state_dict, error in (
        ({}, "exactly one"),
        (
            {"model.first.memory_bank": bank, "model.second.memory_bank": bank},
            "exactly one",
        ),
        ({"model.model.memory_bank": torch.zeros((2, 0, 4))}, "non-empty"),
    ):
        torch.save({"state_dict": state_dict}, checkpoint_path)
        with pytest.raises(ValueError, match=error):
            ExportService._superadd_memory_bank_metadata_from_checkpoint(checkpoint_path)


def test_export_model_uses_configured_formats_native_preprocessing_and_names(tmp_path: Path) -> None:
    """Each selected format must receive the run checkpoint and a descriptive model name."""
    run_directory = tmp_path / "2026-08-31_12-00-00_patchcore"
    checkpoint = run_directory / "weights" / "lightning" / "model.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("checkpoint", encoding="utf-8")
    config = TrainingConfig(model_name="PatchCore")
    (run_directory / "config.json").write_text(json.dumps(config.to_dict()), encoding="utf-8")
    inspection_region = InspectionRegionConfig()
    write_inspection_region(run_directory / "inspection_region.json", inspection_region)
    preprocessing_plan = PreprocessingConfig().resolve("patchcore", (4, 3))
    write_resolved_preprocessing_plan(run_directory / "preprocessing_plan.json", preprocessing_plan)
    write_run_manifest(
        run_directory / "run_manifest.json",
        canonical_checkpoint=CanonicalCheckpoint(checkpoint.resolve(), sha256_file(checkpoint)),
        dataset_manifest_sha256="a" * 64,
        split_counts={"final_test": {"ok": 1, "ng": 1}},
        threshold=0.5,
        threshold_metadata={
            "threshold_value": 0.5,
            "threshold_method": "normal_only_conformal",
            "threshold_revision": "revision-001",
            "calibration_manifest_sha256": "b" * 64,
        },
        extra={
            "inspection_region_hash": inspection_region_hash(inspection_region),
            "inspection_preprocessing": {
                "roi_contract_version": inspection_region.roi_contract_version,
                "metadata_file": "inspection_region.json",
                "metadata_sha256": inspection_region_hash(inspection_region),
                "source_size": [0, 0],
                "rectified_size": [0, 0],
            },
            "preprocessing_contract": {
                "preprocessing_contract_version": preprocessing_plan.preprocessing_contract_version,
                "metadata_file": "preprocessing_plan.json",
                "metadata_sha256": resolved_preprocessing_hash(preprocessing_plan),
                "model_id": preprocessing_plan.model_id,
                "model_input_size": list(preprocessing_plan.model_input_size),
                "score_aggregation": preprocessing_plan.score_aggregation.value,
                "tiled": preprocessing_plan.tiled,
            },
        },
    )
    ResultParser().write_training_run(
        run_directory / "results.json",
        TrainingRun(
            run_name=run_directory.name,
            run_dir=str(run_directory),
            model_name="PatchCore",
            device="cpu",
            predictions=[
                PredictionResult(
                    source_path="final_test.png",
                    predicted_label="OK",
                    ground_truth_label="OK",
                    anomaly_score=0.1,
                    threshold=0.5,
                    dataset_role="final_test_ok",
                )
            ],
        ),
    )
    revision_directory = run_directory / "threshold_revisions"
    revision_directory.mkdir()
    revision_predictions = revision_directory / "threshold-001_predictions.csv"
    ResultParser().export_predictions_csv(
        revision_predictions,
        [
            PredictionResult(
                source_path="final_test.png",
                predicted_label="NG",
                ground_truth_label="OK",
                anomaly_score=0.1,
                threshold=0.05,
                dataset_role="final_test_ok",
                score_semantic="anomalib_postprocessed_pred_score_v1",
            )
        ],
    )
    revision_path = revision_directory / "threshold-001.json"
    revision_path.write_text(
        json.dumps(
            {
                "version": 2,
                "revision_id": "threshold-001",
                "image_operating_point": {
                    "version": 1,
                    "threshold": 0.05,
                    "comparator": "greater_than_or_equal",
                    "score_semantic": "anomalib_postprocessed_pred_score_v1",
                },
                "pixel_operating_point": {
                    "version": 1,
                    "enabled": False,
                    "threshold": None,
                    "comparator": "greater_than_or_equal",
                    "semantic": "continuous_anomaly_map_gte_v1",
                },
                "predictions_file": revision_predictions.name,
                "source": "operator_override",
                "base_calibrated_threshold": 0.5,
                "created_at": "2026-09-04T00:00:00+00:00",
                "operator_note": "inference page adjustment",
            }
        ),
        encoding="utf-8",
    )
    (run_directory / "active_threshold_revision.json").write_text(
        json.dumps(
            {
                "version": 1,
                "revision_file": revision_path.name,
                "revision_sha256": sha256_file(revision_path),
            }
        ),
        encoding="utf-8",
    )
    (run_directory / "environment.json").write_text("{}", encoding="utf-8")
    (run_directory / "dataset_manifest.json").write_text("{}", encoding="utf-8")
    (run_directory / "calibration_manifest.json").write_text("{}", encoding="utf-8")
    (run_directory / "final_test_manifest.json").write_text("{}", encoding="utf-8")
    (run_directory / "predictions.csv").write_text("image_path\nfinal_test.png\n", encoding="utf-8")
    engine = FakeEngine()
    export_directory = tmp_path / "exports"
    received_thresholds: list[float] = []
    received_tolerances: list[tuple[str, float]] = []
    verified_definition = replace(
        ModelRegistry().get("patchcore"),
        supports_export=True,
        support_level=ModelSupportLevel.TORCH_EXPORT_VALIDATED,
    )
    verified_registry = type("VerifiedRegistry", (), {"get": lambda _self, _model_name: verified_definition})()

    def deployment_validator(
        _path: Path,
        _format: str,
        predictions: list[PredictionResult],
        threshold: float,
        score_tolerance: float,
    ) -> dict[str, object]:
        received_thresholds.append(threshold)
        received_tolerances.append((_format, score_tolerance))
        return {
            "status": "PASS",
            "number_of_test_images": len(predictions),
            "decision_match_rate": 1.0,
            "score_tolerance": score_tolerance,
            "map_tolerance": score_tolerance,
            "max_abs_score_error": 0.0,
            "mean_abs_map_error": 0.0,
            "max_abs_map_error": 0.0,
            "artifact": "model.pt",
            "decision_threshold": threshold,
        }

    report = ExportService(
        FakeAnomalibService(engine),
        deployment_validator=deployment_validator,
        model_registry=verified_registry,
    ).export_model(
        run_directory,
        export_directory,
        [ModelExportFormat.TORCH],
    )

    assert report.failures == {}
    assert [result.export_format for result in report.exported] == ["torch"]
    assert all(len(result.sha256) == 64 for result in report.exported)
    assert report.package_directory is not None
    assert {path.name for path in report.package_directory.iterdir()} == {"model.pt", "deployment.json"}
    assert [result.exported_path.name for result in report.exported] == ["model.pt"]
    assert [call["input_size"] for call in engine.calls] == [None]
    assert [call["ckpt_path"] for call in engine.calls] == [checkpoint.resolve()]
    assert received_thresholds == [0.05]
    assert received_tolerances == [
        ("torch", FORMAT_SCORE_TOLERANCES["torch"]),
    ]
    deployment = json.loads((report.package_directory / "deployment.json").read_text(encoding="utf-8"))
    assert deployment["deployment_contract_version"] == DEPLOYMENT_CONTRACT_VERSION
    assert deployment["deployment"]["model_sha256"] == report.exported[0].sha256
    assert deployment["input"] == {
        "dtype": "uint8",
        "range": [0, 255],
        "accepted_layouts": ["HW", "HWC"],
        "canonical_color_order": "RGB",
        "color_input_order": "RGB",
        "mono_conversion": "GRAY_TO_RGB",
    }
    assert deployment["inspection_region"] == inspection_region.to_dict()
    assert deployment["image_preprocessing"] == preprocessing_plan.image_preprocessing.to_dict()
    assert deployment["model_preprocessing"]["resolved_plan"] == preprocessing_plan.to_dict()
    assert deployment["decision"]["threshold"] == 0.05
    assert deployment["decision"]["score_semantic"] == "anomalib_postprocessed_pred_score_v1"
    assert deployment["decision"]["threshold_revision_id"] == "threshold-001"
    assert deployment["decision"]["comparator"] == ">="
    assert deployment["decision"]["operator_note"] == "inference page adjustment"
    assert deployment["validation"]["status"] == "PASS"

    revised_package = ExportService().create_deployment_policy_revision(
        report.package_directory,
        export_directory,
        1.7,
        "line override",
    )
    revised_deployment = json.loads((revised_package / "deployment.json").read_text(encoding="utf-8"))
    assert {path.name for path in revised_package.iterdir()} == {"model.pt", "deployment.json"}
    assert revised_deployment["decision"]["threshold"] == 1.7
    assert revised_deployment["decision"]["threshold_source"] == "operator_override"
    assert revised_deployment["deployment"]["model_sha256"] == deployment["deployment"]["model_sha256"]
    assert revised_deployment["validation"]["decision_threshold"] == deployment["validation"]["decision_threshold"] == 0.05
    assert revised_deployment["model"]["profile"]["preprocessing"] == "anomalib-native"


def test_export_service_allows_an_explicit_per_format_tolerance_override() -> None:
    service = ExportService(score_tolerances={"openvino": 0.02})

    assert service.score_tolerances["openvino"] == 0.02
    assert service.score_tolerances["torch"] == FORMAT_SCORE_TOLERANCES["torch"]


def test_two_file_export_rejects_non_torch_format_selection(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="two-file deployment contract supports only one Torch"):
        ExportService().export_model(tmp_path, tmp_path / "exports", [ModelExportFormat.TORCH, ModelExportFormat.ONNX])


def test_two_file_parity_validation_compares_score_map_and_decision(tmp_path: Path, monkeypatch) -> None:
    from PIL import Image

    source = tmp_path / "input.png"
    Image.new("RGB", (4, 3), (10, 20, 30)).save(source)
    map_path = tmp_path / "map.npz"
    expected_map = np.full((3, 4), 0.4, dtype=np.float32)
    np.savez_compressed(map_path, anomaly_map=expected_map)
    prediction = PredictionResult(
        source_path=str(source),
        predicted_label="OK",
        ground_truth_label="OK",
        anomaly_score=0.4,
        threshold=0.5,
        score_semantic="anomalib_postprocessed_pred_score_v1",
        continuous_anomaly_map=str(map_path),
    )

    class FakePackage:
        def predict(self, frame: np.ndarray) -> DeploymentPrediction:
            assert frame.dtype == np.uint8 and frame.shape == (3, 4, 3)
            return DeploymentPrediction(0.4, 0.5, False, expected_map.copy(), "anomalib_postprocessed_pred_score_v1")

    monkeypatch.setattr("app.services.export_service.DeploymentPackage.load", lambda *_args, **_kwargs: FakePackage())
    report = ExportService._validate_two_file_deployment(tmp_path, [prediction], 0.5, 1e-4)

    assert report["status"] == "PASS"
    assert report["max_abs_score_error"] == pytest.approx(0.0)
    assert report["mean_abs_map_error"] == pytest.approx(0.0)
    assert report["max_abs_map_error"] == pytest.approx(0.0)
    assert report["decision_match_rate"] == pytest.approx(1.0)
    assert report["number_of_test_images"] == 1


def test_deployment_validation_rejects_any_final_test_decision_mismatch(monkeypatch) -> None:
    class FakeTorchInferencer:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def predict(self, _source_path: str) -> dict[str, float]:
            return {"pred_score": 0.8}

    anomalib_deploy = ModuleType("anomalib.deploy")
    anomalib_deploy.TorchInferencer = FakeTorchInferencer
    monkeypatch.setitem(sys.modules, "anomalib.deploy", anomalib_deploy)
    expected = PredictionResult(
        source_path="final_test_ok.png",
        predicted_label="OK",
        ground_truth_label="OK",
        anomaly_score=0.1,
        threshold=0.5,
        dataset_role="final_test_ok",
    )

    with pytest.raises(RuntimeError, match="decision parity failed"):
        ExportService._validate_deployment(Path("model.pt"), "torch", [expected], threshold=0.5)


def test_deployment_validation_passes_rectified_roi_array_to_the_exported_model(tmp_path: Path, monkeypatch) -> None:
    received_images: list[object] = []

    class FakeTorchInferencer:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def predict(self, image: object) -> dict[str, float]:
            received_images.append(image)
            return {"pred_score": 0.1}

    anomalib_deploy = ModuleType("anomalib.deploy")
    anomalib_deploy.TorchInferencer = FakeTorchInferencer
    monkeypatch.setitem(sys.modules, "anomalib.deploy", anomalib_deploy)
    source_path = tmp_path / "final_test_ok.png"
    from PIL import Image

    Image.new("RGB", (64, 64), (20, 30, 40)).save(source_path)
    processor = InspectionRegionProcessor(
        InspectionRegionConfig(
            enabled=True,
            source_width=64,
            source_height=64,
            points_px=((4, 4), (59, 4), (59, 59), (4, 59)),
        )
    )
    expected = PredictionResult(
        source_path=str(source_path),
        predicted_label="OK",
        ground_truth_label="OK",
        anomaly_score=0.1,
        threshold=0.5,
        dataset_role="final_test_ok",
    )

    ExportService._validate_deployment(
        Path("model.pt"),
        "torch",
        [expected],
        threshold=0.5,
        inspection_processor=processor,
    )

    assert len(received_images) == 1
    assert getattr(received_images[0], "shape") == (55, 55, 3)


def test_deployment_validation_rejects_excessive_score_difference(monkeypatch) -> None:
    class FakeTorchInferencer:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def predict(self, _source_path: str) -> dict[str, float]:
            return {"pred_score": 0.2}

    anomalib_deploy = ModuleType("anomalib.deploy")
    anomalib_deploy.TorchInferencer = FakeTorchInferencer
    monkeypatch.setitem(sys.modules, "anomalib.deploy", anomalib_deploy)
    expected = PredictionResult(
        source_path="final_test_ok.png",
        predicted_label="OK",
        ground_truth_label="OK",
        anomaly_score=0.1,
        threshold=0.5,
        dataset_role="final_test_ok",
    )

    with pytest.raises(RuntimeError, match="score parity failed"):
        ExportService._validate_deployment(
            Path("model.pt"),
            "torch",
            [expected],
            threshold=0.5,
            score_tolerance=0.01,
        )


def test_deployment_validation_uses_the_v3_reconstructed_map_score(tmp_path: Path, monkeypatch) -> None:
    received_images: list[object] = []

    class FakeTorchInferencer:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def predict(self, image: object) -> dict[str, object]:
            received_images.append(image)
            anomaly_map = np.full((192, 336), 0.9, dtype=np.float32)
            return {"pred_score": 0.1, "anomaly_map": anomaly_map}

    anomalib_deploy = ModuleType("anomalib.deploy")
    anomalib_deploy.TorchInferencer = FakeTorchInferencer
    monkeypatch.setitem(sys.modules, "anomalib.deploy", anomalib_deploy)
    source_path = tmp_path / "final_test_ok.png"
    from PIL import Image

    Image.new("RGB", (639, 177), (20, 30, 40)).save(source_path)
    pipeline = PreprocessingPipeline(
        InspectionRegionConfig(),
        PreprocessingConfig(tiling=TilingConfig(enabled=True)).resolve("dinomaly_dinov3", (639, 177)),
    )
    expected = PredictionResult(
        source_path=str(source_path),
        predicted_label="NG",
        ground_truth_label="OK",
        anomaly_score=0.9,
        threshold=0.8,
        dataset_role="final_test_ok",
    )

    report = ExportService._validate_deployment(
        Path("model.pt"),
        "torch",
        [expected],
        threshold=0.8,
        preprocessing_pipeline=pipeline,
    )

    assert report["status"] == "PASS"
    assert getattr(received_images[0], "shape") == (*reversed(pipeline.plan.model_input_size), 3)
    assert len(received_images) == 3


def test_deployment_validation_uses_non_tiled_native_score_not_anomaly_map(tmp_path: Path, monkeypatch) -> None:
    class FakeTorchInferencer:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def predict(self, _image: object) -> dict[str, object]:
            return {"pred_score": 0.3, "anomaly_map": np.full((5, 7), 0.9, dtype=np.float32)}

    anomalib_deploy = ModuleType("anomalib.deploy")
    anomalib_deploy.TorchInferencer = FakeTorchInferencer
    monkeypatch.setitem(sys.modules, "anomalib.deploy", anomalib_deploy)
    source_path = tmp_path / "final_test_ok.png"
    from PIL import Image

    Image.new("RGB", (7, 5), (20, 30, 40)).save(source_path)
    pipeline = PreprocessingPipeline(InspectionRegionConfig(), PreprocessingConfig().resolve("patchcore", (7, 5)))
    expected = PredictionResult(
        source_path=str(source_path),
        predicted_label="OK",
        ground_truth_label="OK",
        anomaly_score=0.3,
        threshold=0.5,
        score_semantic="anomalib_postprocessed_pred_score_v1",
        dataset_role="final_test_ok",
    )

    report = ExportService._validate_deployment(
        Path("model.pt"),
        "torch",
        [expected],
        threshold=0.5,
        preprocessing_pipeline=pipeline,
        threshold_semantic="anomalib_postprocessed_pred_score_v1",
    )

    assert report["status"] == "PASS"


def test_deployment_validation_uses_superadd_raw_native_score_and_rejects_semantic_mismatch(tmp_path: Path, monkeypatch) -> None:
    class FakeTorchInferencer:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def predict(self, _image: object) -> dict[str, object]:
            return {
                "pred_score": 1.0,
                "decision_score": 1.7,
                "anomaly_map": np.full((448, 448), 0.4, dtype=np.float32),
            }

    anomalib_deploy = ModuleType("anomalib.deploy")
    anomalib_deploy.TorchInferencer = FakeTorchInferencer
    monkeypatch.setitem(sys.modules, "anomalib.deploy", anomalib_deploy)
    source_path = tmp_path / "final_test_ng.png"
    from PIL import Image

    Image.new("RGB", (3, 2), (20, 30, 40)).save(source_path)
    pipeline = PreprocessingPipeline(InspectionRegionConfig(), PreprocessingConfig().resolve("super_add", (3, 2)))
    expected = PredictionResult(
        source_path=str(source_path),
        predicted_label="NG",
        ground_truth_label="NG",
        anomaly_score=1.7,
        threshold=1.5,
        score_semantic="superadd_native_top_quantile_score_v1",
        dataset_role="final_test_ng",
    )

    assert ExportService._validate_deployment(
        Path("model.pt"),
        "torch",
        [expected],
        threshold=1.5,
        preprocessing_pipeline=pipeline,
        threshold_semantic="superadd_native_top_quantile_score_v1",
    )["status"] == "PASS"
    with pytest.raises(ValueError, match="does not match"):
        ExportService._validate_deployment(
            Path("model.pt"),
            "torch",
            [expected],
            threshold=1.5,
            preprocessing_pipeline=pipeline,
            threshold_semantic="anomalib_postprocessed_pred_score_v1",
        )


def test_deployment_validation_preserves_legacy_v2_map_score_semantics(tmp_path: Path, monkeypatch) -> None:
    class FakeTorchInferencer:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def predict(self, _image: object) -> dict[str, object]:
            anomaly_map = np.zeros((192, 640), dtype=np.float32)
            anomaly_map[176, 638] = 0.7
            anomaly_map[191, 639] = 1.0
            return {"pred_score": 1.0, "anomaly_map": anomaly_map}

    anomalib_deploy = ModuleType("anomalib.deploy")
    anomalib_deploy.TorchInferencer = FakeTorchInferencer
    monkeypatch.setitem(sys.modules, "anomalib.deploy", anomalib_deploy)
    source_path = tmp_path / "final_test_ok.png"
    from PIL import Image

    Image.new("RGB", (639, 177), (20, 30, 40)).save(source_path)
    pipeline = PreprocessingPipeline(
        InspectionRegionConfig(),
        PreprocessingConfig(preprocessing_contract_version=LEGACY_PREPROCESSING_CONTRACT_VERSION).resolve(
            "patchcore", (639, 177)
        ),
    )
    expected = PredictionResult(
        source_path=str(source_path),
        predicted_label="NG",
        ground_truth_label="OK",
        anomaly_score=0.7,
        threshold=0.6,
        dataset_role="final_test_ok",
    )

    report = ExportService._validate_deployment(
        Path("model.pt"),
        "torch",
        [expected],
        threshold=0.6,
        preprocessing_pipeline=pipeline,
    )

    assert report["status"] == "PASS"


def test_export_rejects_models_without_a_validated_deployment_format(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps(TrainingConfig(model_name="anomaly_dino").to_dict()), encoding="utf-8")

    with pytest.raises(ValueError, match="export is unavailable"):
        ExportService().export_model(tmp_path, tmp_path / "exports", [ModelExportFormat.TORCH])
