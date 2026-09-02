"""Tests for selectable trained-model export."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from app.core.dataset_manifest import sha256_file
from app.core.inspection_region import InspectionRegionProcessor, inspection_region_hash, write_inspection_region
from app.models.inspection_region import InspectionRegionConfig
from app.core.run_artifacts import CanonicalCheckpoint, write_run_manifest
from app.core.result_parser import ResultParser
from app.models.prediction_result import PredictionResult
from app.models.training_config import TrainingConfig
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

    def create_inference_components(self, config: TrainingConfig, output_directory: Path) -> dict[str, object]:
        return {"model": object(), "engine": self.engine}


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
            }
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
    (run_directory / "environment.json").write_text("{}", encoding="utf-8")
    (run_directory / "dataset_manifest.json").write_text("{}", encoding="utf-8")
    (run_directory / "calibration_manifest.json").write_text("{}", encoding="utf-8")
    (run_directory / "final_test_manifest.json").write_text("{}", encoding="utf-8")
    (run_directory / "predictions.csv").write_text("image_path\nfinal_test.png\n", encoding="utf-8")
    engine = FakeEngine()
    export_directory = tmp_path / "exports"
    received_thresholds: list[float] = []
    received_tolerances: list[tuple[str, float]] = []

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
            "tested_images": len(predictions),
            "decision_parity": 1.0,
            "threshold": threshold,
            "score_tolerance": score_tolerance,
            "maximum_score_delta": 0.0,
        }

    report = ExportService(
        FakeAnomalibService(engine),
        deployment_validator=deployment_validator,
    ).export_model(
        run_directory,
        export_directory,
        [ModelExportFormat.OPENVINO, ModelExportFormat.TORCH],
    )

    assert report.failures == {}
    assert [result.export_format for result in report.exported] == ["openvino", "torch"]
    assert all(len(result.sha256) == 64 for result in report.exported)
    assert all(result.validation_report and result.validation_report.is_file() for result in report.exported)
    assert report.package_directory and (report.package_directory / "deployment_manifest.json").is_file()
    assert (report.package_directory / "predictions.csv").is_file()
    assert (report.package_directory / "calibration_manifest.json").is_file()
    assert (report.package_directory / "final_test_manifest.json").is_file()
    assert (report.package_directory / "inspection_region.json").is_file()
    assert [result.exported_path.name for result in report.exported] == [
        "patchcore_2026_08_31_12_00_00_openvino.xml",
        "patchcore_2026_08_31_12_00_00_torch.pt",
    ]
    assert [call["input_size"] for call in engine.calls] == [None, None]
    assert [call["ckpt_path"] for call in engine.calls] == [checkpoint.resolve(), checkpoint.resolve()]
    assert received_thresholds == [0.5, 0.5]
    assert received_tolerances == [
        ("openvino", FORMAT_SCORE_TOLERANCES["openvino"]),
        ("torch", FORMAT_SCORE_TOLERANCES["torch"]),
    ]
    deployment_manifest = json.loads((report.package_directory / "deployment_manifest.json").read_text(encoding="utf-8"))
    assert deployment_manifest["threshold_metadata"]["threshold_method"] == "normal_only_conformal"
    assert deployment_manifest["threshold_metadata"]["threshold_revision"] == "revision-001"
    assert deployment_manifest["anomalib_version"] == "2.6.0"
    assert deployment_manifest["deployment_contract_version"] == DEPLOYMENT_CONTRACT_VERSION
    assert deployment_manifest["format_score_tolerances"] == FORMAT_SCORE_TOLERANCES
    assert deployment_manifest["inspection_preprocessing"]["metadata_sha256"] == inspection_region_hash(inspection_region)
    assert deployment_manifest["model"]["profile"]["preprocessing"] == "anomalib-native"
    assert deployment_manifest["exports"][0]["validation"]["maximum_score_delta"] == 0.0
    validation_report = json.loads(report.exported[0].validation_report.read_text(encoding="utf-8"))
    assert validation_report["decision_threshold"] == 0.5
    assert validation_report["deployment_contract_version"] == DEPLOYMENT_CONTRACT_VERSION
    assert validation_report["threshold_metadata"]["calibration_manifest_sha256"] == "b" * 64


def test_export_service_allows_an_explicit_per_format_tolerance_override() -> None:
    service = ExportService(score_tolerances={"openvino": 0.02})

    assert service.score_tolerances["openvino"] == 0.02
    assert service.score_tolerances["torch"] == FORMAT_SCORE_TOLERANCES["torch"]


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
