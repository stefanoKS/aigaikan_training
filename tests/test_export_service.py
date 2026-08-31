"""Tests for selectable trained-model export."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from app.core.dataset_manifest import sha256_file
from app.core.run_artifacts import CanonicalCheckpoint, write_run_manifest
from app.core.result_parser import ResultParser
from app.models.prediction_result import PredictionResult
from app.models.training_config import TrainingConfig
from app.models.training_run import TrainingRun
from app.services.export_service import ExportService, ModelExportFormat


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


def test_export_model_uses_configured_formats_dimensions_and_names(tmp_path: Path) -> None:
    """Each selected format must receive the run checkpoint and a descriptive model name."""
    run_directory = tmp_path / "2026-08-31_12-00-00_patchcore"
    checkpoint = run_directory / "weights" / "lightning" / "model.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("checkpoint", encoding="utf-8")
    config = TrainingConfig(model_name="PatchCore", image_width=640, image_height=480)
    (run_directory / "config.json").write_text(json.dumps(config.to_dict()), encoding="utf-8")
    write_run_manifest(
        run_directory / "run_manifest.json",
        canonical_checkpoint=CanonicalCheckpoint(checkpoint.resolve(), sha256_file(checkpoint)),
        dataset_manifest_sha256="a" * 64,
        split_counts={"final_test": {"ok": 1, "ng": 1}},
        threshold=0.5,
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
    (run_directory / "predictions.csv").write_text("image_path\nfinal_test.png\n", encoding="utf-8")
    engine = FakeEngine()
    export_directory = tmp_path / "exports"

    report = ExportService(
        FakeAnomalibService(engine),
        deployment_validator=lambda _path, _format, predictions, threshold: {
            "status": "PASS",
            "tested_images": len(predictions),
            "decision_parity": 1.0,
            "threshold": threshold,
        },
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
    assert [result.exported_path.name for result in report.exported] == [
        "patchcore_2026_08_31_12_00_00_openvino.xml",
        "patchcore_2026_08_31_12_00_00_torch.pt",
    ]
    assert [call["input_size"] for call in engine.calls] == [(480, 640), (480, 640)]
    assert [call["ckpt_path"] for call in engine.calls] == [checkpoint.resolve(), checkpoint.resolve()]


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