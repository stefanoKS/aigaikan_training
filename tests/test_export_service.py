"""Tests for selectable trained-model export."""

from __future__ import annotations

import json
from pathlib import Path

from app.models.training_config import TrainingConfig
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
    engine = FakeEngine()
    export_directory = tmp_path / "exports"

    report = ExportService(FakeAnomalibService(engine)).export_model(
        run_directory,
        export_directory,
        [ModelExportFormat.OPENVINO, ModelExportFormat.TORCH],
    )

    assert report.failures == {}
    assert [result.export_format for result in report.exported] == ["openvino", "torch"]
    assert [result.exported_path.name for result in report.exported] == [
        "patchcore_2026_08_31_12_00_00_openvino.xml",
        "patchcore_2026_08_31_12_00_00_torch.pt",
    ]
    assert [call["input_size"] for call in engine.calls] == [(480, 640), (480, 640)]
    assert [call["ckpt_path"] for call in engine.calls] == [checkpoint.resolve(), checkpoint.resolve()]