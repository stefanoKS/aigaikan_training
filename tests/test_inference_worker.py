"""Inference worker input and prediction-coverage tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core.run_artifacts import CanonicalCheckpoint, write_run_manifest
from app.core.inspection_region import inspection_region_hash, write_inspection_region
from app.models.inspection_region import InspectionRegionConfig
from app.models.training_config import TrainingConfig
from app.workers import inference_worker


def _write_run(run_directory: Path) -> Path:
    checkpoint = run_directory / "weights" / "model.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("checkpoint", encoding="utf-8")
    (run_directory / "config.json").write_text(json.dumps(TrainingConfig().to_dict()), encoding="utf-8")
    inspection_region = InspectionRegionConfig()
    write_inspection_region(run_directory / "inspection_region.json", inspection_region)
    write_run_manifest(
        run_directory / "run_manifest.json",
        canonical_checkpoint=CanonicalCheckpoint(checkpoint.resolve(), hashlib.sha256(b"checkpoint").hexdigest()),
        dataset_manifest_sha256="a" * 64,
        split_counts={"final_test": {"ok": 1, "ng": 1}},
        threshold=0.5,
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
    return checkpoint


def _fake_service(prediction_paths: list[Path]):
    class FakeEngine:
        def predict(self, **_kwargs):
            return [{"image_path": [str(path) for path in prediction_paths], "pred_score": [0.1] * len(prediction_paths)}]

    class FakeService:
        def create_inference_components(self, _config, _output_directory):
            return {
                "model": object(),
                "engine": FakeEngine(),
                "definition": SimpleNamespace(display_name="PatchCore"),
                "device": "cpu",
                "device_note": "",
            }

    return FakeService


def test_folder_inference_uses_anomalib_discovery_and_logs_selected_patchcore_run(tmp_path: Path, monkeypatch) -> None:
    run_directory = tmp_path / "run"
    checkpoint = _write_run(run_directory)
    input_directory = tmp_path / "input"
    input_directory.mkdir()
    first_image = input_directory / "first.png"
    nested_directory = input_directory / "nested"
    nested_directory.mkdir()
    second_image = nested_directory / "second.jpg"
    first_image.touch()
    second_image.touch()
    messages: list[dict[str, object]] = []
    monkeypatch.setattr(inference_worker, "AnomalibService", _fake_service([first_image, second_image]))
    monkeypatch.setattr(inference_worker, "_save_visualizations", lambda *_args: ("", ""))
    monkeypatch.setattr(inference_worker, "emit", messages.append)

    assert inference_worker.run(run_directory, input_directory) == 0

    log_messages = [str(message["message"]) for message in messages if message["type"] == "log"]
    assert any("Loaded PatchCore" in message and str(checkpoint) in message and "images=2" in message for message in log_messages)
    assert len([message for message in messages if message["type"] == "prediction"]) == 2


def test_folder_inference_fails_when_anomalib_skips_a_discovered_image(tmp_path: Path, monkeypatch) -> None:
    run_directory = tmp_path / "run"
    _write_run(run_directory)
    input_directory = tmp_path / "input"
    input_directory.mkdir()
    first_image = input_directory / "first.png"
    second_image = input_directory / "second.png"
    first_image.touch()
    second_image.touch()
    monkeypatch.setattr(inference_worker, "AnomalibService", _fake_service([first_image]))
    monkeypatch.setattr(inference_worker, "_save_visualizations", lambda *_args: ("", ""))
    monkeypatch.setattr(inference_worker, "emit", lambda _message: None)

    with pytest.raises(ValueError, match="produced 1 predictions for 2 input images"):
        inference_worker.run(run_directory, input_directory)


def test_inference_rejects_roi_metadata_hash_mismatch(tmp_path: Path, monkeypatch) -> None:
    run_directory = tmp_path / "run"
    _write_run(run_directory)
    manifest_path = run_directory / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["inspection_preprocessing"]["metadata_sha256"] = "b" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    image_path = tmp_path / "input.png"
    image_path.touch()
    monkeypatch.setattr(inference_worker, "emit", lambda _message: None)

    with pytest.raises(ValueError, match="metadata hash"):
        inference_worker.run(run_directory, image_path)


def test_enabled_roi_inference_uses_predict_dataset_with_the_saved_processor(tmp_path: Path, monkeypatch) -> None:
    run_directory = tmp_path / "run"
    checkpoint = _write_run(run_directory)
    roi = InspectionRegionConfig(
        enabled=True,
        source_width=64,
        source_height=64,
        points_px=((4, 4), (59, 4), (59, 59), (4, 59)),
    )
    write_inspection_region(run_directory / "inspection_region.json", roi)
    manifest_path = run_directory / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["inspection_preprocessing"].update(
        {
            "roi_contract_version": roi.roi_contract_version,
            "metadata_sha256": inspection_region_hash(roi),
            "source_size": [64, 64],
            "rectified_size": [55, 55],
        }
    )
    manifest["inspection_region_hash"] = inspection_region_hash(roi)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    image_path = tmp_path / "input.png"
    from PIL import Image

    Image.new("RGB", (64, 64), (10, 20, 30)).save(image_path)
    captured: dict[str, object] = {}

    class FakeEngine:
        def predict(self, **kwargs: object) -> list[dict[str, object]]:
            captured.update(kwargs)
            return [{"image_path": [str(image_path)], "pred_score": [0.1]}]

    class FakeService:
        def create_inference_components(self, _config: object, _output_directory: Path) -> dict[str, object]:
            return {
                "model": object(),
                "engine": FakeEngine(),
                "definition": SimpleNamespace(display_name="PatchCore"),
                "device": "cpu",
                "device_note": "",
            }

    monkeypatch.setattr(inference_worker, "AnomalibService", FakeService)
    monkeypatch.setattr(inference_worker, "_save_visualizations", lambda *_args: ("", ""))
    monkeypatch.setattr(inference_worker, "emit", lambda _message: None)

    assert inference_worker.run(run_directory, image_path) == 0

    assert "dataset" in captured
    assert "data_path" not in captured
    assert captured["ckpt_path"] == checkpoint
    assert captured["dataset"].transform.config == roi