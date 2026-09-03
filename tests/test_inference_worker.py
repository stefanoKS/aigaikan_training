"""Inference worker input and prediction-coverage tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace

import numpy as np
import pytest

from app.core.prediction_artifacts import PredictionArtifacts
from app.core.prediction_adapter import PostprocessedPredictionBatch, SUPERADD_NATIVE_IMAGE_SCORE_SEMANTIC
from app.core.preprocessing_contract import resolved_preprocessing_hash, write_resolved_preprocessing_plan
from app.core.result_parser import ResultParser
from app.core.run_artifacts import CanonicalCheckpoint, write_run_manifest
from app.core.threshold_contract import PixelThresholdOperatingPoint
from app.core.inspection_region import inspection_region_hash, write_inspection_region
from app.models.inspection_region import InspectionRegionConfig
from app.models.prediction_result import PredictionResult
from app.models.preprocessing_config import LEGACY_PREPROCESSING_CONTRACT_VERSION, PreprocessingConfig
from app.models.training_config import TrainingConfig
from app.models.training_run import TrainingRun
from app.workers import inference_worker


def _write_run(run_directory: Path, pixel_operating_point: PixelThresholdOperatingPoint | None = None) -> Path:
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
        } | ({"pixel_operating_point": pixel_operating_point.to_dict()} if pixel_operating_point is not None else {}),
    )
    return checkpoint


def _fake_service(prediction_paths: list[Path]):
    class FakeEngine:
        def __init__(self, callbacks: list[object]) -> None:
            self._callbacks = callbacks

        def predict(self, **_kwargs: object) -> None:
            for callback in self._callbacks:
                callback.write_on_batch_end(
                    None,
                    None,
                    {"image_path": [str(path) for path in prediction_paths], "pred_score": [0.1] * len(prediction_paths)},
                    None,
                    None,
                    0,
                    0,
                )

    class FakeService:
        def create_inference_components(self, _config, _output_directory, callbacks=None):
            return {
                "model": object(),
                "engine": FakeEngine(callbacks or []),
                "definition": SimpleNamespace(display_name="PatchCore"),
                "device": "cpu",
                "device_note": "",
            }

    return FakeService


def test_folder_inference_streams_explicit_eight_item_batches(tmp_path: Path, monkeypatch) -> None:
    run_directory = tmp_path / "run"
    _write_run(run_directory)
    input_directory = tmp_path / "input"
    input_directory.mkdir()
    source_paths = [input_directory / f"image_{index:02d}.png" for index in range(9)]
    from PIL import Image

    for source_path in source_paths:
        Image.new("RGB", (8, 8)).save(source_path)
    captured: dict[str, object] = {}

    class FakeEngine:
        def __init__(self, callbacks: list[object]) -> None:
            self._callbacks = callbacks

        def predict(self, **kwargs: object) -> None:
            captured.update(kwargs)
            for batch_index, source_batch in enumerate((source_paths[:8], source_paths[8:])):
                for callback in self._callbacks:
                    callback.write_on_batch_end(
                        None,
                        None,
                        {"image_path": [str(path) for path in source_batch], "pred_score": [0.1] * len(source_batch)},
                        None,
                        None,
                        batch_index,
                        0,
                    )

    class FakeService:
        def create_inference_components(self, _config, _output_directory, callbacks=None):
            return {
                "model": object(),
                "engine": FakeEngine(callbacks or []),
                "definition": SimpleNamespace(display_name="PatchCore"),
                "device": "cpu",
                "device_note": "",
            }

    messages: list[dict[str, object]] = []
    monkeypatch.setattr(inference_worker, "AnomalibService", FakeService)
    monkeypatch.setattr(inference_worker, "save_prediction_artifacts", lambda *_args, **_kwargs: PredictionArtifacts())
    monkeypatch.setattr(inference_worker, "emit", messages.append)

    assert inference_worker.run(run_directory, input_directory) == 0

    loader = captured["dataloaders"]
    assert loader.batch_size == 8
    assert "dataset" not in captured
    assert captured["return_predictions"] is False
    assert [message["current"] for message in messages if message["type"] == "progress"] == list(range(10))
    assert len([message for message in messages if message["type"] == "prediction"]) == 9


def test_folder_inference_uses_anomalib_discovery_and_logs_selected_patchcore_run(tmp_path: Path, monkeypatch) -> None:
    run_directory = tmp_path / "run"
    checkpoint = _write_run(run_directory)
    input_directory = tmp_path / "input"
    input_directory.mkdir()
    first_image = input_directory / "first.png"
    nested_directory = input_directory / "nested"
    nested_directory.mkdir()
    second_image = nested_directory / "second.jpg"
    from PIL import Image

    Image.new("RGB", (8, 8)).save(first_image)
    Image.new("RGB", (8, 8)).save(second_image)
    messages: list[dict[str, object]] = []
    monkeypatch.setattr(inference_worker, "AnomalibService", _fake_service([first_image, second_image]))
    monkeypatch.setattr(inference_worker, "save_prediction_artifacts", lambda *_args, **_kwargs: PredictionArtifacts())
    monkeypatch.setattr(inference_worker, "emit", messages.append)

    assert inference_worker.run(run_directory, input_directory) == 0

    log_messages = [str(message["message"]) for message in messages if message["type"] == "log"]
    assert any("Loaded PatchCore" in message and str(checkpoint) in message and "images=2" in message for message in log_messages)
    assert len([message for message in messages if message["type"] == "prediction"]) == 2


def test_folder_inference_uses_the_active_threshold_revision(tmp_path: Path, monkeypatch) -> None:
    run_directory = tmp_path / "run"
    _write_run(run_directory)
    revision_directory = run_directory / "threshold_revisions"
    revision_directory.mkdir()
    predictions_path = revision_directory / "threshold-001_predictions.csv"
    predictions_path.write_text("image_path\n", encoding="utf-8")
    revision_path = revision_directory / "threshold-001.json"
    revision_path.write_text(
        json.dumps(
            {
                "version": 1,
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
                "predictions_file": predictions_path.name,
            }
        ),
        encoding="utf-8",
    )
    (run_directory / "active_threshold_revision.json").write_text(
        json.dumps(
            {
                "version": 1,
                "revision_file": revision_path.name,
                "revision_sha256": hashlib.sha256(revision_path.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    input_path = tmp_path / "input.png"
    from PIL import Image

    Image.new("RGB", (8, 8)).save(input_path)
    messages: list[dict[str, object]] = []
    monkeypatch.setattr(inference_worker, "AnomalibService", _fake_service([input_path]))
    monkeypatch.setattr(inference_worker, "save_prediction_artifacts", lambda *_args, **_kwargs: PredictionArtifacts())
    monkeypatch.setattr(inference_worker, "emit", messages.append)

    assert inference_worker.run(run_directory, input_path) == 0

    prediction = next(message for message in messages if message["type"] == "prediction")
    assert prediction["threshold"] == 0.05
    assert prediction["predicted_label"] == "NG"


def test_folder_inference_uses_active_revision_quality_evidence(tmp_path: Path, monkeypatch) -> None:
    run_directory = tmp_path / "run"
    _write_run(run_directory)
    ResultParser().write_training_run(
        run_directory / "results.json",
        TrainingRun(
            run_name="run",
            run_dir=str(run_directory),
            model_name="PatchCore",
            device="cpu",
            predictions=[
                PredictionResult(
                    source_path="canonical_ok.png",
                    predicted_label="NG",
                    ground_truth_label="OK",
                    anomaly_score=0.8,
                    threshold=0.5,
                )
            ],
        ),
    )
    revision_directory = run_directory / "threshold_revisions"
    revision_directory.mkdir()
    predictions_path = revision_directory / "threshold-001_predictions.csv"
    ResultParser().export_predictions_csv(
        predictions_path,
        [
            PredictionResult(
                source_path="revision_ok.png",
                predicted_label="OK",
                ground_truth_label="OK",
                anomaly_score=0.1,
                threshold=0.5,
            )
        ],
    )
    revision_path = revision_directory / "threshold-001.json"
    revision_path.write_text(
        json.dumps(
            {
                "version": 1,
                "revision_id": "threshold-001",
                "image_operating_point": {
                    "version": 1,
                    "threshold": 0.5,
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
                "predictions_file": predictions_path.name,
            }
        ),
        encoding="utf-8",
    )
    (run_directory / "active_threshold_revision.json").write_text(
        json.dumps(
            {
                "version": 1,
                "revision_file": revision_path.name,
                "revision_sha256": hashlib.sha256(revision_path.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    input_path = tmp_path / "input.png"
    from PIL import Image

    Image.new("RGB", (8, 8)).save(input_path)
    messages: list[dict[str, object]] = []
    monkeypatch.setattr(inference_worker, "AnomalibService", _fake_service([input_path]))
    monkeypatch.setattr(inference_worker, "save_prediction_artifacts", lambda *_args, **_kwargs: PredictionArtifacts())
    monkeypatch.setattr(inference_worker, "emit", messages.append)

    assert inference_worker.run(run_directory, input_path) == 0
    assert any(message["type"] == "prediction" for message in messages)


def test_folder_inference_fails_when_anomalib_skips_a_discovered_image(tmp_path: Path, monkeypatch) -> None:
    run_directory = tmp_path / "run"
    _write_run(run_directory)
    input_directory = tmp_path / "input"
    input_directory.mkdir()
    first_image = input_directory / "first.png"
    second_image = input_directory / "second.png"
    from PIL import Image

    Image.new("RGB", (8, 8)).save(first_image)
    Image.new("RGB", (8, 8)).save(second_image)
    monkeypatch.setattr(inference_worker, "AnomalibService", _fake_service([first_image]))
    monkeypatch.setattr(inference_worker, "save_prediction_artifacts", lambda *_args, **_kwargs: PredictionArtifacts())
    monkeypatch.setattr(inference_worker, "emit", lambda _message: None)

    with pytest.raises(ValueError, match="produced 1 predictions for 2 input images"):
        inference_worker.run(run_directory, input_directory)


def test_folder_inference_warns_for_a_run_that_fails_final_test_false_reject_policy(tmp_path: Path, monkeypatch) -> None:
    run_directory = tmp_path / "run"
    _write_run(run_directory)
    ResultParser().write_training_run(
        run_directory / "results.json",
        TrainingRun(
            run_name="run",
            run_dir=str(run_directory),
            model_name="PatchCore",
            device="cpu",
            predictions=[
                PredictionResult(
                    source_path="known_ok.png",
                    predicted_label="NG",
                    ground_truth_label="OK",
                    anomaly_score=0.8,
                    threshold=0.5,
                )
            ],
        ),
    )
    input_path = tmp_path / "input.png"
    from PIL import Image

    Image.new("RGB", (8, 8)).save(input_path)
    messages: list[dict[str, object]] = []
    monkeypatch.setattr(inference_worker, "AnomalibService", _fake_service([input_path]))
    monkeypatch.setattr(inference_worker, "save_prediction_artifacts", lambda *_args, **_kwargs: PredictionArtifacts())
    monkeypatch.setattr(inference_worker, "emit", messages.append)

    assert inference_worker.run(run_directory, input_path) == 0
    assert any(
        message["type"] == "log"
        and message["level"] == "warning"
        and "False reject rate 1 exceeds the configured maximum" in message["message"]
        for message in messages
    )


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
        def __init__(self, callbacks: list[object]) -> None:
            self._callbacks = callbacks

        def predict(self, **kwargs: object) -> None:
            captured.update(kwargs)
            for callback in self._callbacks:
                callback.write_on_batch_end(
                    None, None, {"image_path": [str(image_path)], "pred_score": [0.1]}, None, None, 0, 0
                )

    class FakeService:
        def create_inference_components(self, _config: object, _output_directory: Path, callbacks=None) -> dict[str, object]:
            return {
                "model": object(),
                "engine": FakeEngine(callbacks or []),
                "definition": SimpleNamespace(display_name="PatchCore"),
                "device": "cpu",
                "device_note": "",
            }

    monkeypatch.setattr(inference_worker, "AnomalibService", FakeService)
    monkeypatch.setattr(inference_worker, "save_prediction_artifacts", lambda *_args, **_kwargs: PredictionArtifacts())
    monkeypatch.setattr(inference_worker, "emit", lambda _message: None)

    assert inference_worker.run(run_directory, image_path) == 0

    assert "dataloaders" in captured
    assert "data_path" not in captured
    assert captured["ckpt_path"] == checkpoint
    assert captured["dataloaders"].dataset.transform.config == roi
    assert captured["dataloaders"].batch_size == 8
    assert captured["return_predictions"] is False


def test_preprocessing_v2_inference_uses_prepared_geometry_and_reconstructed_source_map(tmp_path: Path, monkeypatch) -> None:
    run_directory = tmp_path / "run"
    _write_run(run_directory, PixelThresholdOperatingPoint(enabled=True, threshold=0.6))
    plan = PreprocessingConfig(
        preprocessing_contract_version=LEGACY_PREPROCESSING_CONTRACT_VERSION
    ).resolve("patchcore", (639, 177))
    write_resolved_preprocessing_plan(run_directory / "preprocessing_plan.json", plan)
    manifest_path = run_directory / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["preprocessing_contract"] = {
        "preprocessing_contract_version": 2,
        "metadata_file": "preprocessing_plan.json",
        "metadata_sha256": resolved_preprocessing_hash(plan),
        "project_policy_sha256": "a" * 64,
        "model_id": "patchcore",
        "model_input_size": [640, 192],
        "score_aggregation": "max",
        "tiled": False,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    source_path = tmp_path / "source.png"
    from PIL import Image

    Image.new("RGB", (639, 177), (10, 20, 30)).save(source_path)
    captured: dict[str, object] = {}

    class FakePredictDataset:
        def __init__(self, path: Path) -> None:
            self.path = Path(path)

        @property
        def collate_fn(self):
            return lambda items: items

    class FakeEngine:
        def __init__(self, callbacks: list[object]) -> None:
            self._callbacks = callbacks

        def predict(self, **kwargs: object) -> None:
            captured.update(kwargs)
            prepared_path = next(captured["dataloaders"].dataset.path.rglob("*.png"))
            captured["prepared_size"] = Image.open(prepared_path).size
            anomaly_map = np.zeros((192, 640), dtype=np.float32)
            anomaly_map[176, 638] = 0.7
            anomaly_map[191, 639] = 1.0
            for callback in self._callbacks:
                callback.write_on_batch_end(
                    None,
                    None,
                    {"image_path": [str(prepared_path)], "pred_score": [1.0], "anomaly_map": [anomaly_map]},
                    None,
                    None,
                    0,
                    0,
                )

    class FakeService:
        def create_inference_components(
            self, _config: object, _output_directory: Path, received_plan: object, callbacks=None
        ) -> dict[str, object]:
            assert received_plan == plan
            return {
                "model": object(),
                "engine": FakeEngine(callbacks or []),
                "definition": SimpleNamespace(display_name="PatchCore"),
                "device": "cpu",
                "device_note": "",
            }

    anomalib_data = ModuleType("anomalib.data")
    anomalib_data.PredictDataset = FakePredictDataset
    monkeypatch.setitem(sys.modules, "anomalib.data", anomalib_data)
    monkeypatch.setattr(inference_worker, "AnomalibService", FakeService)
    monkeypatch.setattr(inference_worker, "_discover_images", lambda _path: (source_path.resolve(),))
    messages: list[dict[str, object]] = []
    monkeypatch.setattr(inference_worker, "emit", messages.append)

    assert inference_worker.run(run_directory, source_path) == 0

    assert captured["prepared_size"] == (640, 192)
    prediction = next(message for message in messages if message["type"] == "prediction")
    assert prediction["anomaly_score"] == pytest.approx(0.7)
    assert prediction["predicted_label"] == "NG"
    assert prediction["threshold"] == pytest.approx(0.5)
    assert np.load(prediction["continuous_anomaly_map"])["anomaly_map"].shape == (177, 639)
    assert prediction["map_display_normalization"]["minimum"] == 0.0
    assert prediction["region_metadata"]["coordinate_space"] == "source_image"
    assert prediction["pixel_threshold"] == 0.6
    assert prediction["pixel_threshold_comparator"] == "greater_than_or_equal"
    assert np.asarray(Image.open(prediction["binary_mask"])).max() == 255
    inference_manifest = json.loads(
        next((run_directory / "inference").glob("*/inference_manifest.json")).read_text(encoding="utf-8")
    )
    assert inference_manifest["decision_threshold"] == pytest.approx(0.5)
    assert inference_manifest["decision_threshold_source"] == "run_manifest"


def test_superadd_folder_inference_uses_native_raw_score_not_saturated_postprocessing(tmp_path: Path, monkeypatch) -> None:
    run_directory = tmp_path / "run"
    _write_run(run_directory)
    plan = PreprocessingConfig().resolve("super_add", (639, 177))
    write_resolved_preprocessing_plan(run_directory / "preprocessing_plan.json", plan)
    (run_directory / "config.json").write_text(
        json.dumps(TrainingConfig(model_name="super_add").to_dict()),
        encoding="utf-8",
    )
    manifest_path = run_directory / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["threshold"] = 0.65
    manifest["threshold_metadata"] = {
        "threshold_value": 0.65,
        "threshold_raw": 0.65,
        "threshold_deployed": 0.65,
        "score_semantic": SUPERADD_NATIVE_IMAGE_SCORE_SEMANTIC,
        "decision_comparator": "greater_than_or_equal",
    }
    manifest["preprocessing_contract"] = {
        "preprocessing_contract_version": 3,
        "metadata_file": "preprocessing_plan.json",
        "metadata_sha256": resolved_preprocessing_hash(plan),
        "project_policy_sha256": "a" * 64,
        "model_id": "super_add",
        "model_input_size": [640, 448],
        "score_aggregation": "max",
        "tiled": False,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    source_path = tmp_path / "source.png"
    from PIL import Image

    Image.new("RGB", (639, 177), (10, 20, 30)).save(source_path)

    class FakePredictDataset:
        def __init__(self, path: Path) -> None:
            self.path = Path(path)

        @property
        def collate_fn(self):
            return lambda items: items

    class FakeEngine:
        def __init__(self, callbacks: list[object]) -> None:
            self._callbacks = callbacks

        def predict(self, **kwargs: object) -> None:
            prepared_path = next(kwargs["dataloaders"].dataset.path.rglob("*.png"))
            postprocessed_map = np.full((448, 640), 0.4, dtype=np.float32)
            raw_map = np.full((448, 640), 0.6, dtype=np.float32)
            output = PostprocessedPredictionBatch(
                {"image_path": [str(prepared_path)], "pred_score": [1.0], "anomaly_map": [postprocessed_map]},
                (0.6,),
                (raw_map,),
            )
            for callback in self._callbacks:
                callback.write_on_batch_end(None, None, output, None, None, 0, 0)

    class FakeService:
        def create_inference_components(self, _config, _output_directory, received_plan, callbacks=None):
            assert received_plan == plan
            return {
                "model": object(),
                "engine": FakeEngine(callbacks or []),
                "definition": SimpleNamespace(display_name="SuperADD"),
                "device": "cpu",
                "device_note": "",
            }

    anomalib_data = ModuleType("anomalib.data")
    anomalib_data.PredictDataset = FakePredictDataset
    monkeypatch.setitem(sys.modules, "anomalib.data", anomalib_data)
    monkeypatch.setattr(inference_worker, "AnomalibService", FakeService)
    monkeypatch.setattr(inference_worker, "_discover_images", lambda _path: (source_path.resolve(),))
    messages: list[dict[str, object]] = []
    monkeypatch.setattr(inference_worker, "emit", messages.append)

    assert inference_worker.run(run_directory, source_path) == 0

    prediction = next(message for message in messages if message["type"] == "prediction")
    assert prediction["anomaly_score"] == pytest.approx(0.6)
    assert prediction["score_semantic"] == SUPERADD_NATIVE_IMAGE_SCORE_SEMANTIC
    assert prediction["postprocessed_image_score"] == pytest.approx(1.0)
    assert prediction["predicted_label"] == "OK"