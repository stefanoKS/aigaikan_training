"""Tests for training worker progress reporting and calibration data isolation."""

from pathlib import Path
import pickle

import numpy as np
from PIL import Image

from app.core.preprocessing_pipeline import PreprocessingPipeline
from app.models.inspection_region import InspectionRegionConfig
from app.models.preprocessing_config import LEGACY_PREPROCESSING_CONTRACT_VERSION, PreprocessingConfig
from app.workers.training_worker import (
    TrainingProgressReporter,
    _final_test_predictions,
    _peak_gpu_memory_mb,
    calibration_samples_from_predictions,
    configure_worker_stdio,
    create_training_progress_callback,
    emit,
)


class _FakeTrainer:
    num_training_batches = 4
    num_val_batches = [2]
    num_test_batches = [3]


def test_progress_callback_reports_batch_counts() -> None:
    """Training, validation, and test callbacks emit deterministic stage updates."""
    messages: list[dict[str, object]] = []
    callback = TrainingProgressReporter(messages.append)
    trainer = _FakeTrainer()

    callback.on_train_epoch_start(trainer, None)
    callback.on_train_batch_end(trainer, None, None, None, 1)
    callback.on_validation_epoch_start(trainer, None)
    callback.on_validation_batch_end(trainer, None, None, None, 0)
    callback.on_test_epoch_start(trainer, None)
    callback.on_test_batch_end(trainer, None, None, None, 2)

    assert messages == [
        {"type": "stage", "name": "Training model"},
        {"type": "stage_progress", "current": 0, "total": 4},
        {"type": "stage_progress", "current": 2, "total": 4},
        {"type": "stage", "name": "Calibrating model"},
        {"type": "stage_progress", "current": 0, "total": 2},
        {"type": "stage_progress", "current": 1, "total": 2},
        {"type": "stage", "name": "Evaluating test images"},
        {"type": "stage_progress", "current": 0, "total": 3},
        {"type": "stage_progress", "current": 3, "total": 3},
    ]


def test_lightning_progress_callback_is_pickleable_for_checkpoint_saves() -> None:
    callback = create_training_progress_callback(emit)

    restored = pickle.loads(pickle.dumps(callback))

    assert type(restored).__module__ == "app.workers.training_worker"


def test_calibration_samples_reject_final_test_predictions(tmp_path: Path) -> None:
    staged_path = (tmp_path / "final_test_ng" / "item.png").resolve()
    output = {"image_path": [str(staged_path)], "pred_score": [0.9], "anomaly_map": [None]}

    try:
        calibration_samples_from_predictions(output, {staged_path: tmp_path / "source.png"})
    except ValueError as exc:
        assert "unexpected staged role" in str(exc)
    else:
        raise AssertionError("Final-test predictions must never be used for threshold calibration")


def test_cpu_prediction_does_not_claim_gpu_peak_memory() -> None:
    assert _peak_gpu_memory_mb("cpu") is None


def test_worker_stdio_configuration_is_safe_under_pytest_capture() -> None:
    configure_worker_stdio()


def test_preprocessing_v2_calibration_and_final_test_use_the_same_valid_only_source_score(tmp_path: Path) -> None:
    source_path = (tmp_path / "source.png").resolve()
    validation_path = (tmp_path / "validation_ok" / "tile.png").resolve()
    final_test_path = (tmp_path / "final_test_ng" / "tile.png").resolve()
    pipeline = PreprocessingPipeline(
        InspectionRegionConfig(),
        PreprocessingConfig(
            preprocessing_contract_version=LEGACY_PREPROCESSING_CONTRACT_VERSION
        ).resolve("dinomaly_dinov3", (639, 177)),
    )
    anomaly_map = np.zeros((192, 640), dtype=np.float32)
    anomaly_map[176, 638] = 0.7
    anomaly_map[191, 639] = 1.0
    tile = pipeline.plan.tiles[0]

    calibration_samples = calibration_samples_from_predictions(
        {"image_path": [str(validation_path)], "pred_score": [1.0], "anomaly_map": [anomaly_map]},
        {validation_path: source_path},
        pipeline,
        {validation_path: tile},
    )
    final_predictions = _final_test_predictions(
        {"image_path": [str(final_test_path)], "pred_score": [1.0], "anomaly_map": [anomaly_map]},
        {final_test_path: source_path},
        0.6,
        pipeline,
        {final_test_path: tile},
    )

    assert calibration_samples[0].score == final_predictions[0].anomaly_score
    assert calibration_samples[0].score < 1.0
    assert final_predictions[0].source_path == str(source_path)
    assert final_predictions[0].predicted_label == "NG"


def test_final_test_predictions_persist_maps_and_score_provenance_when_an_artifact_directory_is_supplied(tmp_path: Path) -> None:
    source_path = (tmp_path / "source.png").resolve()
    Image.new("RGB", (321, 77), (10, 20, 30)).save(source_path)
    staged_path = (tmp_path / "final_test_ok" / "item.png").resolve()
    pipeline = PreprocessingPipeline(
        InspectionRegionConfig(),
        PreprocessingConfig().resolve("dinomaly_dinov3", (321, 77)),
    )
    anomaly_map = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)

    predictions = _final_test_predictions(
        {"image_path": [str(staged_path)], "pred_score": [0.9], "anomaly_map": [anomaly_map]},
        {staged_path: source_path},
        threshold=0.5,
        preprocessing_pipeline=pipeline,
        preprocessing_tile_by_staged_path={staged_path: pipeline.plan.tiles[0]},
        artifact_directory=tmp_path / "prediction_artifacts",
        inspection_region=InspectionRegionConfig(),
    )

    prediction = predictions[0]
    assert prediction.score_semantic == "anomalib_postprocessed_pred_score_v1"
    assert prediction.native_image_score == 0.9
    persisted_map = np.load(prediction.continuous_anomaly_map)["anomaly_map"]
    assert persisted_map.shape == (77, 321)
    assert persisted_map.min() == 2.0
    assert persisted_map.max() == 5.0
    assert prediction.anomaly_map
    assert prediction.overlay_image
    assert prediction.region_metadata["coordinate_space"] == "source_image"
    assert prediction.binary_mask == ""


def test_final_test_pixel_mask_threshold_is_independent_from_image_decision_threshold(tmp_path: Path) -> None:
    source_path = (tmp_path / "source.png").resolve()
    Image.new("RGB", (8, 6), (10, 20, 30)).save(source_path)
    staged_path = (tmp_path / "final_test_ok" / "item.png").resolve()
    anomaly_map = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)

    prediction = _final_test_predictions(
        {"image_path": [str(staged_path)], "pred_score": [0.1], "anomaly_map": [anomaly_map]},
        {staged_path: source_path},
        threshold=0.5,
        artifact_directory=tmp_path / "prediction_artifacts",
        inspection_region=InspectionRegionConfig(),
        pixel_threshold=3.5,
    )[0]

    assert prediction.predicted_label == "OK"
    assert prediction.pixel_threshold == 3.5
    assert prediction.pixel_threshold_comparator == "greater_than_or_equal"
    assert prediction.pixel_threshold_semantic == "continuous_anomaly_map_gte_v1"
    assert np.array_equal(
        np.asarray(Image.open(prediction.binary_mask)),
        np.array([[0, 0], [255, 255]], dtype=np.uint8),
    )
    assert Path(prediction.contour_overlay_image).is_file()


def test_native_final_test_artifacts_rectify_an_enabled_inspection_region(tmp_path: Path) -> None:
    source_path = (tmp_path / "source.png").resolve()
    Image.new("RGB", (16, 12), (10, 20, 30)).save(source_path)
    staged_path = (tmp_path / "final_test_ok" / "item.png").resolve()
    inspection_region = InspectionRegionConfig(
        enabled=True,
        source_width=16,
        source_height=12,
        points_px=((1, 1), (14, 1), (14, 10), (1, 10)),
    )

    prediction = _final_test_predictions(
        {"image_path": [str(staged_path)], "pred_score": [0.1], "anomaly_map": [np.ones((2, 2), dtype=np.float32)]},
        {staged_path: source_path},
        threshold=0.5,
        artifact_directory=tmp_path / "prediction_artifacts",
        inspection_region=inspection_region,
    )[0]

    assert Image.open(prediction.overlay_image).size == inspection_region.rectified_size()
    assert prediction.region_metadata["rectified_size"] == [13, 9]