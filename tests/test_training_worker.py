"""Tests for training worker progress reporting and calibration data isolation."""

from pathlib import Path

from app.workers.training_worker import (
    TrainingProgressReporter,
    _peak_gpu_memory_mb,
    calibration_samples_from_predictions,
    configure_worker_stdio,
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