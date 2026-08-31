"""Tests for training worker progress reporting."""

from app.workers.training_worker import TrainingProgressReporter


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