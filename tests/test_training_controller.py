"""Tests for training process log classification."""

from pathlib import Path
from unittest.mock import MagicMock

from app.core.training_controller import TrainingController
from app.ui.main_window import MainWindow


def test_dependency_warnings_are_not_reported_as_training_errors() -> None:
    """Library deprecations should remain visible without implying a failed run."""
    assert TrainingController._stderr_log_level("FutureWarning: deprecated import") == "warning"
    assert TrainingController._stderr_log_level("Traceback\nValueError: failed") == "error"
    assert TrainingController._stderr_log_level("GPU available: True (cuda)") == "info"


def test_framework_stdout_is_not_treated_as_a_worker_protocol_error() -> None:
    """Lightning text output must not break JSON Lines message processing."""
    controller = TrainingController()
    emitted_logs: list[tuple[str, str]] = []
    controller.log_message.connect(lambda level, message: emitted_logs.append((level, message)))
    controller._buffer = "Trainer.fit stopped: max_epochs=1 reached.\n"

    controller._handle_stdout()

    assert emitted_logs == [("info", "Trainer.fit stopped: max_epochs=1 reached.")]


def test_stage_progress_messages_emit_the_batch_counts() -> None:
    """The controller forwards worker batch progress to the UI layer."""
    controller = TrainingController()
    emitted_progress: list[tuple[int, int]] = []
    controller.stage_progress_changed.connect(lambda current, total: emitted_progress.append((current, total)))
    controller._buffer = '{"type": "stage_progress", "current": 3, "total": 8}\n'

    controller._handle_stdout()

    assert emitted_progress == [(3, 8)]


def test_training_completion_stops_indeterminate_stage_progress() -> None:
    """A completed run must leave the stage bar fixed at 100 percent."""
    window = MagicMock()
    window.current_project = None
    window.result_parser.read_training_run.return_value = MagicMock()
    window.results_page = MagicMock()
    window.training_page = MagicMock()
    window.navigation = MagicMock()

    MainWindow._training_completed(window, str(Path("C:/runs/finished")))

    window.training_page.stage_progress.setRange.assert_called_once_with(0, 1)
    window.training_page.stage_progress.setValue.assert_called_once_with(1)