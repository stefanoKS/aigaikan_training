"""Training process control via QProcess."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, Signal

from app.core.result_parser import ResultParser


class TrainingController(QObject):
    """Launch and track the training worker in a separate process."""

    stage_changed = Signal(str)
    progress_changed = Signal(int, int)
    log_message = Signal(str, str)
    metric_emitted = Signal(str, object)
    result_image_emitted = Signal(str)
    completed = Signal(str)
    failed = Signal(str, str)
    running_changed = Signal(bool)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._process = QProcess(self)
        self._parser = ResultParser()
        self._buffer = ""
        self._project_file: Path | None = None
        self._process.readyReadStandardOutput.connect(self._handle_stdout)
        self._process.readyReadStandardError.connect(self._handle_stderr)
        self._process.finished.connect(self._handle_finished)

    def start(self, project_file: Path) -> None:
        """Start training for a project file."""
        if self.is_busy():
            raise RuntimeError("Training is already running for this project.")
        self._project_file = project_file
        self._buffer = ""
        self._process.start(
            sys.executable,
            ["-m", "app.workers.training_worker", "--project-file", str(project_file)],
        )
        self.running_changed.emit(True)

    def cancel(self) -> None:
        """Cancel the active worker."""
        if not self.is_busy():
            return
        self._process.terminate()
        if not self._process.waitForFinished(3000):
            self._process.kill()

    def is_busy(self) -> bool:
        """Return whether the worker is still running."""
        return self._process.state() != QProcess.ProcessState.NotRunning

    def _handle_stdout(self) -> None:
        self._buffer += bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            message = self._parser.parse_worker_line(line)
            if message.type == "stage":
                self.stage_changed.emit(str(message.payload.get("name", "")))
            elif message.type == "progress":
                self.progress_changed.emit(
                    int(message.payload.get("current", 0)),
                    int(message.payload.get("total", 0)),
                )
            elif message.type == "log":
                self.log_message.emit(
                    str(message.payload.get("level", "info")),
                    str(message.payload.get("message", "")),
                )
            elif message.type == "metric":
                self.metric_emitted.emit(
                    self._parser.normalize_metric_name(str(message.payload.get("name", ""))),
                    message.payload.get("value"),
                )
            elif message.type == "result_image":
                self.result_image_emitted.emit(str(message.payload.get("path", "")))
            elif message.type == "completed":
                self.completed.emit(str(message.payload.get("result_dir", "")))
            elif message.type == "error":
                self.failed.emit(
                    str(message.payload.get("message", "Training failed")),
                    str(message.payload.get("details", "")),
                )

    def _handle_stderr(self) -> None:
        text = bytes(self._process.readAllStandardError()).decode("utf-8", errors="replace").strip()
        if text:
            self.log_message.emit("error", text)

    def _handle_finished(self) -> None:
        self.running_changed.emit(False)
