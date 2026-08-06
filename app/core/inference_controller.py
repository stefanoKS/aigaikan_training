"""Inference process control."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, Signal

from app.core.result_parser import ResultParser


class InferenceController(QObject):
    """Launch and track the inference worker in a separate process."""

    log_message = Signal(str, str)
    completed = Signal(str)
    failed = Signal(str, str)
    running_changed = Signal(bool)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._process = QProcess(self)
        self._buffer = ""
        self._parser = ResultParser()
        self._process.readyReadStandardOutput.connect(self._handle_stdout)
        self._process.readyReadStandardError.connect(self._handle_stderr)
        self._process.finished.connect(self._handle_finished)

    def start_for_image(self, image_path: Path) -> None:
        """Run inference for a single image."""
        if self.is_busy():
            raise RuntimeError("Inference is already running.")
        self._buffer = ""
        self._process.start(
            sys.executable,
            ["-m", "app.workers.inference_worker", "--image", str(image_path)],
        )
        self.running_changed.emit(True)

    def cancel(self) -> None:
        """Cancel the worker."""
        if not self.is_busy():
            return
        self._process.terminate()
        if not self._process.waitForFinished(3000):
            self._process.kill()

    def is_busy(self) -> bool:
        """Return whether the worker is active."""
        return self._process.state() != QProcess.ProcessState.NotRunning

    def _handle_stdout(self) -> None:
        self._buffer += bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            message = self._parser.parse_worker_line(line)
            if message.type == "log":
                self.log_message.emit(
                    str(message.payload.get("level", "info")),
                    str(message.payload.get("message", "")),
                )
            elif message.type == "completed":
                self.completed.emit(str(message.payload.get("result_dir", "")))
            elif message.type == "error":
                self.failed.emit(
                    str(message.payload.get("message", "Inference failed")),
                    str(message.payload.get("details", "")),
                )

    def _handle_stderr(self) -> None:
        text = bytes(self._process.readAllStandardError()).decode("utf-8", errors="replace").strip()
        if text:
            self.log_message.emit("error", text)

    def _handle_finished(self) -> None:
        self.running_changed.emit(False)
