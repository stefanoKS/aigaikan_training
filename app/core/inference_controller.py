"""Inference process control."""

from __future__ import annotations

from json import JSONDecodeError
import sys
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, Signal

from app.core.result_parser import ResultParser
from app.models.prediction_result import PredictionResult


class InferenceController(QObject):
    """Launch and track the inference worker in a separate process."""

    log_message = Signal(str, str)
    progress_changed = Signal(int, int)
    prediction_emitted = Signal(object)
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

    def start(self, run_directory: Path, input_path: Path) -> None:
        """Run inference for a selected model run and image or folder input."""
        if self.is_busy():
            raise RuntimeError("Inference is already running.")
        self._buffer = ""
        self._process.start(
            sys.executable,
            [
                "-m",
                "app.workers.inference_worker",
                "--run-dir",
                str(run_directory),
                "--input",
                str(input_path),
            ],
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
            try:
                message = self._parser.parse_worker_line(line)
            except (JSONDecodeError, KeyError, TypeError):
                self.log_message.emit(self._stdout_log_level(line), line)
                continue
            if message.type == "log":
                self.log_message.emit(
                    str(message.payload.get("level", "info")),
                    str(message.payload.get("message", "")),
                )
            elif message.type == "progress":
                self.progress_changed.emit(
                    int(message.payload.get("current", 0)),
                    int(message.payload.get("total", 0)),
                )
            elif message.type == "prediction":
                self.prediction_emitted.emit(
                    PredictionResult(
                        source_path=str(message.payload.get("source_path", "")),
                        predicted_label=str(message.payload.get("predicted_label", "")),
                        ground_truth_label=str(message.payload.get("ground_truth_label", "Unknown")),
                        anomaly_score=float(message.payload.get("anomaly_score", 0.0)),
                        threshold=float(message.payload.get("threshold", 0.0)),
                        original_image=str(message.payload.get("original_image", "")),
                        anomaly_map=str(message.payload.get("anomaly_map", "")),
                        overlay_image=str(message.payload.get("overlay_image", "")),
                    )
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
            self.log_message.emit(self._stderr_log_level(text), text)

    @staticmethod
    def _stderr_log_level(text: str) -> str:
        normalized = text.lower()
        if "traceback" in normalized or "exception" in normalized:
            return "error"
        if "warning" in normalized or "deprecated" in normalized or "triton not found" in normalized:
            return "warning"
        return "info"

    @staticmethod
    def _stdout_log_level(text: str) -> str:
        return "warning" if "warning" in text.lower() else "info"

    def _handle_finished(self) -> None:
        self.running_changed.emit(False)
