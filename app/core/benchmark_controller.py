"""Non-blocking process control for the checkpoint benchmark runner."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

from PySide6.QtCore import QObject, QProcess, Signal


class BenchmarkController(QObject):
    """Launch the batch-one benchmark script without blocking the Qt event loop."""

    log_message = Signal(str, str)
    progress_changed = Signal(int, int)
    completed = Signal(str, str)
    failed = Signal(str)
    running_changed = Signal(bool)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._process = QProcess(self)
        self._process.readyReadStandardOutput.connect(self._handle_stdout)
        self._process.readyReadStandardError.connect(self._handle_stderr)
        self._process.finished.connect(self._handle_finished)
        self._output_json = Path()
        self._output_csv = Path()

    def start(
        self,
        run_directory: Path,
        input_path: Path,
        *,
        device: str,
        mode: str,
        warmup_frames: int,
        measured_frames: int,
        target_fps: float,
        reserve_percent: float,
    ) -> None:
        """Start an isolated benchmark process with persistent in-process model state."""
        if self.is_busy():
            raise RuntimeError("Industrial inference benchmark is already running.")
        destination = run_directory / "benchmarks"
        destination.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._output_json = destination / f"industrial_benchmark_{timestamp}.json"
        self._output_csv = self._output_json.with_suffix(".csv")
        script = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_run_inference.py"
        self._process.start(
            sys.executable,
            [
                str(script),
                "--run-dir", str(run_directory),
                "--input", str(input_path),
                "--device", device,
                "--mode", mode,
                "--warmup", str(warmup_frames),
                "--iterations", str(measured_frames),
                "--target-fps", str(target_fps),
                "--reserve-percent", str(reserve_percent),
                "--output", str(self._output_json),
                "--csv-output", str(self._output_csv),
            ],
        )
        self.running_changed.emit(True)

    def cancel(self) -> None:
        """Terminate an active benchmark between process-level work units."""
        if not self.is_busy():
            return
        self._process.terminate()
        if not self._process.waitForFinished(3000):
            self._process.kill()

    def is_busy(self) -> bool:
        return self._process.state() != QProcess.ProcessState.NotRunning

    def _handle_stdout(self) -> None:
        text = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in text.splitlines():
            self.log_message.emit("info", line)
            if line.startswith("Measured frame ") and "/" in line:
                try:
                    current, total = line.removeprefix("Measured frame ").split("/", 1)
                    self.progress_changed.emit(int(current), int(total))
                except ValueError:
                    pass

    def _handle_stderr(self) -> None:
        text = bytes(self._process.readAllStandardError()).decode("utf-8", errors="replace").strip()
        if text:
            self.log_message.emit("error", text)

    def _handle_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self.running_changed.emit(False)
        if exit_code == 0 and self._output_json.is_file() and self._output_csv.is_file():
            self.completed.emit(str(self._output_json), str(self._output_csv))
        elif exit_code != 0:
            self.failed.emit("Industrial inference benchmark failed or was cancelled. See the inference log for details.")