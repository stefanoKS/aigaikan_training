"""Inference process argument tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QApplication

from app.core.inference_controller import InferenceController


class _FakeProcess:
    def __init__(self) -> None:
        self.program = ""
        self.arguments: list[str] = []

    def state(self) -> QProcess.ProcessState:
        return QProcess.ProcessState.NotRunning

    def start(self, program: str, arguments: list[str]) -> None:
        self.program = program
        self.arguments = arguments


class _StdoutProcess(_FakeProcess):
    def __init__(self, output: str) -> None:
        super().__init__()
        self._output = output.encode("utf-8")

    def readAllStandardOutput(self) -> bytes:
        return self._output


def test_controller_starts_inference_without_a_threshold_override() -> None:
    QApplication.instance() or QApplication([])
    controller = InferenceController()
    process = _FakeProcess()
    controller._process = process

    controller.start(Path("run"), Path("input"))

    assert process.arguments == ["-m", "app.workers.inference_worker", "--run-dir", "run", "--input", "input"]


def test_controller_preserves_binary_mask_from_worker_prediction() -> None:
    QApplication.instance() or QApplication([])
    controller = InferenceController()
    controller._process = _StdoutProcess(
        json.dumps(
            {
                "type": "prediction",
                "source_path": "source.png",
                "predicted_label": "NG",
                "ground_truth_label": "Unknown",
                "anomaly_score": 0.9,
                "threshold": 0.5,
                "binary_mask": "binary_mask.png",
            }
        )
        + "\n"
    )
    predictions = []
    controller.prediction_emitted.connect(predictions.append)

    controller._handle_stdout()

    assert len(predictions) == 1
    assert predictions[0].binary_mask == "binary_mask.png"