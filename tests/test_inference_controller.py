"""Inference process argument tests."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QProcess

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


def test_controller_starts_inference_without_a_threshold_override() -> None:
    QCoreApplication.instance() or QCoreApplication([])
    controller = InferenceController()
    process = _FakeProcess()
    controller._process = process

    controller.start(Path("run"), Path("input"))

    assert process.arguments == ["-m", "app.workers.inference_worker", "--run-dir", "run", "--input", "input"]