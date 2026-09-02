"""Inference-page log behavior tests."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app.ui.pages.inference_page import InferencePage


def test_inference_log_is_scrollable_and_follows_new_messages() -> None:
    application = QApplication.instance() or QApplication([])
    page = InferencePage()
    page.log_output.setFixedHeight(40)
    page.show()
    for index in range(20):
        page.append_log("info", f"message {index}")
    application.processEvents()

    scrollbar = page.log_output.verticalScrollBar()
    assert scrollbar.value() == scrollbar.maximum()

    page.close()