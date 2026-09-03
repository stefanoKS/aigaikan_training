"""Inference-page log behavior tests."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app.models.prediction_result import PredictionResult
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


def test_inference_page_filters_ng_export_after_inference_without_changing_prediction_labels() -> None:
    application = QApplication.instance() or QApplication([])
    page = InferencePage()
    source_path = r"C:\inspection\line_a\camera_01\part.png"
    page.set_training_run(Path("run"), "PatchCore", threshold=0.8)
    page.append_prediction(
        PredictionResult(
            source_path=source_path,
            predicted_label="OK",
            ground_truth_label="Unknown",
            anomaly_score=0.7,
            threshold=0.8,
        )
    )

    assert application is not None
    assert page.input_label.wordWrap()
    assert page.export_threshold() == 0.8
    assert not page.results_table.selectionModel().selectedRows()
    displayed_path = page.results_table.item(0, 0).text()
    assert "\n" in displayed_path
    assert displayed_path.replace("\n", "") == source_path
    assert page.results_table.item(0, 0).toolTip() == source_path
    assert page.ng_predictions_for_export() == []
    assert page.export_ng_images_button.isEnabled()
    assert page.results_table.item(0, 1).text() == "OK"
    page.export_threshold_check.setChecked(True)
    assert page.export_ng_images_button.isEnabled()
    page.export_threshold_spin.setValue(0.6)
    assert page.ng_predictions_for_export()[0].source_path == source_path
    assert page.export_ng_images_button.isEnabled()
    assert page.results_table.item(0, 1).text() == "OK"