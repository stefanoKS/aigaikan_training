"""Inference-page log behavior tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app.models.prediction_result import PredictionResult
from app.ui.pages.inference_page import InferencePage


_SCORE_SEMANTIC = "anomalib_postprocessed_pred_score_v1"


def _preview_page(threshold: float = 0.1) -> InferencePage:
    page = InferencePage()
    page.set_training_run(
        Path("run"),
        "PatchCore",
        threshold=threshold,
        score_semantic=_SCORE_SEMANTIC,
    )
    return page


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


def test_inference_page_shows_original_overlay_and_binary_mask_previews(tmp_path: Path) -> None:
    application = QApplication.instance() or QApplication([])
    original_path = tmp_path / "original.png"
    overlay_path = tmp_path / "overlay.png"
    mask_path = tmp_path / "mask.png"
    from PIL import Image

    Image.new("RGB", (16, 12), (20, 30, 40)).save(original_path)
    Image.new("RGB", (16, 12), (200, 50, 20)).save(overlay_path)
    Image.new("L", (16, 12), 255).save(mask_path)
    page = InferencePage()
    page.resize(900, 700)
    page.show()
    page.append_prediction(
        PredictionResult(
            source_path=str(original_path),
            predicted_label="NG",
            ground_truth_label="Unknown",
            anomaly_score=0.9,
            threshold=0.5,
            original_image=str(original_path),
            overlay_image=str(overlay_path),
            binary_mask=str(mask_path),
        )
    )
    application.processEvents()

    assert tuple(page.preview_labels) == ("Original", "Overlay", "Mask")
    assert not page.preview_labels["Original"].pixmap().isNull()
    assert not page.preview_labels["Overlay"].pixmap().isNull()
    assert not page.preview_labels["Mask"].pixmap().isNull()

    page.close()


def test_inference_page_displays_industrial_batch_one_benchmark_summary() -> None:
    application = QApplication.instance() or QApplication([])
    page = InferencePage()
    payload = {
        "metadata": {
            "backbone": "vit_small_plus_patch16_dinov3.lvd1689m",
            "model_precision": "float16",
            "prepared_canvas_size": [448, 448],
            "peak_cuda_memory_allocated": 256 * 1024 * 1024,
        },
        "timing": {
            "preprocess_total_ms": {"p50_ms": 2.0, "p95_ms": 3.0},
            "model_forward_ms": {"p50_ms": 20.0, "p95_ms": 24.0},
            "model_pipeline_ms": {"p50_ms": 22.0, "p95_ms": 26.0},
            "end_to_end_compute_ms": {"p50_ms": 25.0, "p95_ms": 30.0, "p99_ms": 35.0},
        },
        "measured_steady_state_fps": 40.0,
        "conservative_p95_fps": 33.333,
        "deadline": {
            "frame_period_ms": 100.0,
            "allowed_compute_budget_ms": 80.0,
            "pass": False,
            "reason": "P95 end-to-end compute latency exceeds the budget.",
        },
    }

    page.display_benchmark(payload)

    assert application is not None
    assert page.benchmark_summary_labels["Backbone"].text().startswith("vit_small_plus")
    assert page.benchmark_summary_labels["PASS / FAIL"].text() == "FAIL"
    assert "exceeds" in page.benchmark_summary_labels["Assessment"].text()
    assert page.export_benchmark_json_button.isEnabled()
    assert page.export_benchmark_csv_button.isEnabled()
    page.close()


def test_image_decision_preview_changes_only_derived_columns_and_preserves_artifacts() -> None:
    application = QApplication.instance() or QApplication([])
    page = _preview_page()
    prediction = PredictionResult(
        source_path="part.png",
        predicted_label="NG",
        ground_truth_label="Unknown",
        anomaly_score=0.12,
        threshold=0.08,
        score_semantic=_SCORE_SEMANTIC,
        continuous_anomaly_map="continuous.npz",
        anomaly_map="heatmap.png",
        overlay_image="overlay.png",
        binary_mask="mask.png",
        contour_overlay_image="contours.png",
    )
    page.append_prediction(prediction)
    page.decision_preview_check.setChecked(True)
    page.decision_preview_spin.setValue(0.15)
    application.processEvents()

    assert page.results_table.columnCount() == 7
    assert [page.results_table.item(0, column).text() for column in range(1, 5)] == ["NG", "0.12", "0.08", "Available"]
    assert page.results_table.item(0, 5).text() == "OK"
    assert page.results_table.item(0, 6).text() == "NG → OK"
    assert page.decision_preview_counts() == {
        "inference_ok": 0,
        "inference_ng": 1,
        "displayed_ok": 1,
        "displayed_ng": 0,
        "ok_to_ng": 0,
        "ng_to_ok": 1,
    }
    assert prediction.predicted_label == "NG"
    assert prediction.threshold == pytest.approx(0.08)
    assert prediction.continuous_anomaly_map == "continuous.npz"
    assert prediction.anomaly_map == "heatmap.png"
    assert prediction.overlay_image == "overlay.png"
    assert prediction.binary_mask == "mask.png"
    assert prediction.contour_overlay_image == "contours.png"
    assert page.threshold_label.text() == "0.1"
    assert page.displayed_decision_label.text() == "OK"
    assert page.displayed_threshold_label.text() == "0.15"
    page.close()


def test_image_decision_preview_preserves_tiny_and_unbounded_score_scales() -> None:
    application = QApplication.instance() or QApplication([])
    page = _preview_page(0.034)
    page.append_prediction(
        PredictionResult("tiny-a.png", "NG", "Unknown", 0.034, 0.034, score_semantic=_SCORE_SEMANTIC)
    )
    page.append_prediction(
        PredictionResult("tiny-b.png", "NG", "Unknown", 0.041, 0.034, score_semantic=_SCORE_SEMANTIC)
    )
    page.decision_preview_check.setChecked(True)
    page.decision_preview_spin.setValue(0.041)
    application.processEvents()

    assert page.decision_preview_spin.value() == pytest.approx(0.041, abs=1e-9)
    assert page.results_table.item(0, 5).text() == "OK"
    assert page.results_table.item(1, 5).text() == "NG"
    assert page.results_table.item(1, 6).text() == "—"
    assert page.decision_preview_spin.singleStep() < 0.001

    unbounded_page = _preview_page(3.0)
    unbounded_page.append_prediction(
        PredictionResult("distance.png", "NG", "Unknown", 3.5, 3.0, score_semantic=_SCORE_SEMANTIC)
    )
    unbounded_page.decision_preview_check.setChecked(True)
    unbounded_page.decision_preview_spin.setValue(3.5)

    assert unbounded_page.results_table.item(0, 5).text() == "NG"
    assert unbounded_page.decision_preview_spin.value() == pytest.approx(3.5)
    unbounded_page.close()
    page.close()


def test_image_decision_preview_reset_and_custom_copy_filter_are_independent() -> None:
    application = QApplication.instance() or QApplication([])
    page = _preview_page(0.5)
    prediction = PredictionResult("part.png", "NG", "Unknown", 0.7, 0.5, score_semantic=_SCORE_SEMANTIC)
    page.append_prediction(prediction)
    page.export_threshold_check.setChecked(True)
    page.export_threshold_spin.setValue(0.8)
    page.decision_preview_check.setChecked(True)
    page.decision_preview_spin.setValue(0.9)
    application.processEvents()

    assert page.export_threshold() == pytest.approx(0.8)
    assert page.ng_predictions_for_export() == []
    assert page.results_table.item(0, 5).text() == "OK"
    page.reset_decision_preview()

    assert not page.decision_preview_check.isChecked()
    assert page.decision_preview_spin.value() == pytest.approx(0.5)
    assert page.results_table.item(0, 5).text() == "NG"
    assert page.export_threshold() == pytest.approx(0.8)
    page.set_active_decision_threshold(0.6, "active decision revision: threshold-001", _SCORE_SEMANTIC)

    assert page.export_threshold() == pytest.approx(0.8)
    page.close()


def test_streamed_prediction_uses_active_preview_and_loading_run_clears_preview_state() -> None:
    application = QApplication.instance() or QApplication([])
    page = _preview_page(0.1)
    page.decision_preview_check.setChecked(True)
    page.decision_preview_spin.setValue(0.15)
    page.append_prediction(
        PredictionResult("streamed.png", "NG", "Unknown", 0.12, 0.1, score_semantic=_SCORE_SEMANTIC)
    )
    application.processEvents()

    assert page.results_table.item(0, 5).text() == "OK"
    assert page.results_table.item(0, 6).text() == "NG → OK"
    page.set_training_run(Path("other-run"), "PatchCore", threshold=0.8, score_semantic=_SCORE_SEMANTIC)

    assert not page.predictions
    assert page.results_table.rowCount() == 0
    assert not page.decision_preview_check.isChecked()
    assert page.decision_preview_spin.value() == pytest.approx(0.8)
    assert page.decision_preview_counts()["ok_to_ng"] == 0
    page.close()


def test_image_decision_preview_fails_closed_for_inconsistent_score_semantics() -> None:
    page = _preview_page(0.1)
    page.append_prediction(
        PredictionResult("compatible.png", "NG", "Unknown", 0.12, 0.1, score_semantic=_SCORE_SEMANTIC)
    )
    page.append_prediction(
        PredictionResult("incompatible.png", "NG", "Unknown", 0.12, 0.1, score_semantic="other_score_domain")
    )

    assert not page.decision_preview_check.isEnabled()
    assert not page.save_decision_revision_button.isEnabled()
    assert page.results_table.item(0, 5).text() == "Not available"
    assert page.results_table.item(1, 5).text() == "Not available"
    assert "unavailable" in page.decision_preview_summary_label.text()
    page.close()


def test_image_decision_preview_save_signal_carries_precise_threshold_and_note() -> None:
    page = _preview_page(0.034)
    page.append_prediction(
        PredictionResult("part.png", "NG", "Unknown", 0.041, 0.034, score_semantic=_SCORE_SEMANTIC)
    )
    received: list[tuple[float, str]] = []
    page.decision_revision_save_requested.connect(lambda threshold, note: received.append((threshold, note)))
    page.decision_preview_check.setChecked(True)
    page.decision_preview_spin.setValue(0.041)
    page.decision_preview_note_edit.setText("line adjustment")

    page.save_decision_revision_button.click()

    assert received == [(pytest.approx(0.041, abs=1e-9), "line adjustment")]
    page.close()


def test_image_decision_preview_save_is_disabled_while_inference_runs() -> None:
    page = _preview_page()
    page.append_prediction(
        PredictionResult("part.png", "NG", "Unknown", 0.12, 0.1, score_semantic=_SCORE_SEMANTIC)
    )
    page.decision_preview_check.setChecked(True)

    assert page.save_decision_revision_button.isEnabled()
    page.set_running(True)

    assert not page.decision_preview_check.isEnabled()
    assert not page.save_decision_revision_button.isEnabled()
    page.set_running(False)

    assert page.decision_preview_check.isEnabled()
    assert page.save_decision_revision_button.isEnabled()
    page.close()