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