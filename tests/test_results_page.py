"""Focused Results-page evidence-state tests."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app.models.training_run import TrainingRun
from app.services.export_service import ModelExportFormat
from app.ui.pages.results_page import ResultsPage


def test_normal_only_results_show_warning_and_hide_defect_metrics() -> None:
    application = QApplication.instance() or QApplication([])
    page = ResultsPage()
    page.set_training_run(
        TrainingRun(
            run_name="normal-only",
            run_dir="",
            model_name="PatchCore",
            device="cpu",
            quality_status="NOT VERIFIED",
            metrics={
                "Defect Detection Evidence": "NOT MEASURED",
                "NG Detection Rate": "NOT MEASURED",
                "Escape Rate": "NOT MEASURED",
                "AUROC": "NOT MEASURED",
                "Precision": "NOT MEASURED",
                "Recall": "NOT MEASURED",
                "F1": "NOT MEASURED",
            },
        )
    )

    assert application is not None
    assert not page.no_ng_warning_label.isHidden()
    assert page.metric_labels["Image AUROC"].text() == "NOT MEASURED"
    assert page.metric_labels["Precision"].text() == "NOT MEASURED"


def test_aigaikan_export_defaults_to_torch_and_keeps_runtime_compatibility_pending() -> None:
    application = QApplication.instance() or QApplication([])
    page = ResultsPage()
    page.set_training_run(
        TrainingRun(
            run_name="exportable",
            run_dir="",
            model_name="PatchCore",
            device="cpu",
            anomalib_export_parity_status="Validated with Anomalib deployment inferencer: TORCH",
        )
    )

    assert application is not None
    assert page.export_format_checks[ModelExportFormat.TORCH].isChecked()
    assert not page.export_format_checks[ModelExportFormat.ONNX].isChecked()
    assert not page.export_format_checks[ModelExportFormat.OPENVINO].isChecked()
    assert page.export_model_button.text() == "Export for AIGAIKAN"
    assert page.metric_labels["Anomalib Export Parity"].text().startswith("Validated")
    assert page.metric_labels["AIGAIKAN Compatibility"].text() == "Pending AIGAIKAN runtime validation"