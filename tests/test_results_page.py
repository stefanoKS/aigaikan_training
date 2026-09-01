"""Focused Results-page evidence-state tests."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app.models.training_run import TrainingRun
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