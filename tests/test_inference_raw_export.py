"""Tests for raw NG candidate export from inference results."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app.models.prediction_result import PredictionResult
from app.ui.main_window import MainWindow
from app.ui.pages.inference_page import InferencePage


def test_raw_ng_export_copies_source_bytes_without_exporting_heatmaps(tmp_path: Path) -> None:
    raw_source = tmp_path / "input" / "part.png"
    raw_source.parent.mkdir()
    raw_source.write_bytes(b"raw-inspection-image")
    heatmap = tmp_path / "heatmap.png"
    heatmap.write_bytes(b"heatmap-image")
    prediction = PredictionResult(
        source_path=str(raw_source),
        predicted_label="NG",
        ground_truth_label="Unknown",
        anomaly_score=0.9,
        threshold=0.5,
        anomaly_map=str(heatmap),
    )

    copied_paths = MainWindow._copy_raw_ng_images([prediction], tmp_path / "exported")

    assert len(copied_paths) == 1
    assert copied_paths[0].name == "NG_0001_part.png"
    assert copied_paths[0].read_bytes() == b"raw-inspection-image"
    assert copied_paths[0].read_bytes() != heatmap.read_bytes()


def test_raw_ng_export_copies_every_eligible_result_when_no_rows_are_selected(tmp_path: Path) -> None:
    application = QApplication.instance() or QApplication([])
    sources = []
    for name, content in (("ok.png", b"ok"), ("ng_one.png", b"ng-one"), ("ng_two.png", b"ng-two")):
        source = tmp_path / "input" / name
        source.parent.mkdir(exist_ok=True)
        source.write_bytes(content)
        sources.append(source)
    page = InferencePage()
    page.set_training_run(tmp_path / "run", "PatchCore", threshold=0.5)
    for source, score in zip(sources, (0.1, 0.5, 0.9), strict=True):
        page.append_prediction(
            PredictionResult(
                source_path=str(source),
                predicted_label="NG" if score >= 0.5 else "OK",
                ground_truth_label="Unknown",
                anomaly_score=score,
                threshold=0.5,
            )
        )

    assert application is not None
    assert not page.results_table.selectionModel().selectedRows()
    copied_paths = MainWindow._copy_raw_ng_images(page.ng_predictions_for_export(), tmp_path / "exported")

    assert [path.name for path in copied_paths] == ["NG_0001_ng_one.png", "NG_0002_ng_two.png"]
    assert [path.read_bytes() for path in copied_paths] == [b"ng-one", b"ng-two"]
    page.close()