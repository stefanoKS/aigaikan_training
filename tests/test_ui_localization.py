"""Runtime UI language selection tests."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QAbstractButton, QApplication, QComboBox, QGroupBox, QLabel, QLineEdit, QMessageBox, QTableWidget, QWidget

from app.core.project_manager import ProjectManager
from app.core.settings_manager import SettingsManager
from app.models.prediction_result import PredictionResult
from app.ui import main_window as main_window_module
from app.ui.main_window import MainWindow


def test_language_selector_translates_ui_without_changing_runtime_input_paths() -> None:
    application = QApplication.instance() or QApplication([])
    settings = SettingsManager()
    window = MainWindow(settings, ProjectManager(settings.default_projects_directory()))
    input_path = Path("C:/inspection/line_a/part.png")
    window.inference_page.set_input_path(input_path)

    window.language_combo.setCurrentIndex(window.language_combo.findData("ja"))
    application.processEvents()

    assert window.language_label.text() == "言語"
    assert window.navigation.item(0).text() == "ホーム / プロジェクト"
    assert window.inference_page.load_run_button.text() == "学習済み実行を読み込む"
    assert window.inference_page.export_ng_images_button.text() == "NG 画像を出力"
    assert window.inference_page.input_label.text() == str(input_path)

    window.language_combo.setCurrentIndex(window.language_combo.findData("en"))
    application.processEvents()

    assert window.language_label.text() == "Language"
    assert window.navigation.item(0).text() == "Home / Projects"
    assert window.inference_page.load_run_button.text() == "Load Training Run"
    assert window.inference_page.export_ng_images_button.text() == "Export NG Images"
    assert window.inference_page.input_label.text() == str(input_path)
    window.close()


def test_japanese_translation_covers_every_static_button_caption() -> None:
    application = QApplication.instance() or QApplication([])
    settings = SettingsManager()
    window = MainWindow(settings, ProjectManager(settings.default_projects_directory()))
    button_sources = {
        button: button.text()
        for button in window.findChildren(QAbstractButton)
        if button.text()
    }

    assert button_sources
    assert all(source in window.ui_translator._JAPANESE for source in button_sources.values())

    window.language_combo.setCurrentIndex(window.language_combo.findData("ja"))
    application.processEvents()

    for button, source in button_sources.items():
        assert button.text() == window.ui_translator.text(source)

    window.language_combo.setCurrentIndex(window.language_combo.findData("en"))
    application.processEvents()

    for button, source in button_sources.items():
        assert button.text() == source
    window.close()


def test_dynamic_training_action_button_remains_translated_after_update() -> None:
    application = QApplication.instance() or QApplication([])
    settings = SettingsManager()
    window = MainWindow(settings, ProjectManager(settings.default_projects_directory()))

    window.language_combo.setCurrentIndex(window.language_combo.findData("ja"))
    window.ui_translator.set_button_text(window.training_page.start_button, "Run Evaluation")
    application.processEvents()

    assert window.training_page.start_button.text() == "評価を実行"

    window.ui_translator.set_button_text(window.training_page.start_button, "Start Training")
    application.processEvents()

    assert window.training_page.start_button.text() == "学習を開始"
    window.close()


def test_japanese_translation_covers_all_static_visible_ui_text() -> None:
    application = QApplication.instance() or QApplication([])
    settings = SettingsManager()
    window = MainWindow(settings, ProjectManager(settings.default_projects_directory()))
    technical_values = {
        "-", "0", "00:00:00", "0.1%", "0.5%", "1.0%", "ANOMALIB TRAINER", "PatchCore", "PaDiM",
        "AnomalyDINO", "SuperADD", "EfficientAD", "SuperSimpleNet", "Dinomaly (DINOv2)", "Dinomaly (DINOv3)",
        "CPU", "CUDA", "FP32", "auto", "legacy unversioned", "run manifest",
    }
    sources = _static_text_sources(window)
    window.language_combo.setCurrentIndex(window.language_combo.findData("ja"))
    application.processEvents()

    untranslated = sorted(
        source
        for getter, source in sources
        if source not in technical_values and window.ui_translator.text(source) == source
    )
    assert not untranslated

    assert all(getter() == window.ui_translator.text(source) for getter, source in sources)
    window.close()


def test_japanese_translation_refreshes_dynamic_superadd_and_preview_source_text() -> None:
    application = QApplication.instance() or QApplication([])
    settings = SettingsManager()
    window = MainWindow(settings, ProjectManager(settings.default_projects_directory()))
    window.language_combo.setCurrentIndex(window.language_combo.findData("ja"))

    window.config_page.model_combo.setCurrentIndex(window.config_page.model_combo.findData("super_add"))
    window.preprocess_images_page.active_source_label.setText("Active preview source: Project Good Images | frame.png")
    window._retranslate_ui()
    application.processEvents()

    assert "画像フォルダー" in window.config_page.model_support_label.text()
    assert "候補" in window.config_page.superadd_guidance_label.text()
    assert window.preprocess_images_page.active_source_label.text() == "使用中のプレビュー元: プロジェクトの正常画像 | frame.png"
    window.close()


def test_dynamic_training_stage_translation_returns_to_english() -> None:
    application = QApplication.instance() or QApplication([])
    settings = SettingsManager()
    window = MainWindow(settings, ProjectManager(settings.default_projects_directory()))
    window.language_combo.setCurrentIndex(window.language_combo.findData("ja"))

    window._update_training_stage("Validating dataset")

    assert window.training_page.current_stage_label.text() == "データセットを検証中"
    window.language_combo.setCurrentIndex(window.language_combo.findData("en"))
    application.processEvents()

    assert window.training_page.current_stage_label.text() == "Validating dataset"
    window.close()


def test_japanese_translation_refreshes_results_and_inference_decision_previews() -> None:
    application = QApplication.instance() or QApplication([])
    settings = SettingsManager()
    window = MainWindow(settings, ProjectManager(settings.default_projects_directory()))
    window.language_combo.setCurrentIndex(window.language_combo.findData("ja"))

    window.results_page.display_decision_preview(
        SimpleNamespace(
            calibrated_threshold=0.1,
            active_threshold=0.1,
            proposed_threshold=0.2,
            score_semantic="superadd_native_top_quantile_score_v1",
            ok_to_ng_changes=1,
            ng_to_ok_changes=2,
            false_reject_rate=0.1,
            ng_recall=0.9,
            outside_calibration_range=True,
        )
    )
    window.inference_page.set_training_run(
        Path("run"),
        "PatchCore",
        threshold=0.1,
        score_semantic="anomalib_postprocessed_pred_score_v1",
    )
    window.inference_page.append_prediction(
        PredictionResult(
            "input.png",
            "OK",
            "Unknown",
            0.2,
            0.1,
            score_semantic="anomalib_postprocessed_pred_score_v1",
        )
    )
    window.inference_page.decision_preview_check.setChecked(True)
    window.inference_page.decision_preview_spin.setValue(0.1)
    application.processEvents()

    assert "既存値" in window.results_page.threshold_preview_label.text()
    assert "警告" in window.results_page.threshold_preview_label.text()
    assert window.inference_page.decision_preview_summary_label.text().startswith("推論時:")
    window.close()


def test_japanese_translation_applies_to_main_window_dialog_titles_and_fixed_messages(monkeypatch) -> None:
    application = QApplication.instance() or QApplication([])
    settings = SettingsManager()
    window = MainWindow(settings, ProjectManager(settings.default_projects_directory()))
    window.language_combo.setCurrentIndex(window.language_combo.findData("ja"))
    captured: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda _parent, title, text, *_args, **_kwargs: captured.append((title, text)),
    )

    main_window_module.QMessageBox.information(
        window,
        "No Project",
        "Create or open a project first.",
    )

    assert application is not None
    assert captured == [("プロジェクト未選択", "最初にプロジェクトを作成または開いてください。")]
    window.close()


def _static_text_sources(window: MainWindow) -> list[tuple[object, str]]:
    """Capture all initialized text surfaces whose values must survive an English/Japanese toggle."""
    values: list[tuple[object, str]] = []
    for widget in (window, *window.findChildren(QWidget)):
        if isinstance(widget, QAbstractButton) and widget.text():
            values.append((widget.text, widget.text()))
        elif isinstance(widget, QGroupBox) and widget.title():
            values.append((widget.title, widget.title()))
        elif isinstance(widget, QLabel) and widget.text():
            values.append((widget.text, widget.text()))
        elif isinstance(widget, QLineEdit) and widget.placeholderText():
            values.append((widget.placeholderText, widget.placeholderText()))
        if isinstance(widget, QComboBox):
            values.extend((lambda index=index, combo=widget: combo.itemText(index), widget.itemText(index)) for index in range(widget.count()))
        if isinstance(widget, QTableWidget):
            values.extend(
                (
                    lambda index=index, table=widget: table.horizontalHeaderItem(index).text(),
                    widget.horizontalHeaderItem(index).text(),
                )
                for index in range(widget.columnCount())
                if widget.horizontalHeaderItem(index) is not None
            )
    return [(getter, source) for getter, source in values if source]