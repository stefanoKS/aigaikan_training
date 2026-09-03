"""Runtime UI language selection tests."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app.core.project_manager import ProjectManager
from app.core.settings_manager import SettingsManager
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