"""Main-window navigation layout tests."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from PIL import Image

from app.models.dataset_config import DatasetRole, FolderImportMode
from app.core.project_manager import ProjectManager
from app.core.settings_manager import SettingsManager
from app.models.project_config import ProjectConfig
from app.ui.main_window import MainWindow
from app.ui.styles import APP_STYLE


def test_navigation_items_reserve_space_for_their_styled_padding() -> None:
    application = QApplication.instance() or QApplication([])
    settings = SettingsManager()
    window = MainWindow(settings, ProjectManager(settings.default_projects_directory()))
    window.resize(557, 878)
    window.show()
    application.processEvents()

    assert application is not None
    assert all(window.navigation.item(index).sizeHint().height() == 46 for index in range(window.navigation.count()))
    item_rectangles = [
        window.navigation.visualItemRect(window.navigation.item(index))
        for index in range(window.navigation.count())
    ]
    assert all(current.bottom() < following.top() for current, following in zip(item_rectangles, item_rectangles[1:]))

    window.close()


def test_global_style_covers_all_popup_dialog_surfaces() -> None:
    application = QApplication.instance() or QApplication([])
    application.setStyleSheet(APP_STYLE)

    assert "QDialog, QMessageBox, QMenu" in APP_STYLE
    assert "QFileDialog QListView" in APP_STYLE
    assert "QToolTip" in APP_STYLE


def test_training_log_scrolls_to_its_latest_message() -> None:
    application = QApplication.instance() or QApplication([])
    settings = SettingsManager()
    window = MainWindow(settings, ProjectManager(settings.default_projects_directory()))
    window.training_page.log_output.setFixedHeight(40)
    window.training_page.show()
    for index in range(20):
        window.training_page.append_log("info", f"message {index}")
    application.processEvents()

    scrollbar = window.training_page.log_output.verticalScrollBar()
    assert scrollbar.value() == scrollbar.maximum()

    window.close()


def test_default_inference_run_skips_newer_invalid_run(tmp_path: Path, monkeypatch) -> None:
    application = QApplication.instance() or QApplication([])
    settings = SettingsManager()
    window = MainWindow(settings, ProjectManager(settings.default_projects_directory()))
    runs_directory = tmp_path / "runs"
    invalid_config = runs_directory / "newer-invalid" / "config.json"
    valid_config = runs_directory / "older-valid" / "config.json"
    invalid_config.parent.mkdir(parents=True)
    valid_config.parent.mkdir(parents=True)
    invalid_config.touch()
    valid_config.touch()
    os.utime(invalid_config, (2, 2))
    os.utime(valid_config, (1, 1))
    attempts: list[Path] = []

    def select_completed_run(run_directory: Path, show_error: bool) -> bool:
        assert not show_error
        attempts.append(run_directory)
        return run_directory.name == "older-valid"

    monkeypatch.setattr(window, "_set_inference_run", select_completed_run)
    window._inference_run_directory = None
    window._load_default_inference_run(ProjectConfig(name="test", project_path=str(tmp_path)))

    assert attempts == [invalid_config.parent, valid_config.parent]
    window.close()


def test_default_dialog_directory_uses_documents_then_home(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))

    assert MainWindow._default_dialog_directory() == tmp_path

    documents_directory = tmp_path / "Documents"
    documents_directory.mkdir()
    assert MainWindow._default_dialog_directory() == documents_directory


def test_inspection_roi_is_ready_to_adjust_after_dataset_selection_before_training(tmp_path: Path) -> None:
    application = QApplication.instance() or QApplication([])
    manager = ProjectManager(tmp_path / "projects")
    window = MainWindow(SettingsManager(), manager)
    project = manager.create_project("roi-before-training")
    window._set_current_project(project)
    source_directory = tmp_path / "source"
    source_directory.mkdir()
    Image.new("RGB", (64, 64), (20, 30, 40)).save(source_directory / "first.png")
    Image.new("RGB", (64, 64), (30, 40, 50)).save(source_directory / "second.png")
    window.dataset_manager.assign_folder(
        project.dataset,
        DatasetRole.OK_TRAIN,
        source_directory,
        FolderImportMode.REFERENCE,
    )

    window._validate_dataset(show_dialog=False)
    window.inspection_region_page.enable_checkbox.setChecked(True)
    application.processEvents()

    assert window.inspection_region_page.source_label.text().startswith("first.png")
    assert window.inspection_region_page.overlay_canvas.isEnabled()
    assert window.inspection_region_page.save_button.isEnabled()
    assert project.last_training_status == "Not trained"

    window.close()