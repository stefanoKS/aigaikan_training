"""Main-window navigation layout tests."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from PIL import Image

from app.models.dataset_config import DatasetRole, FolderImportMode
from app.models.preprocessing_config import LEGACY_PREPROCESSING_CONTRACT_VERSION, PaddingPolicy, PreprocessingConfig
from app.core.preprocessing_contract import read_preprocessing_config
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


def test_saving_preprocessing_policy_marks_existing_results_for_retraining(tmp_path: Path) -> None:
    application = QApplication.instance() or QApplication([])
    manager = ProjectManager(tmp_path / "projects")
    window = MainWindow(SettingsManager(), manager)
    project = manager.create_project("preprocessing-policy")
    window._set_current_project(project)
    model_index = window.config_page.model_combo.findData("dinomaly_dinov3")
    window.config_page.model_combo.setCurrentIndex(model_index)
    window.config_page.tiling_check.setChecked(True)
    aggregation_index = window.config_page.score_aggregation_combo.findData("top_k_mean")
    window.config_page.score_aggregation_combo.setCurrentIndex(aggregation_index)
    window.config_page.top_k_fraction_spin.setValue(2.5)
    application.processEvents()

    assert window._save_training_config(show_dialog=False)

    persisted = read_preprocessing_config(project.root_path / "preprocessing.json")
    assert persisted.tiling.enabled
    assert persisted.score_aggregation.value == "top_k_mean"
    assert persisted.top_k_fraction == 0.025
    assert project.training.dinomaly_encoder_id == "vit_base_patch16_dinov3.lvd1689m"
    assert project.last_training_status == "Retraining required"
    assert window.inference_page.status_label.text() == "Retraining required"
    window.close()


def test_v3_padding_geometry_and_custom_policy_persist_and_invalid_sizes_are_blocked(tmp_path: Path) -> None:
    application = QApplication.instance() or QApplication([])
    manager = ProjectManager(tmp_path / "projects")
    window = MainWindow(SettingsManager(), manager)
    project = manager.create_project("dynamic-padding")
    window._set_current_project(project)
    source_directory = tmp_path / "source"
    source_directory.mkdir()
    Image.new("RGB", (321, 77), (20, 30, 40)).save(source_directory / "first.png")
    window.dataset_manager.assign_folder(
        project.dataset,
        DatasetRole.OK_TRAIN,
        source_directory,
        FolderImportMode.REFERENCE,
    )
    model_index = window.config_page.model_combo.findData("dinomaly_dinov3")
    window.config_page.model_combo.setCurrentIndex(model_index)
    application.processEvents()

    assert window.config_page.rectified_roi_size_label.text() == "321 x 77 px"
    assert window.config_page.automatic_right_padding_label.text() == "15 px"
    assert window.config_page.automatic_bottom_padding_label.text() == "3 px"
    assert window.config_page.prepared_image_size_label.text() == "336 x 80 px"

    custom_index = window.config_page.padding_policy_combo.findData(PaddingPolicy.CUSTOM.value)
    window.config_page.padding_policy_combo.setCurrentIndex(custom_index)
    window.config_page.custom_right_padding_spin.setValue(15)
    window.config_page.custom_bottom_padding_spin.setValue(3)
    assert window._save_training_config(show_dialog=False)

    persisted = read_preprocessing_config(project.root_path / "preprocessing.json")
    assert persisted.padding_policy is PaddingPolicy.CUSTOM
    assert persisted.custom_padding_right == 15
    assert persisted.custom_padding_bottom == 3

    window.config_page.custom_right_padding_spin.setValue(1)
    assert "nearest valid size 336x80" in window.config_page.padding_validation_label.text()
    assert not window._save_training_config(show_dialog=False)
    window.close()


def test_saving_training_settings_retains_a_legacy_v2_preprocessing_policy(tmp_path: Path) -> None:
    application = QApplication.instance() or QApplication([])
    manager = ProjectManager(tmp_path / "projects")
    window = MainWindow(SettingsManager(), manager)
    project = manager.create_project("legacy-padding")
    project.preprocessing = PreprocessingConfig(preprocessing_contract_version=LEGACY_PREPROCESSING_CONTRACT_VERSION)
    expected_policy = project.preprocessing.to_dict()
    manager.save_project(project)
    window._set_current_project(project)
    window.config_page.seed_spin.setValue(99)
    application.processEvents()

    assert not window.config_page.padding_policy_combo.isEnabled()
    assert window._save_training_config(show_dialog=False)

    persisted = read_preprocessing_config(project.root_path / "preprocessing.json")
    assert persisted.preprocessing_contract_version == LEGACY_PREPROCESSING_CONTRACT_VERSION
    assert persisted.to_dict() == expected_policy
    window.close()


def test_model_controls_only_allow_verified_runtime_settings() -> None:
    application = QApplication.instance() or QApplication([])
    window = MainWindow(SettingsManager(), ProjectManager(SettingsManager().default_projects_directory()))

    for model_name in ("anomaly_dino", "super_add"):
        window.config_page.model_combo.setCurrentIndex(window.config_page.model_combo.findData(model_name))
        application.processEvents()
        assert window.config_page.max_epochs_spin.value() == 1
        assert not window.config_page.max_epochs_spin.isEnabled()
    window.config_page.model_combo.setCurrentIndex(window.config_page.model_combo.findData("efficient_ad"))
    application.processEvents()
    assert window.config_page.batch_size_spin.value() == 1
    assert not window.config_page.batch_size_spin.isEnabled()
    window.close()