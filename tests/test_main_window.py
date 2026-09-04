"""Main-window navigation layout tests."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QMessageBox
from PIL import Image

from app.models.dataset_config import DatasetRole, FolderImportMode
from app.models.preprocessing_config import LEGACY_PREPROCESSING_CONTRACT_VERSION, PaddingPolicy, PreprocessingConfig
from app.models.image_preprocessing import ColorMode, ImagePreprocessingConfig
from app.models.preprocessing_preview import PreprocessingPreviewState, PreviewSource
from app.core.preprocessing_contract import read_preprocessing_config
from app.core.threshold_contract import ImageThresholdOperatingPoint, PixelThresholdOperatingPoint
from app.services.threshold_revision_service import ThresholdRevisionResult
from app.ui import main_window as main_window_module
from app.core.project_manager import ProjectManager
from app.core.settings_manager import SettingsManager
from app.models.project_config import ProjectConfig
from app.models.training_run import TrainingRun
from app.models.prediction_result import PredictionResult
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
    assert "QRadioButton {\n    color: #e7eff0;" in APP_STYLE
    assert "QRadioButton::indicator:checked" in APP_STYLE


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


def test_saving_image_preprocessing_profile_marks_results_stale_but_preview_source_does_not(
    tmp_path: Path, monkeypatch
) -> None:
    application = QApplication.instance() or QApplication([])
    manager = ProjectManager(tmp_path / "projects")
    window = MainWindow(SettingsManager(), manager)
    project = manager.create_project("image-profile")
    window._set_current_project(project)
    project.last_training_status = "Completed"
    original_hash = read_preprocessing_config(project.root_path / "preprocessing.json")
    monkeypatch.setattr(QMessageBox, "information", lambda *_args, **_kwargs: None)

    window._save_preprocessing_preview_state(
        PreprocessingPreviewState(source=PreviewSource.CUSTOM_IMAGE, custom_image_path=r"C:\preview\custom.png")
    )

    assert project.last_training_status == "Completed"
    assert read_preprocessing_config(project.root_path / "preprocessing.json") == original_hash

    window._save_image_preprocessing_profile(
        ImagePreprocessingConfig(profile_id="gray-v1", color_mode=ColorMode.GRAYSCALE_REPLICATED_RGB)
    )

    assert project.last_training_status == "Retraining required"
    assert read_preprocessing_config(project.root_path / "preprocessing.json").image_preprocessing.color_mode is ColorMode.GRAYSCALE_REPLICATED_RGB
    assert application is not None
    window.close()
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


def test_results_json_export_view_uses_displayed_threshold_revision(tmp_path: Path, monkeypatch) -> None:
    application = QApplication.instance() or QApplication([])
    window = MainWindow(SettingsManager(), ProjectManager(tmp_path / "projects"))
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    revision_path = run_directory / "threshold_revisions" / "threshold-001.json"
    revision_path.parent.mkdir()
    predictions_path = revision_path.with_name("threshold-001_predictions.csv")
    predictions_path.write_text("image_path\n", encoding="utf-8")
    revision = ThresholdRevisionResult(
        revision_path,
        predictions_path,
        ImageThresholdOperatingPoint(0.8),
        PixelThresholdOperatingPoint(enabled=True, threshold=0.85),
    )
    run = TrainingRun(
        run_name="run",
        run_dir=str(run_directory),
        model_name="PatchCore",
        device="cpu",
        threshold_metadata={"threshold_value": 0.5, "threshold_revision": "evaluation-001"},
        predictions=[
            PredictionResult(
                source_path="canonical.png",
                predicted_label="NG",
                ground_truth_label="OK",
                anomaly_score=0.7,
                threshold=0.5,
            )
        ],
    )
    revised_prediction = PredictionResult(
        source_path="revision.png",
        predicted_label="OK",
        ground_truth_label="OK",
        anomaly_score=0.7,
        threshold=0.8,
    )
    window.results_page.set_training_run(run)
    window.results_page.display_threshold_revision("threshold-001", 0.8, 0.85, [revised_prediction])
    monkeypatch.setattr(window.threshold_revision_service, "read_active_revision", lambda _path: revision)

    exported = window._displayed_results_export_run(run)

    assert exported is not run
    assert exported.predictions == [revised_prediction]
    assert exported.threshold_metadata["threshold_value"] == 0.8
    assert exported.threshold_metadata["threshold_revision"] == "threshold-001"
    assert run.predictions[0].source_path == "canonical.png"
    assert application is not None
    window.close()


def test_saving_changed_superadd_backbone_or_precision_requires_retraining(tmp_path: Path) -> None:
    application = QApplication.instance() or QApplication([])
    manager = ProjectManager(tmp_path / "projects")
    window = MainWindow(SettingsManager(), manager)
    project = manager.create_project("superadd-contract")
    project.last_training_status = "Completed"
    window._set_current_project(project)
    window.config_page.model_combo.setCurrentIndex(window.config_page.model_combo.findData("super_add"))
    window.config_page.superadd_backbone_combo.setCurrentIndex(
        window.config_page.superadd_backbone_combo.findData("vit_small_plus_patch16_dinov3.lvd1689m")
    )
    window.config_page.superadd_precision_combo.setCurrentIndex(
        window.config_page.superadd_precision_combo.findData("float32")
    )

    assert window._save_training_config(show_dialog=False)
    assert project.training.superadd_backbone_id == "vit_small_plus_patch16_dinov3.lvd1689m"
    assert project.training.superadd_precision == "float32"
    assert project.last_training_status == "Retraining required"
    persisted = manager.load_project(project.root_path)
    assert persisted.training.superadd_backbone_id == "vit_small_plus_patch16_dinov3.lvd1689m"
    assert persisted.training.superadd_precision == "float32"

    project.last_training_status = "Completed"
    window.config_page.superadd_backbone_combo.setCurrentIndex(
        window.config_page.superadd_backbone_combo.findData("vit_base_patch16_dinov3.lvd1689m")
    )
    assert window._save_training_config(show_dialog=False)
    assert project.last_training_status == "Retraining required"
    assert application is not None
    window.close()


def test_inference_preview_save_revises_only_the_inference_selected_run_and_preserves_pixel_policy(
    tmp_path: Path, monkeypatch
) -> None:
    application = QApplication.instance() or QApplication([])
    window = MainWindow(SettingsManager(), ProjectManager(tmp_path / "projects"))
    inference_run = tmp_path / "inference-run"
    results_run = tmp_path / "results-run"
    inference_run.mkdir()
    results_run.mkdir()
    semantic = "anomalib_postprocessed_pred_score_v1"
    window._inference_run_directory = inference_run
    window.inference_page.set_training_run(inference_run, "PatchCore", 0.1, score_semantic=semantic)
    prediction = PredictionResult("inference.png", "NG", "Unknown", 0.12, 0.1, score_semantic=semantic)
    window.inference_page.append_prediction(prediction)
    window.inference_page.decision_preview_check.setChecked(True)
    window.inference_page.decision_preview_spin.setValue(0.15)
    window.results_page.current_run_directory = results_run
    window.results_page._predictions = [
        PredictionResult("results.png", "NG", "OK", 0.9, 0.5, score_semantic=semantic)
    ]
    captured: dict[str, object] = {}
    revision_path = inference_run / "threshold_revisions" / "threshold-001.json"
    predictions_path = revision_path.with_name("threshold-001_predictions.csv")
    revision = ThresholdRevisionResult(
        revision_path,
        predictions_path,
        ImageThresholdOperatingPoint(0.15, semantic),
        PixelThresholdOperatingPoint(enabled=True, threshold=0.85),
        operator_note="line trial",
    )
    monkeypatch.setattr(window.threshold_revision_service, "read_active_revision", lambda path: None)
    monkeypatch.setattr(main_window_module, "read_persisted_threshold", lambda path: 0.1)
    monkeypatch.setattr(
        main_window_module,
        "read_persisted_threshold_metadata",
        lambda path: {"threshold_value": 0.1, "score_semantic": semantic},
    )
    monkeypatch.setattr(
        window.threshold_revision_service,
        "preview_decision_threshold",
        lambda path, proposed, score_semantic: SimpleNamespace(
            active_threshold=0.1,
            proposed_threshold=proposed,
            score_semantic=score_semantic,
            ok_to_ng_changes=1,
            ng_to_ok_changes=2,
            false_reject_rate=0.2,
            ng_recall=0.8,
            outside_calibration_range=False,
        ),
    )

    def create_revision(path, image_operating_point, pixel_operating_point=None, operator_note=""):
        captured.update(
            {
                "run_directory": path,
                "image_operating_point": image_operating_point,
                "pixel_operating_point": pixel_operating_point,
                "operator_note": operator_note,
            }
        )
        return revision

    refreshed_results: list[object] = []
    monkeypatch.setattr(window.threshold_revision_service, "create_revision", create_revision)
    monkeypatch.setattr(window.result_parser, "read_predictions_csv", lambda _path: [])
    monkeypatch.setattr(window.results_page, "display_threshold_revision", lambda *args: refreshed_results.append(args))
    monkeypatch.setattr(QMessageBox, "question", lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes)

    window._save_inference_decision_revision(0.15, "line trial")

    assert captured["run_directory"] == inference_run
    assert captured["image_operating_point"] == ImageThresholdOperatingPoint(0.15, semantic)
    assert captured["pixel_operating_point"] is None
    assert captured["operator_note"] == "line trial"
    assert not refreshed_results
    assert window.inference_page.export_threshold() == pytest.approx(0.15)
    assert not window.inference_page.decision_preview_check.isChecked()
    assert window.inference_page.results_table.item(0, 5).text() == "OK"
    assert prediction.predicted_label == "NG"
    assert prediction.threshold == pytest.approx(0.1)
    assert application is not None
    window.close()


def test_inference_preview_save_refreshes_matching_results_run_and_keeps_active_threshold_on_failure(
    tmp_path: Path, monkeypatch
) -> None:
    application = QApplication.instance() or QApplication([])
    window = MainWindow(SettingsManager(), ProjectManager(tmp_path / "projects"))
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    semantic = "anomalib_postprocessed_pred_score_v1"
    window._inference_run_directory = run_directory
    window.results_page.current_run_directory = run_directory
    window.inference_page.set_training_run(run_directory, "PatchCore", 0.1, score_semantic=semantic)
    window.inference_page.append_prediction(PredictionResult("input.png", "NG", "Unknown", 0.12, 0.1, score_semantic=semantic))
    window.inference_page.decision_preview_check.setChecked(True)
    window.inference_page.decision_preview_spin.setValue(0.15)
    revision_path = run_directory / "threshold_revisions" / "threshold-001.json"
    revision = ThresholdRevisionResult(
        revision_path,
        revision_path.with_name("threshold-001_predictions.csv"),
        ImageThresholdOperatingPoint(0.15, semantic),
        PixelThresholdOperatingPoint(),
    )
    monkeypatch.setattr(window.threshold_revision_service, "read_active_revision", lambda path: None)
    monkeypatch.setattr(main_window_module, "read_persisted_threshold", lambda path: 0.1)
    monkeypatch.setattr(main_window_module, "read_persisted_threshold_metadata", lambda path: {"threshold_value": 0.1, "score_semantic": semantic})
    monkeypatch.setattr(
        window.threshold_revision_service,
        "preview_decision_threshold",
        lambda *_args: SimpleNamespace(ok_to_ng_changes=0, ng_to_ok_changes=1, false_reject_rate=None, ng_recall=None, outside_calibration_range=False),
    )
    monkeypatch.setattr(window.threshold_revision_service, "create_revision", lambda *_args, **_kwargs: revision)
    revised_predictions = [PredictionResult("final.png", "OK", "OK", 0.12, 0.15, score_semantic=semantic)]
    monkeypatch.setattr(window.result_parser, "read_predictions_csv", lambda _path: revised_predictions)
    refreshed_results: list[tuple[object, ...]] = []
    monkeypatch.setattr(window.results_page, "display_threshold_revision", lambda *args: refreshed_results.append(args))
    monkeypatch.setattr(QMessageBox, "question", lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes)

    window._save_inference_decision_revision(0.15, "")

    assert refreshed_results and refreshed_results[0][0] == "threshold-001"
    assert window.inference_page.export_threshold() == pytest.approx(0.15)

    window.inference_page.decision_preview_check.setChecked(True)
    window.inference_page.decision_preview_spin.setValue(0.2)
    monkeypatch.setattr(
        window.threshold_revision_service,
        "create_revision",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("cannot save")),
    )
    monkeypatch.setattr(QMessageBox, "warning", lambda *_args, **_kwargs: None)
    window._save_inference_decision_revision(0.2, "")

    assert window.inference_page.export_threshold() == pytest.approx(0.15)
    assert window.inference_page.decision_preview_spin.value() == pytest.approx(0.2)
    assert application is not None
    window.close()


def test_inference_preview_save_rejects_stale_score_semantics_before_revision(tmp_path: Path, monkeypatch) -> None:
    application = QApplication.instance() or QApplication([])
    window = MainWindow(SettingsManager(), ProjectManager(tmp_path / "projects"))
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    window._inference_run_directory = run_directory
    window.inference_page.set_training_run(
        run_directory,
        "PatchCore",
        0.1,
        score_semantic="anomalib_postprocessed_pred_score_v1",
    )
    window.inference_page.append_prediction(
        PredictionResult("input.png", "NG", "Unknown", 0.12, 0.1, score_semantic="anomalib_postprocessed_pred_score_v1")
    )
    monkeypatch.setattr(window.threshold_revision_service, "read_active_revision", lambda path: None)
    monkeypatch.setattr(main_window_module, "read_persisted_threshold", lambda path: 0.1)
    monkeypatch.setattr(
        main_window_module,
        "read_persisted_threshold_metadata",
        lambda path: {"threshold_value": 0.1, "score_semantic": "other_score_domain"},
    )
    calls: list[object] = []
    monkeypatch.setattr(window.threshold_revision_service, "create_revision", lambda *_args, **_kwargs: calls.append(True))
    monkeypatch.setattr(QMessageBox, "warning", lambda *_args, **_kwargs: None)

    window._save_inference_decision_revision(0.15, "")

    assert not calls
    assert window.inference_page.active_deployment_threshold == pytest.approx(0.1)
    assert application is not None
    window.close()