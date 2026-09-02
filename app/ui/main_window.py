"""Main application window."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

from PySide6.QtCore import QSize, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QFont, QFontDatabase, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.core.dataset_manager import DatasetManager
from app.core.dataset_manifest import build_effective_split
from app.core.dataset_validator import DatasetValidationReport, DatasetValidator
from app.core.inference_controller import InferenceController
from app.core.model_registry import ModelExecutionMode, ModelRegistry
from app.core.preprocessing_contract import preprocessing_hash
from app.core.run_comparison import compare_training_runs
from app.core.threshold_calibrator import ThresholdMethod
from app.core.project_manager import ProjectManager
from app.core.result_parser import ResultParser
from app.core.inspection_region import inspection_region_hash
from app.core.run_artifacts import (
    read_canonical_checkpoint,
    read_persisted_threshold,
    read_run_manifest,
    read_verified_inspection_region,
    read_verified_preprocessing_plan,
)
from app.core.settings_manager import SettingsManager
from app.core.training_controller import TrainingController
from app.models.dataset_config import DatasetRole, FolderImportMode, SUPPORTED_IMAGE_EXTENSIONS
from app.models.inspection_region import InspectionRegionConfig
from app.models.preprocessing_config import (
    LEGACY_PREPROCESSING_CONTRACT_VERSION,
    PaddingPolicy,
    PreprocessingConfig,
    ScoreAggregation,
    TilingConfig,
)
from app.models.project_config import ProjectConfig
from app.models.training_config import DeviceMode, TrainingConfig
from app.models.training_run import TrainingRun
from app.models.prediction_result import PredictionResult
from app.services.export_service import ExportService
from app.ui.pages.config_page import ConfigPage
from app.ui.pages.dataset_page import DatasetPage
from app.ui.pages.home_page import HomePage
from app.ui.pages.inference_page import InferencePage
from app.ui.pages.inspection_region_page import InspectionRegionPage
from app.ui.pages.results_page import ResultsPage
from app.ui.pages.training_page import TrainingPage
from app.ui.styles import APP_STYLE


class MainWindow(QMainWindow):
    """Main application shell with left navigation."""

    PAGE_DEFINITIONS = (
        ("Home / Projects", HomePage),
        ("Dataset", DatasetPage),
        ("Inspection Region", InspectionRegionPage),
        ("Training Configuration", ConfigPage),
        ("Training", TrainingPage),
        ("Results", ResultsPage),
        ("Inference", InferencePage),
    )

    def __init__(self, settings_manager: SettingsManager, project_manager: ProjectManager) -> None:
        super().__init__()
        self._configure_application_font()
        self.settings_manager = settings_manager
        self.project_manager = project_manager
        self.dataset_manager = DatasetManager()
        self.dataset_validator = DatasetValidator()
        self.model_registry = ModelRegistry()
        self.training_controller = TrainingController(self)
        self.inference_controller = InferenceController(self)
        self.result_parser = ResultParser()
        self.export_service = ExportService()
        self.current_project: ProjectConfig | None = None
        self._run_metrics: dict[str, str] = {}
        self._inference_run_directory: Path | None = None
        self._inference_input_path: Path | None = None
        self.setWindowTitle("Anomalib Trainer")
        self.resize(1400, 900)

        splitter = QSplitter()
        self.navigation = QListWidget()
        self.navigation.setObjectName("Navigation")
        self.navigation.setMinimumWidth(190)
        self.navigation.setMaximumWidth(240)
        self.pages = QStackedWidget()
        self.page_instances: dict[str, QWidget] = {}
        self.page_scroll_areas: dict[str, QScrollArea] = {}

        for index, (title, page_type) in enumerate(self.PAGE_DEFINITIONS):
            navigation_item = QListWidgetItem(title)
            navigation_item.setSizeHint(QSize(0, 46))
            self.navigation.addItem(navigation_item)
            page = ConfigPage(self.model_registry) if page_type is ConfigPage else page_type()
            page.setObjectName("WorkspacePage")
            self.page_instances[title] = page
            page_scroll_area = self._create_page_scroll_area(page)
            self.page_scroll_areas[title] = page_scroll_area
            self.pages.addWidget(page_scroll_area)
            if index == 0:
                self.navigation.setCurrentRow(0)

        self.navigation.currentRowChanged.connect(self._set_active_page)
        splitter.addWidget(self.navigation)
        splitter.addWidget(self.pages)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([220, 1180])

        shell = QWidget()
        shell.setObjectName("AppShell")
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)
        header = QWidget()
        header.setObjectName("AppHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(24, 14, 24, 14)
        brand_logo = QLabel()
        brand_logo.setObjectName("BrandLogo")
        brand_logo.setFixedSize(40, 40)
        brand_logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._set_brand_logo(brand_logo)
        brand = QLabel("ANOMALIB TRAINER")
        brand.setObjectName("BrandTitle")
        self.workspace_title = QLabel("PROJECT WORKSPACE")
        self.workspace_title.setObjectName("WorkspaceTitle")
        self.project_indicator = QLabel("NO PROJECT OPEN")
        self.project_indicator.setObjectName("ProjectIndicator")
        header_layout.addWidget(brand_logo)
        header_layout.addSpacing(10)
        header_layout.addWidget(brand)
        header_layout.addSpacing(16)
        header_layout.addWidget(self.workspace_title)
        header_layout.addStretch(1)
        header_layout.addWidget(self.project_indicator)
        shell_layout.addWidget(header)
        shell_layout.addWidget(splitter, stretch=1)
        self.setCentralWidget(shell)

        self.home_page = cast(HomePage, self.page_instances["Home / Projects"])
        self.dataset_page = cast(DatasetPage, self.page_instances["Dataset"])
        self.inspection_region_page = cast(InspectionRegionPage, self.page_instances["Inspection Region"])
        self.config_page = cast(ConfigPage, self.page_instances["Training Configuration"])
        self.training_page = cast(TrainingPage, self.page_instances["Training"])
        self.results_page = cast(ResultsPage, self.page_instances["Results"])
        self.inference_page = cast(InferencePage, self.page_instances["Inference"])
        self._connect_actions()
        self._set_active_page(0)
        self._refresh_project_views()

    def show_dependency_error(self, message: str, details: str = "") -> None:
        """Show a friendly dependency error."""
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Critical)
        dialog.setWindowTitle("Missing Dependencies")
        dialog.setText(message)
        if details:
            dialog.setDetailedText(details)
        dialog.exec()

    def _configure_application_font(self) -> None:
        """Load a Windows font when PySide does not discover system fonts itself."""
        for font_name in ("bahnschrift.ttf", "segoeui.ttf", "arial.ttf"):
            font_path = Path("C:/Windows/Fonts") / font_name
            if not font_path.is_file():
                continue
            font_id = QFontDatabase.addApplicationFont(str(font_path))
            if font_id < 0:
                continue
            font_families = QFontDatabase.applicationFontFamilies(font_id)
            if font_families:
                self.setFont(QFont(font_families[0], 10))
                return

    @staticmethod
    def _set_brand_logo(logo_label: QLabel) -> None:
        """Load the packaged AIGAIKAN brand mark into the application header."""
        logo_path = Path(__file__).resolve().parents[1] / "resources" / "icons" / "00_brand_logo.png"
        logo = QPixmap(str(logo_path))
        if not logo.isNull():
            logo_label.setPixmap(
                logo.scaled(
                    36,
                    36,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

    @staticmethod
    def _create_page_scroll_area(page: QWidget) -> QScrollArea:
        """Wrap a workspace page so its controls remain available at any viewport height."""
        scroll_area = QScrollArea()
        scroll_area.setObjectName("PageScrollArea")
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        page.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        scroll_area.viewport().setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setWidget(page)
        return scroll_area

    def _connect_actions(self) -> None:
        self.home_page.new_project_button.clicked.connect(self._create_project)
        self.home_page.open_project_button.clicked.connect(self._choose_project_to_open)
        self.home_page.save_project_button.clicked.connect(self._save_project)
        self.home_page.recent_projects_list.itemActivated.connect(self._open_recent_project)

        for role_name, widgets in self.dataset_page.role_widgets.items():
            role = DatasetRole(role_name)
            import_button = cast(QPushButton, widgets["import_button"])
            browse_button = cast(QPushButton, widgets["browse_button"])
            import_button.clicked.connect(lambda _checked=False, selected_role=role: self._choose_dataset_folder(selected_role))
            browse_button.clicked.connect(lambda _checked=False, selected_role=role: self._open_dataset_folder(selected_role))
        self.dataset_page.validate_button.clicked.connect(lambda: self._validate_dataset(show_dialog=True))
        self.dataset_page.clear_button.clicked.connect(self._clear_dataset)
        self.inspection_region_page.save_button.clicked.connect(self._save_inspection_region)

        self.config_page.save_button.clicked.connect(lambda: self._save_training_config(show_dialog=True))
        self.config_page.load_button.clicked.connect(self._refresh_config_page)
        self.config_page.reset_button.clicked.connect(self._reset_training_config)
        self.config_page.model_combo.currentIndexChanged.connect(self._update_model_action)
        self.config_page.model_combo.currentIndexChanged.connect(self._update_estimated_training_steps)
        self.config_page.batch_size_spin.valueChanged.connect(self._update_estimated_training_steps)
        self.config_page.max_epochs_spin.valueChanged.connect(self._update_estimated_training_steps)
        self.config_page.target_training_steps_spin.valueChanged.connect(self._update_estimated_training_steps)
        self.config_page.seed_spin.valueChanged.connect(self._sync_split_seed_on_initial_edit)
        self.config_page.model_combo.currentIndexChanged.connect(self._refresh_preprocessing_geometry)
        self.config_page.padding_policy_combo.currentIndexChanged.connect(self._refresh_preprocessing_geometry)
        self.config_page.custom_right_padding_spin.valueChanged.connect(self._refresh_preprocessing_geometry)
        self.config_page.custom_bottom_padding_spin.valueChanged.connect(self._refresh_preprocessing_geometry)
        self.config_page.tiling_check.toggled.connect(self._refresh_preprocessing_geometry)

        self.results_page.browse_export_directory_button.clicked.connect(self._choose_model_export_directory)
        self.results_page.export_model_button.clicked.connect(self._export_model)
        self.results_page.export_csv_button.clicked.connect(self._export_results_csv)
        self.results_page.export_json_button.clicked.connect(self._export_results_json)
        self.results_page.open_folder_button.clicked.connect(self._open_results_folder)
        self.results_page.compare_button.clicked.connect(self._compare_results)

        self.training_page.start_button.clicked.connect(self._start_training)
        self.training_page.cancel_button.clicked.connect(self.training_controller.cancel)
        self.training_page.open_log_button.clicked.connect(self._open_project_logs)
        self.training_controller.stage_changed.connect(self._update_training_stage)
        self.training_controller.progress_changed.connect(self._update_training_progress)
        self.training_controller.stage_progress_changed.connect(self._update_training_stage_progress)
        self.training_controller.log_message.connect(self._append_training_log)
        self.training_controller.metric_emitted.connect(self._record_metric)
        self.training_controller.completed.connect(self._training_completed)
        self.training_controller.failed.connect(self._training_failed)
        self.training_controller.running_changed.connect(self._set_training_running)

        self.inference_page.load_run_button.clicked.connect(self._choose_inference_run)
        self.inference_page.select_image_button.clicked.connect(self._choose_inference_image)
        self.inference_page.select_folder_button.clicked.connect(self._choose_inference_folder)
        self.inference_page.run_inference_button.clicked.connect(self._start_inference)
        self.inference_page.cancel_inference_button.clicked.connect(self.inference_controller.cancel)
        self.inference_page.export_csv_button.clicked.connect(self._export_inference_csv)
        self.inference_controller.log_message.connect(self._append_inference_log)
        self.inference_controller.progress_changed.connect(self.inference_page.set_progress)
        self.inference_controller.prediction_emitted.connect(self._record_inference_prediction)
        self.inference_controller.completed.connect(self._inference_completed)
        self.inference_controller.failed.connect(self._inference_failed)
        self.inference_controller.running_changed.connect(self.inference_page.set_running)

    def _create_project(self) -> None:
        name, accepted = QInputDialog.getText(self, "New Project", "Project name")
        if not accepted:
            return
        try:
            project = self.project_manager.create_project(name)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Could Not Create Project", str(exc))
            return
        self._set_current_project(project)

    def _choose_project_to_open(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Open Anomalib Project",
            str(self._default_dialog_directory()),
        )
        if selected:
            self._open_project_path(Path(selected))

    def _open_recent_project(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self._open_project_path(Path(str(path)))

    def _open_project_path(self, path: Path) -> None:
        try:
            project = self.project_manager.load_project(path)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Could Not Open Project", str(exc))
            return
        self._set_current_project(project)

    def _set_current_project(self, project: ProjectConfig) -> None:
        self.current_project = project
        self.results_page.set_default_export_directory(project.root_path)
        self._add_recent_project(project)
        self._refresh_project_views()

    def _choose_model_export_directory(self) -> None:
        default_directory = self.results_page.export_directory()
        if default_directory is None:
            default_directory = self.current_project.root_path if self.current_project else self._default_dialog_directory()
        selected = QFileDialog.getExistingDirectory(self, "Select Model Export Folder", str(default_directory))
        if selected:
            self.results_page.export_directory_edit.setText(selected)

    def _export_model(self) -> None:
        project = self.current_project
        if project is None:
            QMessageBox.information(self, "No Project", "Open a project before exporting a model.")
            return
        run_directory = self.results_page.current_run_directory
        if run_directory is None:
            QMessageBox.information(self, "No Training Run", "Complete or load a training run before exporting a model.")
            return
        export_directory = self.results_page.export_directory() or project.root_path
        try:
            report = self.export_service.export_model(
                run_directory,
                export_directory,
                self.results_page.selected_export_formats(),
            )
        except (OSError, ValueError, RuntimeError) as exc:
            QMessageBox.warning(self, "Model Export Failed", str(exc))
            return

        exported_lines = [f"{result.export_format.upper()}: {result.exported_path}" for result in report.exported]
        if report.package_directory is not None:
            exported_lines.insert(0, f"Deployment package: {report.package_directory}")
        failure_lines = [f"{export_format.upper()}: {message}" for export_format, message in report.failures.items()]
        message = "\n".join([*exported_lines, *failure_lines])
        if report.exported and report.failures:
            QMessageBox.warning(self, "Model Export Partially Completed", message)
        elif report.exported:
            QMessageBox.information(self, "Model Export Completed", message)
        else:
            QMessageBox.warning(self, "Model Export Failed", message or "No model formats were exported.")
        if self.results_page.current_run is not None:
            self.results_page.current_run.export_status = "Partial" if report.failures and report.exported else (
                "Exported" if report.exported else "Failed"
            )
            if report.exported:
                self.results_page.current_run.metrics["Export Size (bytes)"] = sum(
                    result.exported_path.stat().st_size for result in report.exported
                )
            parity_formats = ", ".join(result.export_format.upper() for result in report.exported)
            self.results_page.current_run.anomalib_export_parity_status = (
                f"Validated with Anomalib deployment inferencer: {parity_formats}"
                if parity_formats
                else "Not validated"
            )
            self.result_parser.write_training_run(
                run_directory / "results.json",
                self.results_page.current_run,
            )
            self.results_page.set_training_run(self.results_page.current_run)

    def _export_results_csv(self) -> None:
        run = self.results_page.current_run
        if run is None:
            QMessageBox.information(self, "No Training Run", "Complete or load a training run before exporting results.")
            return
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results CSV",
            str(self._default_dialog_directory() / f"{run.run_name}_predictions.csv"),
            "CSV Files (*.csv)",
        )
        if selected:
            self.result_parser.export_predictions_csv(Path(selected), self.results_page.filtered_predictions())

    def _export_results_json(self) -> None:
        run = self.results_page.current_run
        if run is None:
            QMessageBox.information(self, "No Training Run", "Complete or load a training run before exporting results.")
            return
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results JSON",
            str(self._default_dialog_directory() / f"{run.run_name}_results.json"),
            "JSON Files (*.json)",
        )
        if selected:
            self.result_parser.write_training_run(Path(selected), run)

    def _open_results_folder(self) -> None:
        directory = self.results_page.current_run_directory
        if directory is None:
            QMessageBox.information(self, "No Training Run", "Complete or load a training run before opening results.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))

    def _compare_results(self) -> None:
        run = self.results_page.current_run
        if run is None:
            QMessageBox.information(self, "No Training Run", "Complete or load a training run before comparing results.")
            return
        selected_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Two or More Results JSON Files to Compare",
            str(self._default_dialog_directory()),
            "JSON Files (*.json)",
        )
        if not selected_paths:
            return
        try:
            comparisons = [self.result_parser.read_training_run(Path(selected)) for selected in selected_paths]
            report = compare_training_runs([run, *comparisons])
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Could Not Compare Runs", str(exc))
            return
        lines = [report.reason]
        for metric_name, values in report.metric_rows.items():
            rendered_values = " | ".join(
                f"{comparison_run.run_name}: {value if value is not None else 'NOT MEASURED'}"
                for comparison_run, value in zip(report.runs, values, strict=True)
            )
            lines.append(f"{metric_name}: {rendered_values}")
        message = "\n".join(lines)
        if report.direct_quality_comparison_allowed:
            QMessageBox.information(self, "Run Comparison", message)
        else:
            QMessageBox.warning(self, "Run Comparison Evidence Warning", message)

    def _add_recent_project(self, project: ProjectConfig) -> None:
        for index in range(self.home_page.recent_projects_list.count()):
            item = self.home_page.recent_projects_list.item(index)
            if item.data(Qt.ItemDataRole.UserRole) == project.project_path:
                item.setText(project.name)
                return
        item = QListWidgetItem(project.name)
        item.setData(Qt.ItemDataRole.UserRole, project.project_path)
        self.home_page.recent_projects_list.insertItem(0, item)

    def _save_project(self, show_dialog: bool = True) -> None:
        if self.current_project is None:
            QMessageBox.information(self, "No Project", "Create or open a project first.")
            return
        try:
            self.project_manager.save_project(self.current_project)
        except OSError as exc:
            QMessageBox.warning(self, "Could Not Save Project", str(exc))
            return
        if show_dialog:
            QMessageBox.information(self, "Project Saved", "Project settings were saved.")

    def _refresh_project_views(self) -> None:
        project = self.current_project
        self.home_page.save_project_button.setEnabled(project is not None)
        if project is None:
            self.home_page.project_name_label.setText("-")
            self.home_page.project_path_label.setText("-")
            self.home_page.created_date_label.setText("-")
            self.home_page.last_opened_label.setText("-")
            self.home_page.status_label.setText("Not trained")
            self.project_indicator.setText("NO PROJECT OPEN")
        else:
            self.home_page.project_name_label.setText(project.name)
            self.home_page.project_path_label.setText(project.project_path)
            self.home_page.created_date_label.setText(project.created_at)
            self.home_page.last_opened_label.setText(project.last_opened_at)
            self.home_page.status_label.setText(project.last_training_status)
            self.project_indicator.setText(project.name.upper())
        self._refresh_dataset_page()
        self._refresh_inspection_region_page()
        self._refresh_config_page()
        self._load_latest_results(project)
        self._load_default_inference_run(project)

    @staticmethod
    def _default_dialog_directory() -> Path:
        """Return a familiar starting directory for dialogs without a project path."""
        home_directory = Path.home()
        documents_directory = home_directory / "Documents"
        return documents_directory if documents_directory.is_dir() else home_directory

    def _set_active_page(self, index: int) -> None:
        """Keep the workspace header aligned with the selected navigation page."""
        self.pages.setCurrentIndex(index)
        if 0 <= index < len(self.PAGE_DEFINITIONS):
            self.workspace_title.setText(self.PAGE_DEFINITIONS[index][0].upper())

    def _choose_dataset_folder(self, role: DatasetRole) -> None:
        project = self.current_project
        if project is None:
            QMessageBox.information(self, "No Project", "Create or open a project before selecting data folders.")
            return
        selected = QFileDialog.getExistingDirectory(self, f"Select {role.value} folder", str(project.root_path))
        if not selected:
            return
        source = Path(selected)
        import_mode = (
            FolderImportMode.COPY
            if self.dataset_page.import_mode_combo.currentIndex() == 0
            else FolderImportMode.REFERENCE
        )
        try:
            destination = project.root_path / "dataset" / role.value
            assigned_path = self.project_manager.import_dataset_folder(
                source,
                destination,
                copy_files=import_mode is FolderImportMode.COPY,
            )
            self.dataset_manager.assign_folder(project.dataset, role, assigned_path, import_mode)
            self._validate_dataset(show_dialog=False)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Could Not Select Folder", str(exc))

    def _open_dataset_folder(self, role: DatasetRole) -> None:
        project = self.current_project
        if project is None:
            return
        path = project.dataset.folders[role].resolved_path()
        if path is None or not path.is_dir():
            QMessageBox.information(self, "Folder Not Available", "No existing folder is selected for this dataset role.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))

    def _clear_dataset(self) -> None:
        if self.current_project is None:
            return
        self.dataset_manager.clear(self.current_project.dataset)
        self._save_project(show_dialog=False)
        self.dataset_page.set_validation_rows([])
        self.dataset_page.validation_summary.clear()
        self._refresh_dataset_page()
        self._refresh_inspection_region_page()

    def _validate_dataset(self, show_dialog: bool) -> DatasetValidationReport | None:
        project = self.current_project
        if project is None:
            if show_dialog:
                QMessageBox.information(self, "No Project", "Create or open a project before validating data.")
            return None
        report = self.dataset_validator.validate(project.dataset, project.inspection_region)
        self._update_dataset_metadata(report)
        rows = [
            (issue.level, issue.role, issue.message, issue.path)
            for issue in [*report.errors, *report.warnings]
        ]
        self.dataset_page.set_validation_rows(rows)
        self.dataset_page.validation_summary.setPlainText(
            f"{len(report.errors)} error(s), {len(report.warnings)} warning(s)."
        )
        try:
            split = build_effective_split(project.dataset, project.training.split_seed)
            counts = split.counts()
            self.dataset_page.effective_split_summary.setPlainText(
                "Training\n"
                f"OK: {counts['training']['ok']}\n\n"
                "Validation\n"
                f"OK: {counts['validation']['ok']}  NG: {counts['validation']['ng']}\n\n"
                "Final Test\n"
                f"OK: {counts['final_test']['ok']}  NG: {counts['final_test']['ng']}\n\n"
                f"Split Seed: {split.seed}\n"
                f"Evaluation Method: {split.evaluation_method}"
            )
        except ValueError as exc:
            self.dataset_page.effective_split_summary.setPlainText(f"Split unavailable: {exc}")
        self._save_project(show_dialog=False)
        self._refresh_dataset_page()
        self._refresh_inspection_region_page()
        if show_dialog:
            if report.is_valid:
                QMessageBox.information(self, "Dataset Ready", "The selected dataset is ready for training.")
            else:
                QMessageBox.warning(self, "Dataset Needs Attention", "Resolve the listed errors before training.")
        return report

    def _update_dataset_metadata(self, report: DatasetValidationReport) -> None:
        if self.current_project is None:
            return
        invalid_messages = {"Unsupported file type", "Corrupt image"}
        for role, folder in self.current_project.dataset.folders.items():
            stats = report.stats.get(role.value, {})
            folder.image_count = int(stats.get("image_count", 0))
            folder.typical_resolution = str(stats.get("typical_resolution", ""))
            folder.color_mode = str(stats.get("color_mode", ""))
            folder.invalid_image_count = sum(
                1
                for issue in report.errors
                if issue.role == role.value and issue.message in invalid_messages
            )
            thumbnail_paths = stats.get("thumbnail_paths", [])
            folder.thumbnail_paths = [str(path) for path in thumbnail_paths] if isinstance(thumbnail_paths, list) else []

    def _refresh_dataset_page(self) -> None:
        project = self.current_project
        for role_name, widgets in self.dataset_page.role_widgets.items():
            folder = project.dataset.folders[DatasetRole(role_name)] if project else None
            cast(QLabel, widgets["path"]).setText(folder.path if folder and folder.path else "-")
            cast(QLabel, widgets["count"]).setText(str(folder.image_count) if folder else "0")
            cast(QLabel, widgets["invalid"]).setText(str(folder.invalid_image_count) if folder else "0")
            cast(QLabel, widgets["resolution"]).setText(folder.typical_resolution if folder and folder.typical_resolution else "-")
            cast(QLabel, widgets["color"]).setText(folder.color_mode if folder and folder.color_mode else "-")
            self._set_dataset_preview(cast(QLabel, widgets["preview"]), folder.thumbnail_paths if folder else [])

    def _refresh_inspection_region_page(self) -> None:
        """Load configured original images into the ROI editor without changing project data."""
        project = self.current_project
        source_paths: list[Path] = []
        if project is not None:
            for role, folder in project.dataset.folders.items():
                if role is DatasetRole.MASKS:
                    continue
                directory = folder.resolved_path()
                if directory is not None and directory.is_dir():
                    source_paths.extend(
                        path.resolve()
                        for path in directory.rglob("*")
                        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
                    )
        self.inspection_region_page.set_inspection_region(
            project.inspection_region if project is not None else InspectionRegionConfig()
        )
        self.inspection_region_page.set_dataset_images(tuple(dict.fromkeys(sorted(source_paths))))

    def _save_inspection_region(self) -> None:
        project = self.current_project
        if project is None:
            QMessageBox.information(self, "No Project", "Create or open a project before saving an inspection ROI.")
            return
        try:
            inspection_region = self.inspection_region_page.inspection_region()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Inspection ROI", str(exc))
            return
        roi_errors = [
            issue
            for issue in self.dataset_validator.validate(project.dataset, inspection_region).errors
            if issue.role == "inspection_region"
        ]
        if roi_errors:
            QMessageBox.warning(self, "Invalid Inspection ROI", "\n".join(issue.message for issue in roi_errors[:3]))
            return
        previous_hash = inspection_region_hash(project.inspection_region)
        project.inspection_region = inspection_region
        if inspection_region_hash(inspection_region) != previous_hash:
            self._mark_retraining_required()
        self._save_project(show_dialog=False)
        self._refresh_project_views()
        QMessageBox.information(self, "Inspection ROI Saved", "The fixed inspection region was saved to the project.")

    @staticmethod
    def _set_dataset_preview(preview: QLabel, thumbnail_paths: list[str]) -> None:
        """Display the first validated image for a selected dataset role."""
        preview.setPixmap(QPixmap())
        preview.setToolTip("")
        if not thumbnail_paths:
            preview.setText("No preview\navailable")
            return
        image_path = thumbnail_paths[0]
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            preview.setText("Preview\nunavailable")
            return
        preview.setPixmap(
            pixmap.scaled(
                preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        preview.setToolTip(image_path)

    def _save_training_config(self, show_dialog: bool) -> bool:
        project = self.current_project
        if project is None:
            if show_dialog:
                QMessageBox.information(self, "No Project", "Create or open a project before saving training settings.")
            return False
        config = project.training
        previous_preprocessing_hash = preprocessing_hash(project.preprocessing)
        config.model_name = str(self.config_page.model_combo.currentData())
        config.device = DeviceMode(self.config_page.device_combo.currentText().lower())
        config.batch_size = self.config_page.batch_size_spin.value()
        config.max_epochs = self.config_page.max_epochs_spin.value()
        config.validation_every_n_epochs = self.config_page.validation_every_n_epochs_spin.value()
        config.gradient_clip_val = self.config_page.gradient_clip_spin.value()
        config.accumulate_grad_batches = self.config_page.accumulate_grad_batches_spin.value()
        config.random_seed = self.config_page.seed_spin.value()
        config.split_seed = self.config_page.split_seed_spin.value()
        target_training_steps = self.config_page.target_training_steps_spin.value()
        config.target_training_steps = target_training_steps or None
        config.dinomaly_encoder_id = (
            str(self.config_page.dinomaly_encoder_combo.currentData() or "") if config.is_dinomaly else ""
        )
        config.num_workers = self.config_page.workers_spin.value()
        config.threshold_method = ThresholdMethod(str(self.config_page.threshold_method_combo.currentData()))
        config.target_normal_false_reject_rate = self.config_page.threshold_false_reject_rate()
        config.minimum_required_ng_recall = (
            self.config_page.minimum_ng_recall_spin.value() / 100
            if self.config_page.minimum_ng_recall_check.isChecked()
            else None
        )
        config.pixel_threshold_enabled = self.config_page.pixel_threshold_check.isChecked()
        config.pixel_threshold = self.config_page.pixel_threshold_spin.value()
        config.maximum_final_test_false_reject_rate = self.config_page.maximum_final_test_false_reject_spin.value() / 100
        config.minimum_final_test_ok_images = self.config_page.minimum_final_test_ok_images_spin.value()
        config.minimum_final_test_ng_images = self.config_page.minimum_final_test_ng_images_spin.value()
        preprocessing = project.preprocessing
        updated_preprocessing = self._preprocessing_from_page(preprocessing)
        config.apply_model_defaults(self._training_image_count())
        try:
            config.validate()
            updated_preprocessing.validate()
            rectified_size = self._effective_preprocessing_size()
            if rectified_size is not None:
                updated_preprocessing.resolve(config.model_name, rectified_size)
        except ValueError as exc:
            if show_dialog:
                QMessageBox.warning(self, "Invalid Training Settings", str(exc))
            return False
        project.preprocessing = updated_preprocessing
        if preprocessing_hash(updated_preprocessing) != previous_preprocessing_hash:
            self._mark_retraining_required()
        self._save_project(show_dialog=False)
        self._refresh_config_page()
        if show_dialog:
            QMessageBox.information(self, "Configuration Saved", "Training settings were saved to the project.")
        return True

    def _refresh_config_page(self) -> None:
        config = self.current_project.training if self.current_project else TrainingConfig()
        preprocessing = self.current_project.preprocessing if self.current_project else PreprocessingConfig()
        try:
            definition = self.model_registry.get(config.model_name)
        except ValueError:
            definition = self.model_registry.get("patchcore")
        model_index = self.config_page.model_combo.findData(definition.key)
        self.config_page.model_combo.setCurrentIndex(max(model_index, 0))
        self.config_page.device_combo.setCurrentText(config.device.value.title())
        self.config_page.batch_size_spin.setValue(config.batch_size)
        self.config_page.max_epochs_spin.setValue(config.max_epochs)
        self.config_page.validation_every_n_epochs_spin.setValue(config.validation_every_n_epochs)
        self.config_page.gradient_clip_spin.setValue(config.gradient_clip_val)
        self.config_page.accumulate_grad_batches_spin.setValue(config.accumulate_grad_batches)
        self.config_page.seed_spin.setValue(config.random_seed)
        self.config_page.split_seed_spin.setValue(config.split_seed)
        self.config_page.workers_spin.setValue(config.num_workers)
        self.config_page.target_training_steps_spin.setValue(config.target_training_steps or 0)
        self.config_page.set_dinomaly_encoder(config.dinomaly_encoder_name)
        threshold_method_index = self.config_page.threshold_method_combo.findData(config.threshold_method.value)
        self.config_page.threshold_method_combo.setCurrentIndex(max(threshold_method_index, 0))
        target_rate_index = self.config_page.threshold_fpr_combo.findData(config.target_normal_false_reject_rate)
        self.config_page.threshold_fpr_combo.setCurrentIndex(target_rate_index if target_rate_index >= 0 else 3)
        self.config_page.threshold_fpr_spin.setValue(config.target_normal_false_reject_rate * 100)
        self.config_page.minimum_ng_recall_check.setChecked(config.minimum_required_ng_recall is not None)
        if config.minimum_required_ng_recall is not None:
            self.config_page.minimum_ng_recall_spin.setValue(config.minimum_required_ng_recall * 100)
        self.config_page.pixel_threshold_check.setChecked(config.pixel_threshold_enabled)
        self.config_page.pixel_threshold_spin.setValue(config.pixel_threshold)
        self.config_page.maximum_final_test_false_reject_spin.setValue(config.maximum_final_test_false_reject_rate * 100)
        self.config_page.minimum_final_test_ok_images_spin.setValue(config.minimum_final_test_ok_images)
        self.config_page.minimum_final_test_ng_images_spin.setValue(config.minimum_final_test_ng_images)
        self.config_page.tiling_check.setChecked(preprocessing.tiling.enabled)
        aggregation_index = self.config_page.score_aggregation_combo.findData(preprocessing.score_aggregation.value)
        self.config_page.score_aggregation_combo.setCurrentIndex(max(aggregation_index, 0))
        self.config_page.top_k_fraction_spin.setValue(preprocessing.top_k_fraction * 100)
        self.config_page._update_threshold_controls()
        self.config_page._update_preprocessing_controls()
        self.config_page._update_model_support()
        if preprocessing.preprocessing_contract_version == LEGACY_PREPROCESSING_CONTRACT_VERSION:
            self.config_page.set_padding_policy(PaddingPolicy.AUTOMATIC, 0, 0, editable=False)
        else:
            self.config_page.set_padding_policy(
                preprocessing.padding_policy,
                preprocessing.custom_padding_right,
                preprocessing.custom_padding_bottom,
                editable=True,
            )
        self._refresh_preprocessing_geometry()
        self.training_page.active_model_label.setText(definition.display_name)
        self.training_page.active_device_label.setText(config.device.value)
        self._update_model_action()
        self._update_estimated_training_steps()

    def _training_image_count(self) -> int:
        """Return the deterministic training subset size when project data is available."""
        project = self.current_project
        if project is None:
            return 1
        try:
            return len(build_effective_split(project.dataset, project.training.split_seed).training_ok)
        except ValueError:
            return max(project.dataset.folders[DatasetRole.OK_TRAIN].image_count, 1)

    def _update_estimated_training_steps(self) -> None:
        """Keep the UI estimate aligned with the selected model and saved default policy."""
        model_key = str(self.config_page.model_combo.currentData())
        config = TrainingConfig(
            model_name=model_key,
            batch_size=self.config_page.batch_size_spin.value(),
            max_epochs=self.config_page.max_epochs_spin.value(),
            target_training_steps=self.config_page.target_training_steps_spin.value() or None,
        )
        config.apply_model_defaults(self._training_image_count())
        self.config_page.set_estimated_training_steps(
            config.estimated_training_steps(self._training_image_count()),
            config.max_epochs,
        )

    def _sync_split_seed_on_initial_edit(self) -> None:
        """Keep new projects deterministic until a user deliberately changes the split seed."""
        if self.config_page.split_seed_spin.value() == 42:
            self.config_page.split_seed_spin.setValue(self.config_page.seed_spin.value())

    def _update_model_action(self) -> None:
        try:
            definition = self.model_registry.get(str(self.config_page.model_combo.currentData()))
        except ValueError:
            return
        self.training_page.active_model_label.setText(definition.display_name)
        self.training_page.start_button.setText(
            "Run Evaluation" if definition.execution_mode is ModelExecutionMode.EVALUATE else "Start Training"
        )

    def _reset_training_config(self) -> None:
        if self.current_project is None:
            return
        previous_preprocessing_hash = preprocessing_hash(self.current_project.preprocessing)
        self.current_project.training = TrainingConfig()
        self.current_project.preprocessing = PreprocessingConfig()
        if preprocessing_hash(self.current_project.preprocessing) != previous_preprocessing_hash:
            self._mark_retraining_required()
        self._refresh_config_page()

    def _preprocessing_from_page(self, existing: PreprocessingConfig) -> PreprocessingConfig:
        """Read v3 operator settings while retaining legacy policy semantics byte-for-byte."""
        if existing.preprocessing_contract_version == LEGACY_PREPROCESSING_CONTRACT_VERSION:
            return existing
        return PreprocessingConfig(
            preprocessing_contract_version=existing.preprocessing_contract_version,
            padding_mode=existing.padding_mode,
            padding_policy=self.config_page.padding_policy(),
            padding_value_rgb=existing.padding_value_rgb,
            custom_padding_right=self.config_page.custom_right_padding_spin.value(),
            custom_padding_bottom=self.config_page.custom_bottom_padding_spin.value(),
            tiling=TilingConfig(
                enabled=self.config_page.tiling_check.isChecked(),
                tile_width=existing.tiling.tile_width,
                tile_height=existing.tiling.tile_height,
                overlap_x=existing.tiling.overlap_x,
                final_tile_alignment=existing.tiling.final_tile_alignment,
            ),
            score_aggregation=ScoreAggregation(str(self.config_page.score_aggregation_combo.currentData())),
            top_k_fraction=self.config_page.top_k_fraction_spin.value() / 100,
            aspect_ratio_tolerance=existing.aspect_ratio_tolerance,
        )

    def _effective_preprocessing_size(self) -> tuple[int, int] | None:
        """Return the current ROI size, or the single validated source resolution with no ROI."""
        project = self.current_project
        if project is None:
            return None
        if project.inspection_region.enabled:
            return project.inspection_region.rectified_size() if project.inspection_region.is_configured else None
        sizes: set[tuple[int, int]] = set()
        for role, folder in project.dataset.folders.items():
            if role is DatasetRole.MASKS:
                continue
            directory = folder.resolved_path()
            if directory is None or not directory.is_dir():
                continue
            for source_path in directory.rglob("*"):
                if not source_path.is_file() or source_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                    continue
                try:
                    from PIL import Image

                    with Image.open(source_path) as image:
                        sizes.add(image.size)
                except OSError:
                    continue
        return next(iter(sizes)) if len(sizes) == 1 else None

    def _refresh_preprocessing_geometry(self, *_args: object) -> None:
        """Show only geometry that the active preprocessing policy can actually resolve."""
        project = self.current_project
        if project is None:
            self.config_page.set_preprocessing_geometry(
                rectified_size=None,
                automatic_padding=None,
                prepared_size=None,
                alignment=None,
                allow_nearest_size=False,
            )
            return
        rectified_size = self._effective_preprocessing_size()
        preprocessing = project.preprocessing
        if preprocessing.preprocessing_contract_version == LEGACY_PREPROCESSING_CONTRACT_VERSION:
            self.config_page.set_preprocessing_geometry(
                rectified_size=rectified_size,
                automatic_padding=None,
                prepared_size=None,
                alignment=None,
                validation_message="Legacy preprocessing-v2 remains unchanged. Reset configuration to opt into dynamic padding.",
                allow_nearest_size=False,
            )
            return
        if rectified_size is None:
            self.config_page.set_preprocessing_geometry(
                rectified_size=None,
                automatic_padding=None,
                prepared_size=None,
                alignment=None,
                validation_message="Select a complete ROI or dataset images with one shared resolution.",
                allow_nearest_size=False,
            )
            return
        model_id = str(self.config_page.model_combo.currentData())
        page_policy = self._preprocessing_from_page(preprocessing)
        automatic_policy = replace(page_policy, padding_policy=PaddingPolicy.AUTOMATIC)
        try:
            automatic_plan = automatic_policy.resolve(model_id, rectified_size)
            plan = page_policy.resolve(model_id, rectified_size)
        except ValueError as exc:
            try:
                automatic_plan = automatic_policy.resolve(model_id, rectified_size)
                automatic_padding = automatic_plan.resolved_padding[2:]
                alignment = automatic_plan.model_alignment
            except ValueError:
                automatic_padding = None
                alignment = None
            self.config_page.set_preprocessing_geometry(
                rectified_size=rectified_size,
                automatic_padding=automatic_padding,
                prepared_size=None,
                alignment=alignment,
                validation_message=str(exc),
            )
            return
        self.config_page.set_preprocessing_geometry(
            rectified_size=rectified_size,
            automatic_padding=automatic_plan.resolved_padding[2:],
            prepared_size=plan.model_input_size,
            alignment=plan.model_alignment,
            validation_message="",
        )

    def _mark_retraining_required(self) -> None:
        """Clear current-only views after a project policy affecting model inputs changes."""
        if self.current_project is None:
            return
        self.current_project.last_training_status = "Retraining required"
        self.results_page.clear_results()
        self._inference_run_directory = None
        self.inference_page.set_training_run(Path("-"), "-")
        self.inference_page.set_status("Retraining required")

    def _start_training(self) -> None:
        project = self.current_project
        if project is None:
            QMessageBox.information(self, "No Project", "Create or open a project before starting training.")
            return
        if not self._save_training_config(show_dialog=False):
            return
        report = self._validate_dataset(show_dialog=False)
        if report is None or not report.is_valid:
            QMessageBox.warning(self, "Dataset Needs Attention", "Resolve dataset validation errors before training.")
            return
        self._run_metrics.clear()
        self.training_page.log_output.clear()
        self.training_page.current_stage_label.setText("Starting training")
        self.training_page.stage_progress.setRange(0, 0)
        self.training_page.dataset_counts_label.setText(
            f"OK: {project.dataset.folders[DatasetRole.OK_TRAIN].image_count}, "
            f"NG: {project.dataset.folders[DatasetRole.NG_TEST].image_count}"
        )
        try:
            self.training_controller.start(project.root_path / ProjectManager.PROJECT_FILE_NAME)
        except RuntimeError as exc:
            QMessageBox.warning(self, "Could Not Start Training", str(exc))

    def _open_project_logs(self) -> None:
        if self.current_project is None:
            return
        logs_path = self.current_project.root_path / "logs"
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(logs_path.resolve())))

    def _update_training_progress(self, current: int, total: int) -> None:
        self.training_page.overall_progress.setRange(0, max(total, 1))
        self.training_page.overall_progress.setValue(current)

    def _update_training_stage(self, stage: str) -> None:
        self.training_page.current_stage_label.setText(stage)
        self.training_page.stage_progress.setRange(0, 0)

    def _update_training_stage_progress(self, current: int, total: int) -> None:
        self.training_page.stage_progress.setRange(0, max(total, 1))
        self.training_page.stage_progress.setValue(min(current, max(total, 1)))

    def _append_training_log(self, level: str, message: str) -> None:
        self.training_page.append_log(level, message)

    def _record_metric(self, name: str, value: object) -> None:
        self._run_metrics[name] = str(value)
        self.results_page.set_metrics(self._run_metrics)

    def _set_training_running(self, running: bool) -> None:
        self.training_page.start_button.setEnabled(not running)
        self.training_page.cancel_button.setEnabled(running)

    def _training_completed(self, result_dir: str) -> None:
        run_path = Path(result_dir)
        try:
            completed_run = self.result_parser.read_training_run(run_path / "results.json")
        except (OSError, ValueError, TypeError):
            completed_run = TrainingRun(
                run_name=run_path.name,
                run_dir=str(run_path),
                model_name=self.current_project.training.model_name if self.current_project else "",
                device=self.current_project.training.device.value if self.current_project else "",
                metrics=self._run_metrics,
            )
        self.results_page.set_training_run(completed_run)
        if self.current_project is not None:
            self.current_project.last_training_status = "Completed"
            self._save_project(show_dialog=False)
            self._refresh_project_views()
        self.training_page.current_stage_label.setText("Completed")
        self.training_page.stage_progress.setRange(0, 1)
        self.training_page.stage_progress.setValue(1)
        self._append_training_log("info", f"Results saved to {result_dir}")
        self._set_inference_run(run_path, show_error=False)
        self.navigation.setCurrentRow(self.PAGE_DEFINITIONS.index(("Results", ResultsPage)))

    def _load_latest_results(self, project: ProjectConfig | None) -> None:
        """Restore the most recent completed run when a project is opened."""
        if project is None:
            self.results_page.clear_results()
            return
        expected_roi_hash = inspection_region_hash(project.inspection_region)
        expected_preprocessing_hash = preprocessing_hash(project.preprocessing)
        summaries = sorted(
            (project.root_path / "runs").glob("*/results.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not summaries:
            self.results_page.clear_results()
            return
        for summary_path in summaries:
            try:
                run = self.result_parser.read_training_run(summary_path)
                if (
                    run.inspection_region_hash == expected_roi_hash
                    and run.preprocessing_hash == expected_preprocessing_hash
                ):
                    self.results_page.set_training_run(run)
                    return
            except (OSError, ValueError, TypeError):
                continue
        self.results_page.clear_results()

    def _training_failed(self, message: str, details: str) -> None:
        if self.current_project is not None:
            self.current_project.last_training_status = "Failed"
            self._save_project(show_dialog=False)
            self._refresh_project_views()
        self._append_training_log("error", details or message)
        self.training_page.stage_progress.setRange(0, 1)
        self.training_page.stage_progress.setValue(0)
        QMessageBox.warning(self, "Training Failed", message)

    def _load_default_inference_run(self, project: ProjectConfig | None) -> None:
        """Select the newest saved training run when opening a project."""
        if project is None:
            self._inference_run_directory = None
            self.inference_page.set_training_run(Path("-"), "-")
            return
        if self._inference_run_directory is None or not self._inference_run_directory.is_relative_to(project.root_path):
            run_configs = sorted(
                (project.root_path / "runs").glob("*/config.json"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            for config_path in run_configs:
                if self._set_inference_run(config_path.parent, show_error=False):
                    return
            self._inference_run_directory = None
            self.inference_page.set_training_run(Path("-"), "-")

    def _choose_inference_run(self) -> None:
        initial_directory = (
            self.current_project.root_path / "runs" if self.current_project else self._default_dialog_directory()
        )
        selected = QFileDialog.getExistingDirectory(self, "Select Completed Training Run", str(initial_directory))
        if selected:
            self._set_inference_run(Path(selected), show_error=True)

    def _set_inference_run(self, run_directory: Path, show_error: bool) -> bool:
        run_directory = run_directory.expanduser().resolve()
        config_path = run_directory / "config.json"
        if not config_path.is_file():
            if show_error:
                QMessageBox.warning(
                    self,
                    "Invalid Training Run",
                    "Select a training run containing config.json and run_manifest.json.",
                )
            return False
        try:
            read_canonical_checkpoint(run_directory)
            read_persisted_threshold(run_directory)
            run_inspection_region = read_verified_inspection_region(run_directory)
            run_preprocessing_plan = read_verified_preprocessing_plan(run_directory)
            if self.current_project is not None and inspection_region_hash(run_inspection_region) != inspection_region_hash(
                self.current_project.inspection_region
            ):
                raise ValueError("The run inspection ROI does not match the current project. Train a new compatible run.")
            config = TrainingConfig.from_dict(__import__("json").loads(config_path.read_text(encoding="utf-8")))
            model_name = self.model_registry.get(config.model_name).display_name
            preprocessing_status = "Ready"
            if run_preprocessing_plan is None:
                preprocessing_status = "Historical legacy preprocessing"
            elif self.current_project is not None:
                preprocessing_contract = read_run_manifest(run_directory).get("preprocessing_contract", {})
                if preprocessing_contract.get("project_policy_sha256") != preprocessing_hash(self.current_project.preprocessing):
                    preprocessing_status = "Historical preprocessing policy"
        except (OSError, ValueError, TypeError):
            if show_error:
                QMessageBox.warning(
                    self,
                    "Invalid Training Run",
                    "Select a completed run with a manifest-verified canonical checkpoint and valid configuration.",
                )
            return False
        self._inference_run_directory = run_directory
        self.inference_page.set_training_run(run_directory, model_name)
        self.inference_page.set_status(preprocessing_status)
        return True

    def _choose_inference_image(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image for Inference",
            str(self.current_project.root_path if self.current_project else self._default_dialog_directory()),
            "Images (*.bmp *.jpeg *.jpg *.png *.tif *.tiff *.webp)",
        )
        if selected:
            self._set_inference_input(Path(selected))

    def _choose_inference_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select Image Folder for Inference",
            str(self.current_project.root_path if self.current_project else self._default_dialog_directory()),
        )
        if selected:
            self._set_inference_input(Path(selected))

    def _set_inference_input(self, input_path: Path) -> None:
        self._inference_input_path = input_path.resolve()
        self.inference_page.set_input_path(self._inference_input_path)
        self.inference_page.set_status("Ready")

    def _start_inference(self) -> None:
        if self._inference_run_directory is None:
            QMessageBox.information(self, "No Training Run", "Load a completed training run first.")
            return
        if self._inference_input_path is None:
            QMessageBox.information(self, "No Inference Input", "Select an image or image folder first.")
            return
        self.inference_page.clear_predictions()
        self.inference_page.clear_log()
        self.inference_page.set_progress(0, 1)
        self.inference_page.set_status("Running inference")
        try:
            self.inference_controller.start(self._inference_run_directory, self._inference_input_path)
        except RuntimeError as exc:
            QMessageBox.warning(self, "Could Not Start Inference", str(exc))

    def _append_inference_log(self, level: str, message: str) -> None:
        self.inference_page.append_log(level, message)
        if level == "error":
            self.inference_page.set_status("Inference failed")
        elif message:
            self.inference_page.set_status(message.splitlines()[0])

    def _record_inference_prediction(self, prediction: PredictionResult) -> None:
        self.inference_page.append_prediction(prediction)

    def _inference_completed(self, output_directory: str) -> None:
        self.inference_page.set_status(f"Completed: {Path(output_directory).name}")
        self.inference_page.set_progress(len(self.inference_page.predictions), len(self.inference_page.predictions))
        self.inference_page.append_log("info", f"Results saved to {output_directory}")

    def _inference_failed(self, message: str, details: str) -> None:
        self.inference_page.set_status("Inference failed")
        self.inference_page.append_log("error", details or message)
        QMessageBox.warning(self, "Inference Failed", message)

    def _export_inference_csv(self) -> None:
        if not self.inference_page.predictions:
            QMessageBox.information(self, "No Inference Results", "Run inference before exporting predictions.")
            return
        initial_directory = (
            self.current_project.root_path / "exports" if self.current_project else self._default_dialog_directory()
        )
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Export Inference Results",
            str(initial_directory / "inference_predictions.csv"),
            "CSV files (*.csv)",
        )
        if not selected:
            return
        try:
            path = self.result_parser.export_predictions_csv(Path(selected), self.inference_page.predictions)
        except OSError as exc:
            QMessageBox.warning(self, "Could Not Export Results", str(exc))
            return
        self.inference_page.set_status(f"Exported {path.name}")

