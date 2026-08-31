"""Main application window."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from PySide6.QtCore import Qt, QUrl
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
from app.core.project_manager import ProjectManager
from app.core.result_parser import ResultParser
from app.core.settings_manager import SettingsManager
from app.core.training_controller import TrainingController
from app.models.dataset_config import DatasetRole, FolderImportMode
from app.models.project_config import ProjectConfig
from app.models.training_config import DeviceMode, TrainingConfig
from app.models.training_run import TrainingRun
from app.models.prediction_result import PredictionResult
from app.services.export_service import ExportService
from app.ui.pages.config_page import ConfigPage
from app.ui.pages.dataset_page import DatasetPage
from app.ui.pages.home_page import HomePage
from app.ui.pages.inference_page import InferencePage
from app.ui.pages.results_page import ResultsPage
from app.ui.pages.training_page import TrainingPage
from app.ui.styles import APP_STYLE


class MainWindow(QMainWindow):
    """Main application shell with left navigation."""

    PAGE_DEFINITIONS = (
        ("Home / Projects", HomePage),
        ("Dataset", DatasetPage),
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
        self.setStyleSheet(APP_STYLE)

        splitter = QSplitter()
        self.navigation = QListWidget()
        self.navigation.setObjectName("Navigation")
        self.navigation.setMinimumWidth(190)
        self.navigation.setMaximumWidth(240)
        self.pages = QStackedWidget()
        self.page_instances: dict[str, QWidget] = {}
        self.page_scroll_areas: dict[str, QScrollArea] = {}

        for index, (title, page_type) in enumerate(self.PAGE_DEFINITIONS):
            self.navigation.addItem(QListWidgetItem(title))
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

        self.config_page.save_button.clicked.connect(lambda: self._save_training_config(show_dialog=True))
        self.config_page.load_button.clicked.connect(self._refresh_config_page)
        self.config_page.reset_button.clicked.connect(self._reset_training_config)
        self.config_page.browse_supplemental_button.clicked.connect(self._choose_supplemental_data)
        self.config_page.model_combo.currentIndexChanged.connect(self._update_model_action)
        self.config_page.model_combo.currentIndexChanged.connect(self._update_estimated_training_steps)
        self.config_page.batch_size_spin.valueChanged.connect(self._update_estimated_training_steps)
        self.config_page.max_epochs_spin.valueChanged.connect(self._update_estimated_training_steps)
        self.config_page.target_training_steps_spin.valueChanged.connect(self._update_estimated_training_steps)
        self.config_page.seed_spin.valueChanged.connect(self._sync_split_seed_on_initial_edit)

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
        selected = QFileDialog.getExistingDirectory(self, "Open Anomalib Project")
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
            default_directory = self.current_project.root_path if self.current_project else Path.home()
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
            torch_validated = any(result.export_format == "torch" for result in report.exported)
            self.results_page.current_run.aigaikan_compatibility_status = (
                "Validated with Anomalib TorchInferencer" if torch_validated else "Torch compatibility not requested"
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
        selected, _ = QFileDialog.getSaveFileName(self, "Export Results CSV", f"{run.run_name}_predictions.csv", "CSV Files (*.csv)")
        if selected:
            self.result_parser.export_predictions_csv(Path(selected), self.results_page.filtered_predictions())

    def _export_results_json(self) -> None:
        run = self.results_page.current_run
        if run is None:
            QMessageBox.information(self, "No Training Run", "Complete or load a training run before exporting results.")
            return
        selected, _ = QFileDialog.getSaveFileName(self, "Export Results JSON", f"{run.run_name}_results.json", "JSON Files (*.json)")
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
        selected, _ = QFileDialog.getOpenFileName(self, "Select Results JSON to Compare", "", "JSON Files (*.json)")
        if not selected:
            return
        try:
            comparison = self.result_parser.read_training_run(Path(selected))
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Could Not Compare Runs", str(exc))
            return
        shared_metric_names = sorted(set(run.metrics) & set(comparison.metrics))
        lines = [f"Current: {run.run_name}", f"Compared: {comparison.run_name}"]
        for name in shared_metric_names:
            lines.append(f"{name}: {run.metrics[name]} | {comparison.metrics[name]}")
        QMessageBox.information(self, "Run Comparison", "\n".join(lines))

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
        self._refresh_config_page()
        self._load_latest_results(project)
        self._load_default_inference_run(project)

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

    def _validate_dataset(self, show_dialog: bool) -> DatasetValidationReport | None:
        project = self.current_project
        if project is None:
            if show_dialog:
                QMessageBox.information(self, "No Project", "Create or open a project before validating data.")
            return None
        report = self.dataset_validator.validate(project.dataset)
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
                f"Split Seed: {split.seed}"
            )
        except ValueError as exc:
            self.dataset_page.effective_split_summary.setPlainText(f"Split unavailable: {exc}")
        self._save_project(show_dialog=False)
        self._refresh_dataset_page()
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
        config.model_name = str(self.config_page.model_combo.currentData())
        config.device = DeviceMode(self.config_page.device_combo.currentText().lower())
        config.image_width = self.config_page.image_width_spin.value()
        config.image_height = self.config_page.image_height_spin.value()
        config.batch_size = self.config_page.batch_size_spin.value()
        config.max_epochs = self.config_page.max_epochs_spin.value()
        config.validation_every_n_epochs = self.config_page.validation_every_n_epochs_spin.value()
        config.gradient_clip_val = self.config_page.gradient_clip_spin.value()
        config.accumulate_grad_batches = self.config_page.accumulate_grad_batches_spin.value()
        config.random_seed = self.config_page.seed_spin.value()
        config.split_seed = self.config_page.split_seed_spin.value()
        config.target_training_steps = self.config_page.target_training_steps_spin.value()
        config.coreset_sampling_ratio = self.config_page.coreset_ratio_spin.value()
        config.num_neighbors = self.config_page.neighbors_spin.value()
        config.num_workers = self.config_page.workers_spin.value()
        config.dinomaly_encoder = self.config_page.dinomaly_encoder_combo.currentText()
        config.dinomaly_decoder_depth = self.config_page.dinomaly_decoder_depth_spin.value()
        config.dinomaly_bottleneck_dropout = self.config_page.dinomaly_dropout_spin.value()
        config.dinomaly_context_recentering = self.config_page.dinomaly_context_recentering_check.isChecked()
        config.supplemental_data_path = self.config_page.supplemental_path_edit.text().strip()
        config.zero_shot_class_name = self.config_page.zero_shot_class_name_edit.text().strip()
        config.apply_model_defaults(self._training_image_count())
        try:
            config.validate()
        except ValueError as exc:
            if show_dialog:
                QMessageBox.warning(self, "Invalid Training Settings", str(exc))
            return False
        self._save_project(show_dialog=False)
        self._refresh_config_page()
        if show_dialog:
            QMessageBox.information(self, "Configuration Saved", "Training settings were saved to the project.")
        return True

    def _refresh_config_page(self) -> None:
        config = self.current_project.training if self.current_project else TrainingConfig()
        try:
            definition = self.model_registry.get(config.model_name)
        except ValueError:
            definition = self.model_registry.get("patchcore")
        model_index = self.config_page.model_combo.findData(definition.key)
        self.config_page.model_combo.setCurrentIndex(max(model_index, 0))
        self.config_page.device_combo.setCurrentText(config.device.value.title())
        self.config_page.image_width_spin.setValue(config.image_width)
        self.config_page.image_height_spin.setValue(config.image_height)
        self.config_page.batch_size_spin.setValue(config.batch_size)
        self.config_page.max_epochs_spin.setValue(config.max_epochs)
        self.config_page.validation_every_n_epochs_spin.setValue(config.validation_every_n_epochs)
        self.config_page.gradient_clip_spin.setValue(config.gradient_clip_val)
        self.config_page.accumulate_grad_batches_spin.setValue(config.accumulate_grad_batches)
        self.config_page.seed_spin.setValue(config.random_seed)
        self.config_page.split_seed_spin.setValue(config.split_seed)
        self.config_page.coreset_ratio_spin.setValue(config.coreset_sampling_ratio)
        self.config_page.neighbors_spin.setValue(config.num_neighbors)
        self.config_page.workers_spin.setValue(config.num_workers)
        encoder_index = self.config_page.dinomaly_encoder_combo.findText(config.dinomaly_encoder)
        self.config_page.dinomaly_encoder_combo.setCurrentIndex(max(encoder_index, 0))
        self.config_page.dinomaly_decoder_depth_spin.setValue(config.dinomaly_decoder_depth)
        self.config_page.dinomaly_dropout_spin.setValue(config.dinomaly_bottleneck_dropout)
        self.config_page.target_training_steps_spin.setValue(config.target_training_steps)
        self.config_page.dinomaly_context_recentering_check.setChecked(config.dinomaly_context_recentering)
        self.config_page.supplemental_path_edit.setText(config.supplemental_data_path)
        self.config_page.zero_shot_class_name_edit.setText(config.zero_shot_class_name)
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
            target_training_steps=self.config_page.target_training_steps_spin.value(),
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

    def _choose_supplemental_data(self) -> None:
        try:
            definition = self.model_registry.get(str(self.config_page.model_combo.currentData()))
        except ValueError:
            return
        if definition.key == "cfm":
            selected, _ = QFileDialog.getOpenFileName(self, "Select PointMAE Weights")
        else:
            selected = QFileDialog.getExistingDirectory(self, "Select Supplemental Model Data")
        if selected:
            self.config_page.supplemental_path_edit.setText(selected)

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
        self.current_project.training = TrainingConfig()
        self._refresh_config_page()

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
        self.training_page.log_output.appendPlainText(f"[{level.upper()}] {message}")

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
        summaries = sorted(
            (project.root_path / "runs").glob("*/results.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not summaries:
            self.results_page.clear_results()
            return
        try:
            self.results_page.set_training_run(self.result_parser.read_training_run(summaries[0]))
        except (OSError, ValueError, TypeError):
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
        run_configs = list((project.root_path / "runs").glob("*/config.json"))
        if not run_configs:
            return
        latest_config = max(run_configs, key=lambda path: path.stat().st_mtime)
        if self._inference_run_directory is None or not self._inference_run_directory.is_relative_to(project.root_path):
            self._set_inference_run(latest_config.parent, show_error=False)

    def _choose_inference_run(self) -> None:
        initial_directory = self.current_project.root_path / "runs" if self.current_project else Path.home()
        selected = QFileDialog.getExistingDirectory(self, "Select Completed Training Run", str(initial_directory))
        if selected:
            self._set_inference_run(Path(selected), show_error=True)

    def _set_inference_run(self, run_directory: Path, show_error: bool) -> bool:
        run_directory = run_directory.expanduser().resolve()
        checkpoint_exists = any(run_directory.glob("**/weights/lightning/*.ckpt"))
        config_path = run_directory / "config.json"
        if not config_path.is_file() or not checkpoint_exists:
            if show_error:
                QMessageBox.warning(
                    self,
                    "Invalid Training Run",
                    "Select a training run containing config.json and a Lightning model checkpoint.",
                )
            return False
        try:
            config = TrainingConfig.from_dict(__import__("json").loads(config_path.read_text(encoding="utf-8")))
            model_name = self.model_registry.get(config.model_name).display_name
        except (OSError, ValueError, TypeError):
            if show_error:
                QMessageBox.warning(self, "Invalid Training Run", "The training configuration could not be loaded.")
            return False
        self._inference_run_directory = run_directory
        self.inference_page.set_training_run(run_directory, model_name)
        self.inference_page.set_status("Ready")
        return True

    def _choose_inference_image(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image for Inference",
            str(self.current_project.root_path if self.current_project else Path.home()),
            "Images (*.bmp *.jpeg *.jpg *.png *.tif *.tiff *.webp)",
        )
        if selected:
            self._set_inference_input(Path(selected))

    def _choose_inference_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select Image Folder for Inference",
            str(self.current_project.root_path if self.current_project else Path.home()),
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
        self.inference_page.set_progress(0, 1)
        self.inference_page.set_status("Running inference")
        try:
            self.inference_controller.start(self._inference_run_directory, self._inference_input_path)
        except RuntimeError as exc:
            QMessageBox.warning(self, "Could Not Start Inference", str(exc))

    def _append_inference_log(self, level: str, message: str) -> None:
        if level == "error":
            self.inference_page.set_status("Inference failed")
        elif message:
            self.inference_page.set_status(message)

    def _record_inference_prediction(self, prediction: PredictionResult) -> None:
        self.inference_page.append_prediction(prediction)

    def _inference_completed(self, output_directory: str) -> None:
        self.inference_page.set_status(f"Completed: {Path(output_directory).name}")
        self.inference_page.set_progress(len(self.inference_page.predictions), len(self.inference_page.predictions))

    def _inference_failed(self, message: str, details: str) -> None:
        self.inference_page.set_status("Inference failed")
        QMessageBox.warning(self, "Inference Failed", details or message)

    def _export_inference_csv(self) -> None:
        if not self.inference_page.predictions:
            QMessageBox.information(self, "No Inference Results", "Run inference before exporting predictions.")
            return
        initial_directory = self.current_project.root_path / "exports" if self.current_project else Path.home()
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

