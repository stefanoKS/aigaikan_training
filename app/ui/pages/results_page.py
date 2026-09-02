"""Results page."""

from pathlib import Path

from app.models.training_run import TrainingRun
from app.models.prediction_result import PredictionResult
from app.services.export_service import ModelExportFormat

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class ResultsPage(QWidget):
    """Training results UI."""

    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 28)
        root.setSpacing(16)

        header = QHBoxLayout()
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(
            [
                "All",
                "Correct OK",
                "Correct NG",
                "False OK",
                "False NG",
                "Highest anomaly score",
                "Lowest anomaly score",
            ]
        )
        self.export_csv_button = QPushButton("Export Results CSV")
        self.export_json_button = QPushButton("Export Metrics JSON")
        self.open_folder_button = QPushButton("Open Result Folder")
        self.compare_button = QPushButton("Compare Runs")
        header.addWidget(QLabel("Filter"))
        header.addWidget(self.filter_combo)
        header.addStretch(1)
        header.addWidget(self.export_csv_button)
        header.addWidget(self.export_json_button)
        header.addWidget(self.open_folder_button)
        header.addWidget(self.compare_button)
        root.addLayout(header)

        export_group = QGroupBox("Model Export")
        export_form = QFormLayout(export_group)
        export_directory_row = QHBoxLayout()
        self.export_directory_edit = QLineEdit()
        self.export_directory_edit.setPlaceholderText("Project directory")
        self.browse_export_directory_button = QPushButton("Browse")
        export_directory_row.addWidget(self.export_directory_edit, stretch=1)
        export_directory_row.addWidget(self.browse_export_directory_button)
        self.export_format_checks: dict[ModelExportFormat, QCheckBox] = {}
        torch_checkbox = QCheckBox("Torch (.pt)")
        torch_checkbox.setChecked(True)
        self.export_format_checks[ModelExportFormat.TORCH] = torch_checkbox
        advanced_format_row = QHBoxLayout()
        for export_format, label in (
            (ModelExportFormat.OPENVINO, "OpenVINO IR (.xml + .bin)"),
            (ModelExportFormat.ONNX, "ONNX (.onnx)"),
        ):
            checkbox = QCheckBox(label)
            self.export_format_checks[export_format] = checkbox
            advanced_format_row.addWidget(checkbox)
        advanced_format_row.addStretch(1)
        self.export_model_button = QPushButton("Export for AIGAIKAN")
        self.export_model_button.setObjectName("PrimaryButton")
        self.export_model_button.setEnabled(False)
        export_form.addRow("Destination", export_directory_row)
        export_form.addRow("Export for AIGAIKAN", torch_checkbox)
        export_form.addRow("Advanced Formats", advanced_format_row)
        export_form.addRow("", self.export_model_button)
        root.addWidget(export_group)

        splitter = QSplitter()
        metrics_group = QGroupBox("Metrics")
        metrics_form = QFormLayout(metrics_group)
        self.no_ng_warning_label = QLabel(
            "NO GENUINE NG TEST DATA.\nDEFECT-DETECTION PERFORMANCE HAS NOT BEEN VERIFIED."
        )
        self.no_ng_warning_label.setObjectName("NoNgWarning")
        self.no_ng_warning_label.setWordWrap(True)
        self.no_ng_warning_label.setVisible(False)
        metrics_form.addRow(self.no_ng_warning_label)
        self.metric_labels: dict[str, QLabel] = {}
        for key in (
            "Run Name",
            "Run Date",
            "Model",
            "Device",
            "Training Duration",
            "Evaluation Duration",
            "Image AUROC",
            "Image F1",
            "Precision",
            "Recall",
            "Threshold",
            "OK Test Images",
            "NG Test Images",
            "True OK",
            "False NG",
            "True NG",
            "False OK",
            "Quality Status",
            "Defect Detection Evidence",
            "Threshold Method",
            "Threshold Revision",
            "Pixel Mask Threshold",
            "Calibration Images",
            "Calibration False Reject Target",
            "Calibration False Reject Observed",
            "Canonical Checkpoint",
            "Export Status",
            "Anomalib Export Parity",
            "AIGAIKAN Compatibility",
        ):
            label = QLabel("Not available")
            self.metric_labels[key] = label
            metrics_form.addRow(key, label)
        splitter.addWidget(metrics_group)

        right_column = QWidget()
        right_layout = QVBoxLayout(right_column)
        self.metrics_table = QTableWidget(0, 2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.gallery_table = QTableWidget(0, 11)
        self.gallery_table.setHorizontalHeaderLabels(
            [
                "Original",
                "Anomaly Map",
                "Overlay",
                "Continuous Map",
                "Pixel Mask",
                "Contour Overlay",
                "Pixel Threshold",
                "Predicted",
                "Ground Truth",
                "Score",
                "Source Path",
            ]
        )
        right_layout.addWidget(self.metrics_table, stretch=1)
        right_layout.addWidget(self.gallery_table, stretch=2)
        splitter.addWidget(right_column)
        root.addWidget(splitter, stretch=1)
        self.current_run_directory: Path | None = None
        self.current_run: TrainingRun | None = None
        self._predictions: list[PredictionResult] = []
        self.filter_combo.currentTextChanged.connect(self._apply_prediction_filter)

    def set_metrics(self, metrics: dict[str, str]) -> None:
        """Populate the metrics table."""
        items = list(metrics.items())
        self.metrics_table.setRowCount(len(items))
        for row_index, (key, value) in enumerate(items):
            self.metrics_table.setItem(row_index, 0, QTableWidgetItem(key))
            self.metrics_table.setItem(row_index, 1, QTableWidgetItem(value))

    def clear_results(self) -> None:
        """Clear the previous project's completed-run summary."""
        self.current_run_directory = None
        self.current_run = None
        self._predictions = []
        self.export_model_button.setEnabled(False)
        self.metrics_table.setRowCount(0)
        self.gallery_table.setRowCount(0)
        self.no_ng_warning_label.setVisible(False)
        for label in self.metric_labels.values():
            label.setText("Not available")

    def set_training_run(self, run: TrainingRun) -> None:
        """Populate the completed-run metadata and metrics."""
        self.current_run = run
        self._predictions = list(run.predictions)
        run_directory = Path(run.run_dir)
        self.current_run_directory = run_directory if run_directory.is_dir() else None
        self.export_model_button.setEnabled(self.current_run_directory is not None)
        metric_values = {name: self._format_value(value) for name, value in run.metrics.items()}
        no_ng_evidence = (
            run.metrics.get("Defect Detection Evidence") == "NOT MEASURED" or run.quality_status == "NOT VERIFIED"
        )
        self.no_ng_warning_label.setVisible(no_ng_evidence)
        self.set_metrics(metric_values)
        summary = {
            "Run Name": run.run_name,
            "Run Date": run.run_date.replace("T", " ").replace("+00:00", " UTC"),
            "Model": run.model_name,
            "Device": run.device.upper(),
            "Training Duration": self._format_duration(run.training_duration_seconds),
            "Evaluation Duration": self._format_duration(run.evaluation_duration_seconds),
            "Quality Status": run.quality_status or "Not available",
            "Canonical Checkpoint": Path(run.final_checkpoint_path).name if run.final_checkpoint_path else "Not available",
            "Export Status": run.export_status,
            "Anomalib Export Parity": run.anomalib_export_parity_status,
            "AIGAIKAN Compatibility": run.aigaikan_compatibility_status,
            "Defect Detection Evidence": run.metrics.get("Defect Detection Evidence", "Measured"),
            "Threshold Method": str(
                run.threshold_metadata.get("threshold_method", run.metrics.get("Threshold Method", "Not available"))
            ),
            "Threshold Revision": str(
                run.threshold_metadata.get("threshold_revision", run.metrics.get("Threshold Revision", "Not available"))
            ),
            "Pixel Mask Threshold": self._pixel_mask_threshold_text(run.threshold_metadata),
            "Calibration Images": self._format_value(run.metrics.get("Calibration Image Count")),
            "Calibration False Reject Target": self._format_value(
                run.metrics.get("Calibration Target False Reject Rate")
            ),
            "Calibration False Reject Observed": self._format_value(
                run.metrics.get("Calibration Observed False Reject Rate")
            ),
        }
        metric_summary_keys = {
            "AUROC": "Image AUROC",
            "F1": "Image F1",
            "Precision": "Precision",
            "Recall": "Recall",
            "Decision Threshold": "Threshold",
            "NG Tested": "NG Test Images",
            "OK Tested": "OK Test Images",
            "Actual OK -> Predicted OK": "True OK",
            "Actual OK -> Predicted NG": "False NG",
            "Actual NG -> Predicted NG": "True NG",
            "Actual NG -> Predicted OK (Escaped NG)": "False OK",
        }
        for metric_key, summary_key in metric_summary_keys.items():
            if metric_key in metric_values:
                summary[summary_key] = metric_values[metric_key]
        if no_ng_evidence:
            for key in ("Image AUROC", "Image F1", "Precision", "Recall", "NG Test Images", "True NG", "False OK"):
                summary[key] = "NOT MEASURED"
        for key, label in self.metric_labels.items():
            label.setText(summary.get(key, "Not available"))
        self._apply_prediction_filter()

    def filtered_predictions(self) -> list[PredictionResult]:
        """Return the rows currently selected by the Results filter."""
        selected_filter = self.filter_combo.currentText()
        if selected_filter == "Highest anomaly score":
            return sorted(self._predictions, key=lambda item: item.anomaly_score, reverse=True)
        if selected_filter == "Lowest anomaly score":
            return sorted(self._predictions, key=lambda item: item.anomaly_score)
        if selected_filter == "All":
            return list(self._predictions)
        return [item for item in self._predictions if item.classification_bucket() == selected_filter]

    def _apply_prediction_filter(self) -> None:
        """Render filtered final-test rows without losing the persisted run data."""
        predictions = self.filtered_predictions()
        self.gallery_table.setRowCount(len(predictions))
        for row_index, prediction in enumerate(predictions):
            values = (
                prediction.original_image or prediction.source_path,
                prediction.anomaly_map,
                prediction.overlay_image,
                prediction.continuous_anomaly_map,
                prediction.binary_mask,
                prediction.contour_overlay_image,
                self._format_pixel_threshold(prediction),
                prediction.predicted_label,
                prediction.ground_truth_label,
                f"{prediction.anomaly_score:.6g}",
                prediction.source_path,
            )
            for column_index, value in enumerate(values):
                self.gallery_table.setItem(row_index, column_index, QTableWidgetItem(value))

    def set_default_export_directory(self, directory: Path) -> None:
        """Set the active project's root as the default model export destination."""
        self.export_directory_edit.setText(str(directory))

    def export_directory(self) -> Path | None:
        """Return the selected model export destination, if one is set."""
        value = self.export_directory_edit.text().strip()
        return Path(value) if value else None

    def selected_export_formats(self) -> list[ModelExportFormat]:
        """Return the export formats selected by the user."""
        return [
            export_format
            for export_format, checkbox in self.export_format_checks.items()
            if checkbox.isChecked()
        ]

    @staticmethod
    def _format_value(value: float | str | None) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        return "Not available" if value is None else str(value)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total_seconds = max(round(seconds), 0)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, remaining_seconds = divmod(remainder, 60)
        return f"{hours:02}:{minutes:02}:{remaining_seconds:02}"

    @staticmethod
    def _pixel_mask_threshold_text(threshold_metadata: dict[str, object]) -> str:
        operating_point = threshold_metadata.get("pixel_operating_point")
        if not isinstance(operating_point, dict):
            return "Disabled"
        if not operating_point.get("enabled"):
            return "Disabled"
        threshold = operating_point.get("threshold")
        try:
            return f"{float(threshold):.6g} (map >= threshold)"
        except (TypeError, ValueError):
            return "Invalid"

    @staticmethod
    def _format_pixel_threshold(prediction: PredictionResult) -> str:
        if prediction.pixel_threshold is None:
            return "Not produced"
        return f"{prediction.pixel_threshold:.6g} (map >= threshold)"

