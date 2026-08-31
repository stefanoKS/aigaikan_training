"""Results page."""

from pathlib import Path

from app.models.training_run import TrainingRun
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
        export_format_row = QHBoxLayout()
        for export_format, label in (
            (ModelExportFormat.OPENVINO, "OpenVINO IR (.xml + .bin)"),
            (ModelExportFormat.ONNX, "ONNX (.onnx)"),
            (ModelExportFormat.TORCH, "Torch (.pt)"),
        ):
            checkbox = QCheckBox(label)
            checkbox.setChecked(export_format is ModelExportFormat.OPENVINO)
            self.export_format_checks[export_format] = checkbox
            export_format_row.addWidget(checkbox)
        export_format_row.addStretch(1)
        self.export_model_button = QPushButton("Export Selected Formats")
        self.export_model_button.setObjectName("PrimaryButton")
        self.export_model_button.setEnabled(False)
        export_form.addRow("Destination", export_directory_row)
        export_form.addRow("Formats", export_format_row)
        export_form.addRow("", self.export_model_button)
        root.addWidget(export_group)

        splitter = QSplitter()
        metrics_group = QGroupBox("Metrics")
        metrics_form = QFormLayout(metrics_group)
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
        ):
            label = QLabel("Not available")
            self.metric_labels[key] = label
            metrics_form.addRow(key, label)
        splitter.addWidget(metrics_group)

        right_column = QWidget()
        right_layout = QVBoxLayout(right_column)
        self.metrics_table = QTableWidget(0, 2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.gallery_table = QTableWidget(0, 7)
        self.gallery_table.setHorizontalHeaderLabels(
            ["Original", "Anomaly Map", "Overlay", "Predicted", "Ground Truth", "Score", "Source Path"]
        )
        right_layout.addWidget(self.metrics_table, stretch=1)
        right_layout.addWidget(self.gallery_table, stretch=2)
        splitter.addWidget(right_column)
        root.addWidget(splitter, stretch=1)
        self.current_run_directory: Path | None = None

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
        self.export_model_button.setEnabled(False)
        self.metrics_table.setRowCount(0)
        self.gallery_table.setRowCount(0)
        for label in self.metric_labels.values():
            label.setText("Not available")

    def set_training_run(self, run: TrainingRun) -> None:
        """Populate the completed-run metadata and metrics."""
        run_directory = Path(run.run_dir)
        self.current_run_directory = run_directory if run_directory.is_dir() else None
        self.export_model_button.setEnabled(self.current_run_directory is not None)
        metric_values = {name: self._format_value(value) for name, value in run.metrics.items()}
        self.set_metrics(metric_values)
        summary = {
            "Run Name": run.run_name,
            "Run Date": run.run_date.replace("T", " ").replace("+00:00", " UTC"),
            "Model": run.model_name,
            "Device": run.device.upper(),
            "Training Duration": self._format_duration(run.training_duration_seconds),
            "Evaluation Duration": self._format_duration(run.evaluation_duration_seconds),
        }
        for key in ("Image AUROC", "Image F1", "Precision", "Recall", "Threshold"):
            if key in metric_values:
                summary[key] = metric_values[key]
        for key, label in self.metric_labels.items():
            label.setText(summary.get(key, "Not available"))

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

