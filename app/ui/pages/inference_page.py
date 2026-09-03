"""Inference page."""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.models.prediction_result import PredictionResult


class InferencePage(QWidget):
    """Inference UI."""

    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 28)
        root.setSpacing(16)

        controls = QHBoxLayout()
        self.load_run_button = QPushButton("Load Training Run")
        self.select_image_button = QPushButton("Select Image")
        self.select_folder_button = QPushButton("Select Folder")
        self.run_inference_button = QPushButton("Run Inference")
        self.run_inference_button.setObjectName("PrimaryButton")
        self.cancel_inference_button = QPushButton("Cancel")
        self.cancel_inference_button.setObjectName("AlertButton")
        self.export_csv_button = QPushButton("Export Inference CSV")
        self.export_ng_images_button = QPushButton("Export NG Images")
        self.export_ng_images_button.setToolTip(
            "Exports selected NG rows, or every NG detection when no rows are selected."
        )
        controls.addWidget(self.load_run_button)
        controls.addWidget(self.select_image_button)
        controls.addWidget(self.select_folder_button)
        controls.addWidget(self.run_inference_button)
        controls.addWidget(self.cancel_inference_button)
        controls.addWidget(self.export_csv_button)
        controls.addWidget(self.export_ng_images_button)
        controls.addStretch(1)
        root.addLayout(controls)

        summary_group = QGroupBox("Inference Summary")
        summary_form = QFormLayout(summary_group)
        self.model_label = QLabel("-")
        self.score_label = QLabel("-")
        self.prediction_label = QLabel("-")
        self.threshold_label = QLabel("-")
        self.export_threshold_check = QCheckBox("Use custom export threshold")
        self.export_threshold_spin = QDoubleSpinBox()
        self.export_threshold_spin.setRange(-1_000_000_000.0, 1_000_000_000.0)
        self.export_threshold_spin.setDecimals(6)
        self.export_threshold_spin.setSingleStep(0.01)
        export_threshold_row = QHBoxLayout()
        export_threshold_row.addWidget(self.export_threshold_check)
        export_threshold_row.addWidget(self.export_threshold_spin)
        export_threshold_row.addStretch(1)
        self.run_label = QLabel("No training run loaded")
        self.input_label = QLabel("No image or folder selected")
        self.input_label.setWordWrap(True)
        self.status_label = QLabel("Ready")
        summary_form.addRow("Model", self.model_label)
        summary_form.addRow("Training Run", self.run_label)
        summary_form.addRow("Input", self.input_label)
        summary_form.addRow("Anomaly Score", self.score_label)
        summary_form.addRow("Predicted Result", self.prediction_label)
        summary_form.addRow("Training Threshold", self.threshold_label)
        summary_form.addRow("NG Export Threshold", export_threshold_row)
        summary_form.addRow("Status", self.status_label)
        root.addWidget(summary_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        root.addWidget(self.progress_bar)

        log_group = QGroupBox("Inference Log")
        log_layout = QVBoxLayout(log_group)
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMinimumHeight(140)
        log_layout.addWidget(self.log_output)
        root.addWidget(log_group)

        self.results_table = QTableWidget(0, 5)
        self.results_table.setHorizontalHeaderLabels(
            ["Source", "Prediction", "Score", "Training Threshold", "Heat Map"]
        )
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.results_table.setWordWrap(True)
        self.results_table.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.results_table.setColumnWidth(0, 280)
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.results_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.results_table.itemSelectionChanged.connect(self._show_selected_prediction)

        preview_group = QGroupBox("Selected Prediction")
        preview_layout = QHBoxLayout(preview_group)
        self.preview_labels: dict[str, QLabel] = {}
        for title in ("Original", "Heat Map", "Overlay"):
            column = QVBoxLayout()
            column.addWidget(QLabel(title))
            preview = QLabel("No image")
            preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
            preview.setMinimumSize(220, 180)
            preview.setObjectName("DatasetThumbnail")
            self.preview_labels[title] = preview
            column.addWidget(preview)
            preview_layout.addLayout(column)

        root.addWidget(self.results_table, stretch=1)
        root.addWidget(preview_group)
        self.predictions: list[PredictionResult] = []
        self._trained_threshold: float | None = None
        self._inference_running = False
        self.cancel_inference_button.setEnabled(False)
        self.export_ng_images_button.setEnabled(False)
        self.export_threshold_check.setEnabled(False)
        self.export_threshold_spin.setEnabled(False)
        self.export_threshold_check.toggled.connect(self._update_export_ng_controls)
        self.export_threshold_spin.valueChanged.connect(self._update_export_ng_controls)

    def set_training_run(self, run_directory: Path, model_name: str, threshold: float | None = None) -> None:
        """Display the trained model selected for inference."""
        self.run_label.setText(run_directory.name)
        self.model_label.setText(model_name)
        self._trained_threshold = threshold
        self.export_threshold_check.setChecked(False)
        if threshold is not None:
            self.export_threshold_spin.setValue(threshold)
        self._update_export_ng_controls()

    def export_threshold(self) -> float | None:
        """Return the trained threshold or an explicit post-inference export filter threshold."""
        if self.export_threshold_check.isChecked():
            return self.export_threshold_spin.value()
        return self._trained_threshold

    def ng_predictions_for_export(self) -> list[PredictionResult]:
        """Return selected results whose scores meet the active post-inference export threshold."""
        threshold = self.export_threshold()
        if threshold is None:
            return []
        selected_rows = sorted({index.row() for index in self.results_table.selectionModel().selectedRows()})
        source = (self.predictions[index] for index in selected_rows) if selected_rows else iter(self.predictions)
        return [prediction for prediction in source if prediction.anomaly_score >= threshold]

    def set_input_path(self, input_path: Path) -> None:
        """Display the selected image or folder input."""
        self.input_label.setText(str(input_path))

    def set_status(self, status: str) -> None:
        """Display the current inference state."""
        self.status_label.setText(status)

    def set_progress(self, current: int, total: int) -> None:
        """Display saved prediction progress."""
        self.progress_bar.setRange(0, max(total, 1))
        self.progress_bar.setValue(min(current, max(total, 1)))

    def clear_predictions(self) -> None:
        """Clear the previous inference rows and previews."""
        self.predictions.clear()
        self.results_table.setRowCount(0)
        self.score_label.setText("-")
        self.prediction_label.setText("-")
        self.threshold_label.setText("-")
        self._update_export_ng_controls()
        for label in self.preview_labels.values():
            label.clear()
            label.setText("No image")

    def clear_log(self) -> None:
        """Clear messages from the prior inference request."""
        self.log_output.clear()

    def append_log(self, level: str, message: str) -> None:
        """Append worker output and follow the newest message in the scrollable log."""
        scrollbar = self.log_output.verticalScrollBar()
        follow_latest = scrollbar.value() >= scrollbar.maximum()
        self.log_output.appendPlainText(f"[{level.upper()}] {message}")
        if follow_latest:
            scrollbar.setValue(scrollbar.maximum())

    def append_prediction(self, prediction: PredictionResult) -> None:
        """Add one streamed prediction to the table and preview it."""
        self.predictions.append(prediction)
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        values = (
            self._multiline_source_path(prediction.source_path),
            prediction.predicted_label,
            f"{prediction.anomaly_score:.6g}",
            f"{prediction.threshold:.6g}",
            "Available" if prediction.anomaly_map else "Not produced",
        )
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            if column == 0:
                item.setToolTip(prediction.source_path)
            self.results_table.setItem(row, column, item)
        self.results_table.resizeRowToContents(row)
        self._update_export_ng_controls()
        if row == 0:
            self._show_prediction(prediction)

    def set_running(self, running: bool) -> None:
        """Keep inference controls coherent while the worker is active."""
        self._inference_running = running
        self.load_run_button.setEnabled(not running)
        self.select_image_button.setEnabled(not running)
        self.select_folder_button.setEnabled(not running)
        self.run_inference_button.setEnabled(not running)
        self.cancel_inference_button.setEnabled(running)
        self._update_export_ng_controls()

    def _update_export_ng_controls(self) -> None:
        """Enable post-inference NG filtering only while results are ready for operator review."""
        can_export = not self._inference_running and bool(self.predictions) and self._trained_threshold is not None
        self.export_threshold_check.setEnabled(can_export)
        self.export_threshold_spin.setEnabled(can_export and self.export_threshold_check.isChecked())
        self.export_ng_images_button.setEnabled(can_export)

    def _show_selected_prediction(self) -> None:
        selected_rows = self.results_table.selectionModel().selectedRows()
        if not selected_rows:
            return
        prediction = self.predictions[selected_rows[0].row()]
        self._show_prediction(prediction)

    def _show_prediction(self, prediction: PredictionResult) -> None:
        """Show summary values and previews for one prediction without changing export selection."""
        self.score_label.setText(f"{prediction.anomaly_score:.6g}")
        self.prediction_label.setText(prediction.predicted_label)
        self.threshold_label.setText(f"{prediction.threshold:.6g}")
        self._set_preview("Original", prediction.original_image)
        self._set_preview("Heat Map", prediction.anomaly_map)
        self._set_preview("Overlay", prediction.overlay_image)

    def _set_preview(self, title: str, image_path: str) -> None:
        label = self.preview_labels[title]
        pixmap = QPixmap(image_path)
        if not image_path or pixmap.isNull():
            label.clear()
            label.setText("Not produced")
            return
        label.setPixmap(
            pixmap.scaled(
                label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    @staticmethod
    def _multiline_source_path(source_path: str) -> str:
        """Keep full source provenance visible in narrow result tables."""
        return source_path.replace("\\", "\\\n").replace("/", "/\n")

