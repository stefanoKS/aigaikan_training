"""Inference page."""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
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
        self.calibrated_threshold_label = QLabel("-")
        self.threshold_source_label = QLabel("-")
        self.score_semantic_label = QLabel("-")
        self.preprocessing_summary_label = QLabel("-")
        self.preprocessing_summary_label.setWordWrap(True)
        self.timing_preprocess_label = QLabel("-")
        self.timing_inference_label = QLabel("-")
        self.timing_end_to_end_label = QLabel("-")
        self.timing_batch_wall_label = QLabel("-")
        self.timing_amortized_label = QLabel("-")
        self.timing_batch_one_label = QLabel("-")
        self.export_threshold_check = QCheckBox("Use custom NG image copy filter")
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
        summary_form.addRow("Active Deployment NG Score Threshold", self.threshold_label)
        summary_form.addRow("Calibrated NG Threshold", self.calibrated_threshold_label)
        summary_form.addRow("Threshold Source / Revision", self.threshold_source_label)
        summary_form.addRow("Decision Score Semantic", self.score_semantic_label)
        summary_form.addRow("Saved Preprocessing", self.preprocessing_summary_label)
        summary_form.addRow("Preprocessing Compute Time", self.timing_preprocess_label)
        summary_form.addRow("Model Pipeline Time", self.timing_inference_label)
        summary_form.addRow("File-Source End-to-End Time", self.timing_end_to_end_label)
        summary_form.addRow("Folder Batch Wall Time", self.timing_batch_wall_label)
        summary_form.addRow("Amortized Batch Time per Image", self.timing_amortized_label)
        summary_form.addRow("True Batch-One Latency", self.timing_batch_one_label)
        summary_form.addRow("NG image copy filter", export_threshold_row)
        summary_form.addRow("Status", self.status_label)
        root.addWidget(summary_group)

        benchmark_group = QGroupBox("Industrial Inference Benchmark")
        benchmark_form = QFormLayout(benchmark_group)
        self.benchmark_run_label = QLabel("No completed training run selected")
        self.benchmark_select_run_button = QPushButton("Select Training Run")
        benchmark_run_row = QHBoxLayout()
        benchmark_run_row.addWidget(self.benchmark_run_label, stretch=1)
        benchmark_run_row.addWidget(self.benchmark_select_run_button)
        self.benchmark_input_label = QLabel("No benchmark image or folder selected")
        self.benchmark_input_label.setWordWrap(True)
        self.benchmark_select_image_button = QPushButton("Select Benchmark Image")
        self.benchmark_select_folder_button = QPushButton("Select Benchmark Folder")
        benchmark_input_row = QHBoxLayout()
        benchmark_input_row.addWidget(self.benchmark_input_label, stretch=1)
        benchmark_input_row.addWidget(self.benchmark_select_image_button)
        benchmark_input_row.addWidget(self.benchmark_select_folder_button)
        self.benchmark_device_combo = QComboBox()
        self.benchmark_device_combo.addItem("CUDA", "cuda")
        self.benchmark_device_combo.addItem("CPU", "cpu")
        self.benchmark_mode_combo = QComboBox()
        self.benchmark_mode_combo.addItem("Camera-equivalent", "camera-equivalent")
        self.benchmark_mode_combo.addItem("File end-to-end", "file-end-to-end")
        self.benchmark_warmup_spin = QSpinBox()
        self.benchmark_warmup_spin.setRange(0, 100000)
        self.benchmark_warmup_spin.setValue(20)
        self.benchmark_iterations_spin = QSpinBox()
        self.benchmark_iterations_spin.setRange(1, 1000000)
        self.benchmark_iterations_spin.setValue(200)
        self.benchmark_target_fps_spin = QDoubleSpinBox()
        self.benchmark_target_fps_spin.setRange(0.001, 100000.0)
        self.benchmark_target_fps_spin.setValue(10.0)
        self.benchmark_safety_reserve_spin = QDoubleSpinBox()
        self.benchmark_safety_reserve_spin.setRange(0.0, 99.0)
        self.benchmark_safety_reserve_spin.setValue(20.0)
        self.start_benchmark_button = QPushButton("Start Benchmark")
        self.start_benchmark_button.setObjectName("PrimaryButton")
        self.cancel_benchmark_button = QPushButton("Cancel")
        self.cancel_benchmark_button.setObjectName("AlertButton")
        self.export_benchmark_json_button = QPushButton("Export Benchmark JSON")
        self.export_benchmark_csv_button = QPushButton("Export Benchmark CSV")
        benchmark_actions = QHBoxLayout()
        benchmark_actions.addWidget(self.start_benchmark_button)
        benchmark_actions.addWidget(self.cancel_benchmark_button)
        benchmark_actions.addWidget(self.export_benchmark_json_button)
        benchmark_actions.addWidget(self.export_benchmark_csv_button)
        benchmark_actions.addStretch(1)
        self.benchmark_summary_labels: dict[str, QLabel] = {}
        for key in (
            "Backbone", "Precision", "Prepared Input Size", "Preprocessing P50 / P95", "Model Forward P50 / P95",
            "Model Pipeline P50 / P95", "End-to-End Compute P50 / P95 / P99", "Measured FPS", "Conservative P95 FPS",
            "Target Frame Budget", "Allowed Compute Budget", "PASS / FAIL", "Peak VRAM",
            "Assessment",
        ):
            label = QLabel("Not measured")
            self.benchmark_summary_labels[key] = label
            benchmark_form.addRow(key, label)
        benchmark_form.insertRow(0, "Training Run", benchmark_run_row)
        benchmark_form.insertRow(1, "Benchmark Image/Folder", benchmark_input_row)
        benchmark_form.insertRow(2, "Device", self.benchmark_device_combo)
        benchmark_form.insertRow(3, "Mode", self.benchmark_mode_combo)
        benchmark_form.insertRow(4, "Warmup Frames", self.benchmark_warmup_spin)
        benchmark_form.insertRow(5, "Measured Frames", self.benchmark_iterations_spin)
        benchmark_form.insertRow(6, "Target FPS", self.benchmark_target_fps_spin)
        benchmark_form.insertRow(7, "Safety Reserve %", self.benchmark_safety_reserve_spin)
        benchmark_form.insertRow(8, "", benchmark_actions)
        root.addWidget(benchmark_group)

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
            ["Source", "Prediction", "Score", "Active Deployment NG Score Threshold", "Heat Map"]
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
        for title in ("Original", "Overlay", "Mask"):
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
        self.cancel_benchmark_button.setEnabled(False)
        self.export_benchmark_json_button.setEnabled(False)
        self.export_benchmark_csv_button.setEnabled(False)
        self.export_threshold_check.toggled.connect(self._update_export_ng_controls)
        self.export_threshold_spin.valueChanged.connect(self._update_export_ng_controls)

    def set_training_run(
        self,
        run_directory: Path,
        model_name: str,
        threshold: float | None = None,
        *,
        calibrated_threshold: float | None = None,
        threshold_source: str = "",
        score_semantic: str = "",
        preprocessing_summary: str = "",
    ) -> None:
        """Display the trained model selected for inference."""
        self.run_label.setText(run_directory.name)
        self.model_label.setText(model_name)
        self._trained_threshold = threshold
        self.calibrated_threshold_label.setText(f"{calibrated_threshold:.6g}" if calibrated_threshold is not None else "-")
        self.threshold_source_label.setText(threshold_source or "run manifest")
        self.score_semantic_label.setText(score_semantic or "legacy unversioned")
        self.preprocessing_summary_label.setText(preprocessing_summary or "Historical legacy preprocessing")
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

    def set_benchmark_selection(self, run_directory: Path | None, input_path: Path | None) -> None:
        """Show independent benchmark run and input choices without changing folder inference selections."""
        self.benchmark_run_label.setText(run_directory.name if run_directory is not None else "No completed training run selected")
        self.benchmark_run_label.setToolTip(str(run_directory) if run_directory is not None else "")
        self.benchmark_input_label.setText(str(input_path) if input_path is not None else "No benchmark image or folder selected")

    def set_benchmark_running(self, running: bool) -> None:
        """Keep benchmark controls usable and separate from ordinary folder inference state."""
        for widget in (
            self.benchmark_select_run_button, self.benchmark_select_image_button, self.benchmark_select_folder_button,
            self.benchmark_device_combo, self.benchmark_mode_combo, self.benchmark_warmup_spin,
            self.benchmark_iterations_spin, self.benchmark_target_fps_spin, self.benchmark_safety_reserve_spin,
            self.start_benchmark_button,
        ):
            widget.setEnabled(not running)
        self.cancel_benchmark_button.setEnabled(running)

    def display_benchmark(self, payload: dict[str, object]) -> None:
        """Show the camera-equivalent batch-one result without conflating folder throughput."""
        metadata = payload.get("metadata", {})
        timing = payload.get("timing", {})
        deadline = payload.get("deadline", {})
        if not isinstance(metadata, dict) or not isinstance(timing, dict) or not isinstance(deadline, dict):
            return
        self.benchmark_summary_labels["Backbone"].setText(str(metadata.get("backbone", "Not measured")))
        self.benchmark_summary_labels["Precision"].setText(str(metadata.get("model_precision", "Not measured")))
        self.benchmark_summary_labels["Prepared Input Size"].setText(" x ".join(str(value) for value in metadata.get("prepared_canvas_size", [])))
        for label, phase in (
            ("Preprocessing P50 / P95", "preprocess_total_ms"),
            ("Model Forward P50 / P95", "model_forward_ms"),
            ("Model Pipeline P50 / P95", "model_pipeline_ms"),
        ):
            self.benchmark_summary_labels[label].setText(self._percentile_text(timing.get(phase), "P50 / P95"))
        self.benchmark_summary_labels["End-to-End Compute P50 / P95 / P99"].setText(
            self._percentile_text(timing.get("end_to_end_compute_ms"), "P50 / P95 / P99")
        )
        self.benchmark_summary_labels["Measured FPS"].setText(self._fps_text(payload.get("measured_steady_state_fps")))
        self.benchmark_summary_labels["Conservative P95 FPS"].setText(self._fps_text(payload.get("conservative_p95_fps")))
        self.benchmark_summary_labels["Target Frame Budget"].setText(self._timing_text(deadline.get("frame_period_ms")))
        self.benchmark_summary_labels["Allowed Compute Budget"].setText(self._timing_text(deadline.get("allowed_compute_budget_ms")))
        self.benchmark_summary_labels["PASS / FAIL"].setText("PASS" if deadline.get("pass") else "FAIL")
        self.benchmark_summary_labels["Assessment"].setText(str(deadline.get("reason", "Not measured")))
        self.benchmark_summary_labels["Peak VRAM"].setText(
            self._bytes_text(metadata.get("peak_cuda_memory_allocated"))
        )
        self.export_benchmark_json_button.setEnabled(True)
        self.export_benchmark_csv_button.setEnabled(True)

    @staticmethod
    def _percentile_text(value: object, label: str) -> str:
        if not isinstance(value, dict):
            return "Not measured"
        keys = ("p50_ms", "p95_ms", "p99_ms") if label.endswith("P99") else ("p50_ms", "p95_ms")
        values = [InferencePage._timing_text(value.get(key)) for key in keys]
        return " / ".join(values)

    @staticmethod
    def _fps_text(value: object) -> str:
        try:
            return f"{float(value):.3f} FPS"
        except (TypeError, ValueError):
            return "Not measured"

    @staticmethod
    def _bytes_text(value: object) -> str:
        try:
            return f"{float(value) / (1024 * 1024):.1f} MB"
        except (TypeError, ValueError):
            return "Not measured"

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
        self.timing_preprocess_label.setText("-")
        self.timing_inference_label.setText("-")
        self.timing_end_to_end_label.setText("-")
        self.timing_batch_wall_label.setText("-")
        self.timing_amortized_label.setText("-")
        self.timing_batch_one_label.setText("-")
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
        timing = prediction.timing_metadata
        self.timing_preprocess_label.setText(self._timing_text(timing.get("preprocess_total_ms", timing.get("preprocess_compute_ms"))))
        self.timing_inference_label.setText(self._timing_text(timing.get("model_pipeline_ms", timing.get("inference_total_ms"))))
        self.timing_end_to_end_label.setText(self._timing_text(timing.get("file_source_end_to_end_ms", timing.get("end_to_end_ms"))))
        self.timing_batch_wall_label.setText(self._timing_text(timing.get("batch_wall_ms")))
        self.timing_amortized_label.setText(self._timing_text(timing.get("amortized_batch_ms_per_image")))
        self.timing_batch_one_label.setText(self._timing_text(timing.get("true_batch_one_latency_ms")))
        self._set_preview("Original", prediction.original_image)
        self._set_preview("Overlay", prediction.overlay_image)
        self._set_preview("Mask", prediction.binary_mask)

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
    def _timing_text(value: object) -> str:
        try:
            return f"{float(value):.3f} ms"
        except (TypeError, ValueError):
            return "Not measured"

    @staticmethod
    def _multiline_source_path(source_path: str) -> str:
        """Keep full source provenance visible in narrow result tables."""
        return source_path.replace("\\", "\\\n").replace("/", "/\n")

