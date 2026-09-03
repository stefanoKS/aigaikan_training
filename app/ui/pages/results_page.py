"""Results page."""

from pathlib import Path

from app.models.training_run import TrainingRun
from app.models.prediction_result import PredictionResult
from app.services.export_service import ModelExportFormat

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
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

    threshold_revision_requested = Signal(str, float, bool, float)
    decision_preview_requested = Signal(float)

    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 28)
        root.setSpacing(16)

        header = QHBoxLayout()
        self.filter_combo = QComboBox()
        for label in (
            "All",
            "Correct OK",
            "Correct NG",
            "False OK",
            "False NG",
            "Highest anomaly score",
            "Lowest anomaly score",
        ):
            self.filter_combo.addItem(label, label)
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

        revision_group = QGroupBox("Deployment Decision Revision")
        revision_form = QFormLayout(revision_group)
        self.threshold_revision_combo = QComboBox()
        self.calibrated_threshold_label = QLabel("Not available")
        self.active_deployment_threshold_label = QLabel("Not available")
        self.decision_score_semantic_label = QLabel("Not available")
        self.image_threshold_spin = QDoubleSpinBox()
        self.image_threshold_spin.setRange(-1_000_000_000.0, 1_000_000_000.0)
        self.image_threshold_spin.setDecimals(6)
        self.image_threshold_spin.setSingleStep(0.01)
        self.revision_pixel_mask_check = QCheckBox("Generate pixel mask")
        self.revision_pixel_threshold_spin = QDoubleSpinBox()
        self.revision_pixel_threshold_spin.setRange(-1_000_000_000.0, 1_000_000_000.0)
        self.revision_pixel_threshold_spin.setDecimals(6)
        self.revision_pixel_threshold_spin.setSingleStep(0.01)
        self.operator_note_edit = QLineEdit()
        self.operator_note_edit.setPlaceholderText("Optional operator note")
        pixel_row = QHBoxLayout()
        pixel_row.addWidget(self.revision_pixel_mask_check)
        pixel_row.addWidget(self.revision_pixel_threshold_spin)
        pixel_row.addStretch(1)
        self.apply_threshold_revision_button = QPushButton("Apply Threshold Revision")
        self.preview_threshold_effect_button = QPushButton("Preview Effect")
        self.threshold_preview_label = QLabel("Preview uses persisted scores and does not run the model.")
        self.threshold_preview_label.setObjectName("ModelSupport")
        self.threshold_preview_label.setWordWrap(True)
        self.apply_threshold_revision_button = QPushButton("Save and Activate Decision Revision")
        revision_form.addRow("Revision", self.threshold_revision_combo)
        revision_form.addRow("Calibrated NG Threshold", self.calibrated_threshold_label)
        revision_form.addRow("Active Deployment NG Score Threshold", self.active_deployment_threshold_label)
        revision_form.addRow("Proposed Deployment NG Score Threshold", self.image_threshold_spin)
        revision_form.addRow("Score Semantic", self.decision_score_semantic_label)
        revision_form.addRow("Operator Note", self.operator_note_edit)
        revision_form.addRow("Pixel Mask Threshold", pixel_row)
        revision_form.addRow("Preview Effect", self.preview_threshold_effect_button)
        revision_form.addRow(self.threshold_preview_label)
        revision_form.addRow("", self.apply_threshold_revision_button)
        root.addWidget(revision_group)

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
            "Decision Score Ranges",
            "Raw Score Ranges",
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
            metrics_form.addRow("Deployment NG Score Threshold" if key == "Threshold" else key, label)
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
        self.gallery_table.setIconSize(QSize(112, 72))
        for column_index in (0, 1, 2, 4, 5):
            self.gallery_table.setColumnWidth(column_index, 128)
        right_layout.addWidget(self.metrics_table, stretch=1)
        right_layout.addWidget(self.gallery_table, stretch=2)
        splitter.addWidget(right_column)
        root.addWidget(splitter, stretch=1)
        self.current_run_directory: Path | None = None
        self.current_run: TrainingRun | None = None
        self.active_threshold_revision_id = ""
        self._predictions: list[PredictionResult] = []
        self.filter_combo.currentTextChanged.connect(self._apply_prediction_filter)
        self.revision_pixel_mask_check.toggled.connect(self.revision_pixel_threshold_spin.setEnabled)
        self.preview_threshold_effect_button.clicked.connect(
            lambda: self.decision_preview_requested.emit(self.image_threshold_spin.value())
        )
        self.apply_threshold_revision_button.clicked.connect(self._request_threshold_revision)
        self._set_threshold_revision_enabled(False)

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
        self.active_threshold_revision_id = ""
        self._predictions = []
        self.export_model_button.setEnabled(False)
        self._set_threshold_revision_enabled(False)
        self.metrics_table.setRowCount(0)
        self.gallery_table.setRowCount(0)
        self.no_ng_warning_label.setVisible(False)
        for label in self.metric_labels.values():
            label.setText("Not available")

    def set_training_run(self, run: TrainingRun) -> None:
        """Populate the completed-run metadata and metrics."""
        self.current_run = run
        self.active_threshold_revision_id = ""
        self._predictions = list(run.predictions)
        run_directory = Path(run.run_dir)
        self.current_run_directory = run_directory if run_directory.is_dir() else None
        self.export_model_button.setEnabled(self.current_run_directory is not None)
        self._populate_threshold_revisions(run)
        self._set_threshold_revision_enabled(self.current_run_directory is not None)
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
            "Decision Score Ranges": self._score_range_text(run.threshold_metadata, "decision"),
            "Raw Score Ranges": self._score_range_text(run.threshold_metadata, "raw"),
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

    def _populate_threshold_revisions(self, run: TrainingRun) -> None:
        """List persisted revisions and initialize controls from the active saved operating point."""
        self.threshold_revision_combo.blockSignals(True)
        self.threshold_revision_combo.clear()
        self.threshold_revision_combo.addItem("Create new revision", "")
        active_revision = str(run.threshold_metadata.get("threshold_revision", ""))
        if self.current_run_directory is not None:
            for path in sorted((self.current_run_directory / "threshold_revisions").glob("threshold-*.json")):
                self.threshold_revision_combo.addItem(path.stem, path.stem)
                if path.stem == active_revision:
                    self.threshold_revision_combo.setCurrentIndex(self.threshold_revision_combo.count() - 1)
        self.threshold_revision_combo.blockSignals(False)
        threshold = run.threshold_metadata.get("threshold_value")
        if threshold is None and run.predictions:
            threshold = run.predictions[0].threshold
        try:
            self.image_threshold_spin.setValue(float(threshold))
        except (TypeError, ValueError):
            self.image_threshold_spin.setValue(0.5)
        operating_point = run.threshold_metadata.get("pixel_operating_point")
        if isinstance(operating_point, dict) and operating_point.get("enabled"):
            self.revision_pixel_mask_check.setChecked(True)
            self.revision_pixel_threshold_spin.setValue(float(operating_point.get("threshold", 0.5)))
        else:
            self.revision_pixel_mask_check.setChecked(False)
            self.revision_pixel_threshold_spin.setValue(0.5)
        try:
            calibrated = float(run.threshold_metadata.get("threshold_raw", threshold))
            self.calibrated_threshold_label.setText(f"{calibrated:.6g}")
        except (TypeError, ValueError):
            self.calibrated_threshold_label.setText("Not available")
        try:
            self.active_deployment_threshold_label.setText(f"{float(threshold):.6g}")
        except (TypeError, ValueError):
            self.active_deployment_threshold_label.setText("Not available")
        self.decision_score_semantic_label.setText(str(run.threshold_metadata.get("score_semantic", "Not available")))
        self.operator_note_edit.clear()
        self.threshold_preview_label.setText("Preview uses persisted scores and does not run the model.")

    def _set_threshold_revision_enabled(self, enabled: bool) -> None:
        self.threshold_revision_combo.setEnabled(enabled)
        self.image_threshold_spin.setEnabled(enabled)
        self.revision_pixel_mask_check.setEnabled(enabled)
        self.revision_pixel_threshold_spin.setEnabled(enabled and self.revision_pixel_mask_check.isChecked())
        self.apply_threshold_revision_button.setEnabled(enabled)
        self.preview_threshold_effect_button.setEnabled(enabled)
        self.operator_note_edit.setEnabled(enabled)

    def _request_threshold_revision(self) -> None:
        self.threshold_revision_requested.emit(
            str(self.threshold_revision_combo.currentData() or ""),
            self.image_threshold_spin.value(),
            self.revision_pixel_mask_check.isChecked(),
            self.revision_pixel_threshold_spin.value(),
        )

    def display_threshold_revision(
        self,
        revision_id: str,
        image_threshold: float,
        pixel_threshold: float | None,
        predictions: list[PredictionResult],
    ) -> None:
        """Show regenerated revision predictions while retaining the canonical training run."""
        self.active_threshold_revision_id = revision_id
        self._predictions = list(predictions)
        revision_index = self.threshold_revision_combo.findData(revision_id)
        if revision_index < 0:
            self.threshold_revision_combo.addItem(revision_id, revision_id)
            revision_index = self.threshold_revision_combo.count() - 1
        if revision_index >= 0:
            self.threshold_revision_combo.setCurrentIndex(revision_index)
        self.image_threshold_spin.setValue(image_threshold)
        self.revision_pixel_mask_check.setChecked(pixel_threshold is not None)
        if pixel_threshold is not None:
            self.revision_pixel_threshold_spin.setValue(pixel_threshold)
        self.metric_labels["Threshold Revision"].setText(revision_id)
        self.metric_labels["Threshold"].setText(self._format_value(image_threshold))
        self.metric_labels["Pixel Mask Threshold"].setText(
            self._format_value(pixel_threshold) if pixel_threshold is not None else "Disabled"
        )
        self.active_deployment_threshold_label.setText(self._format_value(image_threshold))
        self._apply_prediction_filter()

    def operator_note(self) -> str:
        """Return the note persisted in an operator-created deployment decision revision."""
        return self.operator_note_edit.text().strip()

    def display_decision_preview(self, preview: object) -> None:
        """Render a persisted-score threshold preview without presenting it as model inference."""
        calibrated = getattr(preview, "calibrated_threshold", None)
        active = getattr(preview, "active_threshold", None)
        proposed = getattr(preview, "proposed_threshold", None)
        semantic = str(getattr(preview, "score_semantic", "Not available"))
        self.calibrated_threshold_label.setText(self._format_value(calibrated))
        self.active_deployment_threshold_label.setText(self._format_value(active))
        self.decision_score_semantic_label.setText(semantic)
        details = [
            f"Existing {self._format_value(active)} -> proposed {self._format_value(proposed)}",
            f"OK->NG changes: {getattr(preview, 'ok_to_ng_changes', 0)}",
            f"NG->OK changes: {getattr(preview, 'ng_to_ok_changes', 0)}",
        ]
        false_reject = getattr(preview, "false_reject_rate", None)
        ng_recall = getattr(preview, "ng_recall", None)
        if false_reject is not None:
            details.append(f"False-reject rate: {float(false_reject):.2%}")
        if ng_recall is not None:
            details.append(f"NG recall: {float(ng_recall):.2%}")
        if getattr(preview, "outside_calibration_range", False):
            details.append("Warning: proposed threshold is outside observed calibration score range.")
        if "superadd" in semantic.casefold():
            details.append("Warning: SuperADD scores are distance values, not probabilities.")
        self.threshold_preview_label.setText(" | ".join(details))

    def displayed_predictions(self) -> list[PredictionResult]:
        """Return all predictions for the currently displayed canonical run or threshold revision."""
        return list(self._predictions)

    def filtered_predictions(self) -> list[PredictionResult]:
        """Return the rows currently selected by the Results filter."""
        selected_filter = str(self.filter_combo.currentData())
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
            image_values = (
                prediction.original_image or prediction.source_path,
                prediction.anomaly_map,
                prediction.overlay_image,
            )
            for column_index, path in enumerate(image_values):
                self.gallery_table.setItem(row_index, column_index, self._thumbnail_item(path))
            values = (
                prediction.continuous_anomaly_map,
                prediction.binary_mask,
                prediction.contour_overlay_image,
                self._format_pixel_threshold(prediction),
                prediction.predicted_label,
                prediction.ground_truth_label,
                f"{prediction.anomaly_score:.6g}",
                prediction.source_path,
            )
            for column_index, value in enumerate(values, start=3):
                if column_index in {4, 5}:
                    self.gallery_table.setItem(row_index, column_index, self._thumbnail_item(value))
                else:
                    self.gallery_table.setItem(row_index, column_index, QTableWidgetItem(value))
            self.gallery_table.setRowHeight(row_index, 80)

    @staticmethod
    def _thumbnail_item(path_value: str) -> QTableWidgetItem:
        """Render an existing image as a fixed-size preview while retaining a readable fallback."""
        item = QTableWidgetItem()
        path = Path(path_value)
        thumbnail = QPixmap(str(path)) if path.is_file() else QPixmap()
        if thumbnail.isNull():
            item.setText(path_value)
            return item
        item.setIcon(QIcon(thumbnail.scaled(112, 72, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
        item.setToolTip(str(path))
        item.setData(Qt.ItemDataRole.UserRole, str(path))
        return item

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
    def _score_range_text(threshold_metadata: dict[str, object], score_kind: str) -> str:
        diagnostics = threshold_metadata.get("final_test_score_ranges")
        if not isinstance(diagnostics, dict):
            return "Not recorded"
        ranges = diagnostics.get(score_kind)
        if not isinstance(ranges, dict) or not ranges:
            return "Not recorded"
        values: list[str] = []
        for semantic, summary in sorted(ranges.items()):
            if not isinstance(semantic, str) or not isinstance(summary, dict):
                return "Invalid"
            try:
                values.append(
                    f"{semantic}: {float(summary['minimum']):.6g} to {float(summary['maximum']):.6g} "
                    f"(n={int(summary['count'])})"
                )
            except (KeyError, TypeError, ValueError):
                return "Invalid"
        return "; ".join(values)

    @staticmethod
    def _format_pixel_threshold(prediction: PredictionResult) -> str:
        if prediction.pixel_threshold is None:
            return "Not produced"
        return f"{prediction.pixel_threshold:.6g} (map >= threshold)"

