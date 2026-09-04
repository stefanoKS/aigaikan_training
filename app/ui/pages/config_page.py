"""Training configuration page."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.core.model_registry import ModelExecutionMode, ModelRegistry
from app.core.threshold_calibrator import ThresholdMethod
from app.core.dinomaly_encoder_registry import DinomalyEncoderRegistry
from app.core.superadd_backbone_registry import LEGACY_HUGE_BACKBONE_ID, SuperAddBackboneRegistry
from app.models.preprocessing_config import PaddingPolicy, ScoreAggregation


class ConfigPage(QWidget):
    """Training configuration UI."""

    ui_text_changed = Signal()

    def __init__(self, model_registry: ModelRegistry | None = None) -> None:
        super().__init__()
        self.model_registry = model_registry or ModelRegistry()
        self._model_definitions = {definition.key: definition for definition in self.model_registry.all()}
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 28)
        root.setSpacing(16)

        basic_group = QGroupBox("Training Configuration")
        basic_form = QFormLayout(basic_group)
        self.model_combo = QComboBox()
        self.model_combo.setMinimumContentsLength(28)
        self._populate_models()
        self.model_support_label = QLabel()
        self.model_support_label.setObjectName("ModelSupport")
        self.model_support_label.setWordWrap(True)
        self.device_combo = QComboBox()
        self.device_combo.addItem("Auto", "auto")
        self.device_combo.addItem("CUDA", "cuda")
        self.device_combo.addItem("CPU", "cpu")
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 512)
        self.batch_size_spin.setValue(8)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(42)
        self.split_seed_spin = QSpinBox()
        self.split_seed_spin.setRange(0, 999999)
        self.split_seed_spin.setValue(42)
        basic_form.addRow("Model", self.model_combo)
        basic_form.addRow("Compatibility", self.model_support_label)
        basic_form.addRow("Device", self.device_combo)
        basic_form.addRow("Random Seed", self.seed_spin)
        basic_form.addRow("Split Seed", self.split_seed_spin)
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 64)
        self.workers_spin.setValue(0)
        basic_form.addRow("Data-loader Workers", self.workers_spin)
        root.addWidget(basic_group)

        self.trainer_group = QGroupBox("Trainer Settings")
        trainer_form = QFormLayout(self.trainer_group)
        self.max_epochs_spin = QSpinBox()
        self.max_epochs_spin.setRange(1, 10000)
        self.max_epochs_spin.setValue(1)
        self.estimated_steps_label = QLabel("-")
        self.estimated_steps_label.setObjectName("EstimatedSteps")
        self.validation_every_n_epochs_spin = QSpinBox()
        self.validation_every_n_epochs_spin.setRange(1, 10000)
        self.validation_every_n_epochs_spin.setValue(1)
        self.gradient_clip_spin = QDoubleSpinBox()
        self.gradient_clip_spin.setRange(0.0, 1000.0)
        self.gradient_clip_spin.setDecimals(3)
        self.gradient_clip_spin.setSingleStep(0.1)
        self.gradient_clip_spin.setSpecialValueText("Disabled")
        self.gradient_clip_spin.setValue(0.0)
        self.accumulate_grad_batches_spin = QSpinBox()
        self.accumulate_grad_batches_spin.setRange(1, 1024)
        self.accumulate_grad_batches_spin.setValue(1)
        trainer_form.addRow("Batch Size", self.batch_size_spin)
        trainer_form.addRow("Max Epochs", self.max_epochs_spin)
        trainer_form.addRow("Estimated Training Steps", self.estimated_steps_label)
        trainer_form.addRow("Validate Every (Epochs)", self.validation_every_n_epochs_spin)
        trainer_form.addRow("Gradient Clip Norm", self.gradient_clip_spin)
        trainer_form.addRow("Accumulate Batches", self.accumulate_grad_batches_spin)
        root.addWidget(self.trainer_group)

        self.dinomaly_group = QGroupBox("Dinomaly Training")
        dinomaly_form = QFormLayout(self.dinomaly_group)
        self.dinomaly_encoder_registry = DinomalyEncoderRegistry()
        self.dinomaly_encoder_combo = QComboBox()
        self.dinomaly_encoder_combo.setMinimumContentsLength(32)
        self.dinomaly_encoder_support_label = QLabel()
        self.dinomaly_encoder_support_label.setObjectName("ModelSupport")
        self.dinomaly_encoder_support_label.setWordWrap(True)
        self.target_training_steps_spin = QSpinBox()
        self.target_training_steps_spin.setRange(0, 1000000)
        self.target_training_steps_spin.setSingleStep(500)
        self.target_training_steps_spin.setSpecialValueText("Automatic baseline")
        self.target_training_steps_spin.setValue(0)
        dinomaly_form.addRow("Encoder", self.dinomaly_encoder_combo)
        dinomaly_form.addRow("Availability", self.dinomaly_encoder_support_label)
        dinomaly_form.addRow("Training Steps Override", self.target_training_steps_spin)
        root.addWidget(self.dinomaly_group)

        self.superadd_group = QGroupBox("SuperADD Settings")
        superadd_form = QFormLayout(self.superadd_group)
        self.superadd_backbone_registry = SuperAddBackboneRegistry()
        self.superadd_backbone_combo = QComboBox()
        self.superadd_backbone_combo.setMinimumContentsLength(32)
        self.superadd_precision_combo = QComboBox()
        self.superadd_precision_combo.addItem("FP32", "float32")
        self.superadd_precision_combo.addItem("FP16 - CUDA only", "float16")
        self.superadd_feature_layers_label = QLabel("Automatic")
        self.superadd_guidance_label = QLabel()
        self.superadd_guidance_label.setObjectName("ModelSupport")
        self.superadd_guidance_label.setWordWrap(True)
        self.superadd_native_score_label = QLabel(
            "SuperADD uses its native top-0.1% anomaly-map mean; this setting is not modified."
        )
        self.superadd_native_score_label.setObjectName("ModelSupport")
        self.superadd_native_score_label.setWordWrap(True)
        superadd_form.addRow("Backbone", self.superadd_backbone_combo)
        superadd_form.addRow("Precision", self.superadd_precision_combo)
        superadd_form.addRow("Feature Layers", self.superadd_feature_layers_label)
        superadd_form.addRow(self.superadd_guidance_label)
        superadd_form.addRow(self.superadd_native_score_label)
        root.addWidget(self.superadd_group)

        self.preprocessing_group = QGroupBox("Preprocessing Policy")
        preprocessing_form = QFormLayout(self.preprocessing_group)
        self.rectified_roi_size_label = QLabel("Select an inspection ROI or dataset image")
        self.padding_policy_combo = QComboBox()
        self.padding_policy_combo.addItem("Automatic", PaddingPolicy.AUTOMATIC.value)
        self.padding_policy_combo.addItem("Custom", PaddingPolicy.CUSTOM.value)
        self.padding_policy_combo.setToolTip("ROI or padding changes require retraining.")
        self.automatic_right_padding_label = QLabel("-")
        self.automatic_bottom_padding_label = QLabel("-")
        self.custom_right_padding_spin = QSpinBox()
        self.custom_right_padding_spin.setRange(0, 100000)
        self.custom_right_padding_spin.setSuffix(" px")
        self.custom_bottom_padding_spin = QSpinBox()
        self.custom_bottom_padding_spin.setRange(0, 100000)
        self.custom_bottom_padding_spin.setSuffix(" px")
        self.prepared_image_size_label = QLabel("-")
        self.model_alignment_label = QLabel("-")
        self.padding_validation_label = QLabel()
        self.padding_validation_label.setObjectName("ModelSupport")
        self.padding_validation_label.setWordWrap(True)
        self.use_nearest_valid_size_button = QPushButton("Use Nearest Valid Size")
        self.use_nearest_valid_size_button.setEnabled(False)
        self.reset_padding_button = QPushButton("Reset to Automatic")
        self.tiling_check = QCheckBox("Enable horizontal tile processing")
        self.score_aggregation_combo = QComboBox()
        self.score_aggregation_combo.addItem("Maximum valid-pixel score", ScoreAggregation.MAX.value)
        self.score_aggregation_combo.addItem("Top-k valid-pixel mean", ScoreAggregation.TOP_K_MEAN.value)
        self.top_k_fraction_spin = QDoubleSpinBox()
        self.top_k_fraction_spin.setRange(0.01, 100.0)
        self.top_k_fraction_spin.setDecimals(2)
        self.top_k_fraction_spin.setSingleStep(0.1)
        self.top_k_fraction_spin.setSuffix("%")
        self.top_k_fraction_spin.setValue(1.0)
        self.superadd_score_aggregation_note = QLabel(
            "SuperADD uses its native top-0.1% anomaly-map mean; this setting is not modified."
        )
        self.superadd_score_aggregation_note.setObjectName("ModelSupport")
        self.superadd_score_aggregation_note.setWordWrap(True)
        preprocessing_form.addRow("Rectified ROI Size", self.rectified_roi_size_label)
        preprocessing_form.addRow("Padding Policy", self.padding_policy_combo)
        preprocessing_form.addRow("Automatic Right Padding", self.automatic_right_padding_label)
        preprocessing_form.addRow("Automatic Bottom Padding", self.automatic_bottom_padding_label)
        preprocessing_form.addRow("Custom Right Padding", self.custom_right_padding_spin)
        preprocessing_form.addRow("Custom Bottom Padding", self.custom_bottom_padding_spin)
        preprocessing_form.addRow("Prepared Image Size", self.prepared_image_size_label)
        preprocessing_form.addRow("Model Alignment Requirement", self.model_alignment_label)
        preprocessing_form.addRow(self.padding_validation_label)
        preprocessing_form.addRow(self.use_nearest_valid_size_button, self.reset_padding_button)
        preprocessing_form.addRow("Tiling", self.tiling_check)
        preprocessing_form.addRow("Score Aggregation", self.score_aggregation_combo)
        preprocessing_form.addRow("Top-k Fraction", self.top_k_fraction_spin)
        preprocessing_form.addRow(self.superadd_score_aggregation_note)
        root.addWidget(self.preprocessing_group)

        self.threshold_group = QGroupBox("Decision Threshold Calibration")
        threshold_form = QFormLayout(self.threshold_group)
        self.threshold_method_combo = QComboBox()
        for label, method in (
            ("Automatic from held-out calibration data", ThresholdMethod.AUTO),
            ("Labeled F1", ThresholdMethod.LABELED_F1),
            ("Labeled recall priority", ThresholdMethod.LABELED_RECALL_PRIORITY),
            ("Normal-only conformal", ThresholdMethod.NORMAL_ONLY_CONFORMAL),
            ("Normal-only maximum (legacy)", ThresholdMethod.NORMAL_ONLY_MAX),
        ):
            self.threshold_method_combo.addItem(label, method.value)
        self.threshold_fpr_combo = QComboBox()
        for label, rate in (("0.1%", 0.001), ("0.5%", 0.005), ("1.0%", 0.01), ("Custom", None)):
            self.threshold_fpr_combo.addItem(label, rate)
        self.threshold_fpr_spin = QDoubleSpinBox()
        self.threshold_fpr_spin.setRange(0.001, 99.999)
        self.threshold_fpr_spin.setDecimals(3)
        self.threshold_fpr_spin.setSuffix("%")
        self.threshold_fpr_spin.setValue(0.5)
        self.minimum_ng_recall_check = QCheckBox("Require minimum NG recall")
        self.minimum_ng_recall_spin = QDoubleSpinBox()
        self.minimum_ng_recall_spin.setRange(0.0, 100.0)
        self.minimum_ng_recall_spin.setDecimals(1)
        self.minimum_ng_recall_spin.setSuffix("%")
        self.minimum_ng_recall_spin.setValue(95.0)
        self.normal_only_calibration_note = QLabel(
            "Normal-only calibration selects a false-reject operating point. Defect-detection performance remains unverified without genuine NG data."
        )
        self.normal_only_calibration_note.setWordWrap(True)
        self.pixel_threshold_check = QCheckBox("Enable pixel mask threshold")
        self.pixel_threshold_check.setToolTip(
            "Creates mask and contour artifacts from continuous anomaly-map values without changing image decisions."
        )
        self.pixel_threshold_spin = QDoubleSpinBox()
        self.pixel_threshold_spin.setRange(-1_000_000_000.0, 1_000_000_000.0)
        self.pixel_threshold_spin.setDecimals(6)
        self.pixel_threshold_spin.setSingleStep(0.01)
        self.pixel_threshold_spin.setValue(0.5)
        threshold_form.addRow("Calibration Method", self.threshold_method_combo)
        threshold_form.addRow("Normal False Reject Target", self.threshold_fpr_combo)
        threshold_form.addRow("Custom Normal False Reject Target", self.threshold_fpr_spin)
        threshold_form.addRow("NG Recall Target", self.minimum_ng_recall_check)
        threshold_form.addRow("Required NG Recall", self.minimum_ng_recall_spin)
        threshold_form.addRow(self.normal_only_calibration_note)
        threshold_form.addRow("Pixel Mask", self.pixel_threshold_check)
        threshold_form.addRow("Pixel Map Threshold", self.pixel_threshold_spin)
        root.addWidget(self.threshold_group)

        self.acceptance_group = QGroupBox("Final-Test Acceptance Policy")
        acceptance_form = QFormLayout(self.acceptance_group)
        self.maximum_final_test_false_reject_spin = QDoubleSpinBox()
        self.maximum_final_test_false_reject_spin.setRange(0.0, 100.0)
        self.maximum_final_test_false_reject_spin.setDecimals(3)
        self.maximum_final_test_false_reject_spin.setSuffix("%")
        self.maximum_final_test_false_reject_spin.setValue(0.5)
        self.minimum_final_test_ok_images_spin = QSpinBox()
        self.minimum_final_test_ok_images_spin.setRange(1, 1000000)
        self.minimum_final_test_ok_images_spin.setValue(10)
        self.minimum_final_test_ng_images_spin = QSpinBox()
        self.minimum_final_test_ng_images_spin.setRange(1, 1000000)
        self.minimum_final_test_ng_images_spin.setValue(10)
        acceptance_form.addRow("Maximum False Reject Rate", self.maximum_final_test_false_reject_spin)
        acceptance_form.addRow("Minimum OK Test Images", self.minimum_final_test_ok_images_spin)
        acceptance_form.addRow("Minimum NG Test Images", self.minimum_final_test_ng_images_spin)
        root.addWidget(self.acceptance_group)

        button_row = QHBoxLayout()
        self.save_button = QPushButton("Save Configuration")
        self.save_button.setObjectName("PrimaryButton")
        self.load_button = QPushButton("Load Configuration")
        self.reset_button = QPushButton("Reset to Defaults")
        button_row.addWidget(self.save_button)
        button_row.addWidget(self.load_button)
        button_row.addWidget(self.reset_button)
        button_row.addStretch(1)
        root.addLayout(button_row)
        root.addStretch(1)
        self.model_combo.currentIndexChanged.connect(self._update_model_support)
        self.superadd_backbone_combo.currentIndexChanged.connect(self._update_superadd_guidance)
        self.threshold_fpr_combo.currentIndexChanged.connect(self._update_threshold_controls)
        self.minimum_ng_recall_check.toggled.connect(self._update_threshold_controls)
        self.pixel_threshold_check.toggled.connect(self._update_threshold_controls)
        self.score_aggregation_combo.currentIndexChanged.connect(self._update_preprocessing_controls)
        self.padding_policy_combo.currentIndexChanged.connect(self._update_preprocessing_controls)
        self.use_nearest_valid_size_button.clicked.connect(self._use_nearest_valid_size)
        self.reset_padding_button.clicked.connect(self._reset_padding_to_automatic)
        self._update_model_support()
        self._update_threshold_controls()
        self._update_preprocessing_controls()

    def set_estimated_training_steps(self, steps: int, epochs: int) -> None:
        """Show the model-adjusted optimizer work without changing layout width."""
        self.estimated_steps_label.setText(f"{steps:,} steps ({epochs:,} epochs)")

    def set_padding_policy(self, policy: PaddingPolicy, right_padding: int, bottom_padding: int, editable: bool) -> None:
        """Load persisted padding settings without upgrading legacy projects implicitly."""
        policy_index = self.padding_policy_combo.findData(policy.value)
        self.padding_policy_combo.blockSignals(True)
        self.padding_policy_combo.setCurrentIndex(max(policy_index, 0))
        self.padding_policy_combo.blockSignals(False)
        self.custom_right_padding_spin.setValue(right_padding)
        self.custom_bottom_padding_spin.setValue(bottom_padding)
        self.padding_policy_combo.setEnabled(editable)
        self.reset_padding_button.setEnabled(editable)
        self.padding_policy_combo.setToolTip(
            "ROI or padding changes require retraining."
            if editable
            else "Legacy preprocessing-v2 is retained unchanged for compatibility with existing runs."
        )
        self._update_preprocessing_controls()
        self.ui_text_changed.emit()

    def set_preprocessing_geometry(
        self,
        *,
        rectified_size: tuple[int, int] | None,
        automatic_padding: tuple[int, int] | None,
        prepared_size: tuple[int, int] | None,
        alignment: tuple[int, int] | None,
        validation_message: str = "",
        allow_nearest_size: bool = True,
    ) -> None:
        """Display model-ready geometry calculated from the current ROI and controls."""
        self.rectified_roi_size_label.setText(
            f"{rectified_size[0]} x {rectified_size[1]} px" if rectified_size is not None else "Select an inspection ROI or dataset image"
        )
        self.automatic_right_padding_label.setText(
            f"{automatic_padding[0]} px" if automatic_padding is not None else "-"
        )
        self.automatic_bottom_padding_label.setText(
            f"{automatic_padding[1]} px" if automatic_padding is not None else "-"
        )
        self.prepared_image_size_label.setText(
            f"{prepared_size[0]} x {prepared_size[1]} px" if prepared_size is not None else "-"
        )
        self.model_alignment_label.setText(
            f"{alignment[0]} x {alignment[1]} px" if alignment is not None else "-"
        )
        self.padding_validation_label.setText(validation_message)
        self.use_nearest_valid_size_button.setEnabled(automatic_padding is not None and allow_nearest_size)
        self.ui_text_changed.emit()

    def padding_policy(self) -> PaddingPolicy:
        """Return the operator-selected v3 padding policy."""
        return PaddingPolicy(str(self.padding_policy_combo.currentData()))

    def _use_nearest_valid_size(self) -> None:
        """Copy the shown automatic right/bottom values into Custom mode explicitly."""
        right_padding = self._padding_label_value(self.automatic_right_padding_label)
        bottom_padding = self._padding_label_value(self.automatic_bottom_padding_label)
        if right_padding is None or bottom_padding is None:
            return
        self.custom_right_padding_spin.setValue(right_padding)
        self.custom_bottom_padding_spin.setValue(bottom_padding)
        self.padding_policy_combo.setCurrentIndex(self.padding_policy_combo.findData(PaddingPolicy.CUSTOM.value))

    def _reset_padding_to_automatic(self) -> None:
        """Return to dynamic model-aligned right/bottom padding."""
        self.padding_policy_combo.setCurrentIndex(self.padding_policy_combo.findData(PaddingPolicy.AUTOMATIC.value))

    @staticmethod
    def _padding_label_value(label: QLabel) -> int | None:
        value = label.text().removesuffix(" px")
        return int(value) if value.isdigit() else None

    def _populate_models(self) -> None:
        """Populate the fixed set of supported model configurations."""
        current_key = str(self.model_combo.currentData())
        definitions = self.model_registry.image_folder_models()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for definition in definitions:
            self.model_combo.addItem(definition.display_name, definition.key)
            item = getattr(self.model_combo.model(), "item", lambda _index: None)(self.model_combo.count() - 1)
            if item is not None:
                item.setToolTip(definition.requirement or definition.support_level.replace("-", " ").title())
        index = self.model_combo.findData(current_key)
        self.model_combo.setCurrentIndex(max(index, 0))
        self.model_combo.blockSignals(False)
        if hasattr(self, "model_support_label"):
            self._update_model_support()

    def _update_model_support(self) -> None:
        """Describe the currently selected model's project compatibility."""
        model_key = str(self.model_combo.currentData())
        definition = self._model_definitions.get(model_key)
        if definition is None:
            self.model_support_label.setText("Model details are unavailable.")
            return
        data_contract = "Image-folder project" if definition.supports_image_folder else "Video project required"
        execution = (
            "Zero-shot evaluation"
            if definition.execution_mode is ModelExecutionMode.EVALUATE
            else "Train and evaluate"
        )
        details = [data_contract, execution, definition.support_level.replace("-", " ").title()]
        if definition.requirement:
            details.append(definition.requirement)
        self.model_support_label.setText(" | ".join(details))
        self._update_model_controls(definition.key)
        self.ui_text_changed.emit()

    def _update_model_controls(self, model_key: str) -> None:
        """Show only inputs that apply to the selected Anomalib model."""
        is_dinomaly = model_key in {"dinomaly_dinov2", "dinomaly_dinov3"}
        self.dinomaly_group.setVisible(is_dinomaly)
        if is_dinomaly:
            self._populate_dinomaly_encoders(model_key)
        is_superadd = model_key == "super_add"
        self.superadd_group.setVisible(is_superadd)
        if is_superadd:
            self._populate_superadd_backbones()
        is_training_model = self._model_definitions[model_key].execution_mode is ModelExecutionMode.TRAIN
        self.trainer_group.setEnabled(is_training_model)
        self.trainer_group.setTitle("Trainer Settings" if is_training_model else "Trainer Settings (Not used for zero-shot evaluation)")
        uses_fixed_one_pass = model_key in {"patchcore", "padim", "anomaly_dino", "super_add"}
        self.max_epochs_spin.setValue(1 if uses_fixed_one_pass else self.max_epochs_spin.value())
        self.max_epochs_spin.setEnabled(not uses_fixed_one_pass and not is_dinomaly and is_training_model)
        self.max_epochs_spin.setToolTip(
            "Dinomaly uses its Anomalib trainer defaults and the configured step budget."
            if is_dinomaly
            else (
                "This model uses one memory-bank collection pass."
                if uses_fixed_one_pass
                else ""
            )
        )
        fixed_batch_size = {"patchcore": 8, "efficient_ad": 1}.get(model_key)
        if fixed_batch_size is not None:
            self.batch_size_spin.setValue(fixed_batch_size)
        self.batch_size_spin.setEnabled(is_training_model and fixed_batch_size is None)
        self.batch_size_spin.setToolTip(
            "EfficientAD requires one-image training batches."
            if model_key == "efficient_ad"
            else ("PatchCore uses batches of eight images." if model_key == "patchcore" else "")
        )
        supports_tiling = is_dinomaly
        self.tiling_check.setEnabled(supports_tiling)
        self.tiling_check.setToolTip("Tiling is currently supported only by Dinomaly models." if not supports_tiling else "")
        if not supports_tiling:
            self.tiling_check.setChecked(False)
        self.score_aggregation_combo.setEnabled(not is_superadd)
        self.top_k_fraction_spin.setEnabled(not is_superadd and self.score_aggregation_combo.currentData() == ScoreAggregation.TOP_K_MEAN.value)
        self.superadd_score_aggregation_note.setVisible(is_superadd)

    def set_superadd_settings(self, backbone_id: str, precision: str) -> None:
        """Load a persisted SuperADD contract without accepting arbitrary text values."""
        if str(self.model_combo.currentData()) != "super_add":
            return
        self._populate_superadd_backbones(backbone_id)
        precision_index = self.superadd_precision_combo.findData(precision)
        self.superadd_precision_combo.setCurrentIndex(max(precision_index, 0))

    def _populate_superadd_backbones(self, requested_identifier: str = "") -> None:
        """Populate fixed DINOv3 backbones and disable runtime-unavailable choices."""
        current_identifier = requested_identifier or str(self.superadd_backbone_combo.currentData() or "")
        presets = self.superadd_backbone_registry.all()
        if current_identifier == LEGACY_HUGE_BACKBONE_ID:
            current_identifier = presets[-1].identifier
        self.superadd_backbone_combo.blockSignals(True)
        self.superadd_backbone_combo.clear()
        first_available_index = -1
        requested_available_index = -1
        unavailable_labels: list[str] = []
        for preset in presets:
            available = self.superadd_backbone_registry.is_available(preset)
            self.superadd_backbone_combo.addItem(preset.display_name, preset.identifier)
            index = self.superadd_backbone_combo.count() - 1
            item = getattr(self.superadd_backbone_combo.model(), "item", lambda _index: None)(index)
            if item is not None:
                item.setEnabled(available)
                item.setToolTip(preset.identifier if available else f"Unavailable: {preset.identifier}")
            if available and first_available_index < 0:
                first_available_index = index
            if available and preset.identifier == current_identifier:
                requested_available_index = index
            if not available:
                unavailable_labels.append(preset.display_name)
        self.superadd_backbone_combo.setCurrentIndex(
            requested_available_index if requested_available_index >= 0 else first_available_index
        )
        self.superadd_backbone_combo.blockSignals(False)
        self._update_superadd_guidance()
        if first_available_index < 0:
            self.superadd_guidance_label.setText("No curated SuperADD backbone is available in the installed timm runtime.")
        elif unavailable_labels:
            self.superadd_guidance_label.setText("Unavailable: " + ", ".join(unavailable_labels))
        self.ui_text_changed.emit()

    def _update_superadd_guidance(self) -> None:
        """Show short latency guidance for the currently selected curated backbone."""
        identifier = str(self.superadd_backbone_combo.currentData() or "")
        try:
            preset = self.superadd_backbone_registry.get(identifier)
        except ValueError:
            self.superadd_guidance_label.setText("Select an available curated SuperADD backbone.")
            self.ui_text_changed.emit()
            return
        self.superadd_guidance_label.setText(f"{preset.display_name}: {preset.guidance}.")
        self.ui_text_changed.emit()

    def set_dinomaly_encoder(self, identifier: str) -> None:
        """Select a persisted curated encoder after the current model family is loaded."""
        model_key = str(self.model_combo.currentData())
        if model_key not in {"dinomaly_dinov2", "dinomaly_dinov3"}:
            return
        self._populate_dinomaly_encoders(model_key, identifier)

    def _populate_dinomaly_encoders(self, model_key: str, requested_identifier: str = "") -> None:
        """Populate only curated encoders for the selected DINO generation."""
        family = "DINOv3" if model_key == "dinomaly_dinov3" else "DINOv2"
        current_identifier = requested_identifier or str(self.dinomaly_encoder_combo.currentData() or "")
        presets = self.dinomaly_encoder_registry.all(family)
        if current_identifier not in {preset.identifier for preset in presets}:
            current_identifier = (
                "vit_base_patch16_dinov3.lvd1689m"
                if family == "DINOv3"
                else "vit_base_patch14_reg4_dinov2"
            )
        self.dinomaly_encoder_combo.blockSignals(True)
        self.dinomaly_encoder_combo.clear()
        first_available_index = -1
        requested_available_index = -1
        unavailable_labels: list[str] = []
        for preset in presets:
            available = self.dinomaly_encoder_registry.is_available(preset)
            self.dinomaly_encoder_combo.addItem(preset.display_name, preset.identifier)
            index = self.dinomaly_encoder_combo.count() - 1
            item = getattr(self.dinomaly_encoder_combo.model(), "item", lambda _index: None)(index)
            if item is not None:
                item.setEnabled(available)
                item.setToolTip(preset.identifier if available else f"Unavailable: {preset.identifier}")
            if available and first_available_index < 0:
                first_available_index = index
            if available and preset.identifier == current_identifier:
                requested_available_index = index
            if not available:
                unavailable_labels.append(preset.display_name)
        self.dinomaly_encoder_combo.setCurrentIndex(
            requested_available_index if requested_available_index >= 0 else first_available_index
        )
        self.dinomaly_encoder_combo.blockSignals(False)
        self.dinomaly_encoder_support_label.setText(
            "No curated encoder is available in the installed timm runtime."
            if first_available_index < 0
            else ("Unavailable: " + ", ".join(unavailable_labels) if unavailable_labels else "All curated encoders are available.")
        )
        self.ui_text_changed.emit()

    def threshold_false_reject_rate(self) -> float:
        """Return the selected normal false-reject rate as a probability."""
        preset = self.threshold_fpr_combo.currentData()
        return float(preset) if preset is not None else self.threshold_fpr_spin.value() / 100

    def _update_threshold_controls(self) -> None:
        """Enable only the optional calibration inputs selected by the user."""
        self.threshold_fpr_spin.setEnabled(self.threshold_fpr_combo.currentData() is None)
        self.minimum_ng_recall_spin.setEnabled(self.minimum_ng_recall_check.isChecked())
        self.pixel_threshold_spin.setEnabled(self.pixel_threshold_check.isChecked())

    def _update_preprocessing_controls(self) -> None:
        """Expose top-k tuning only when its aggregation strategy is active."""
        self.top_k_fraction_spin.setEnabled(
            self.score_aggregation_combo.isEnabled()
            and self.score_aggregation_combo.currentData() == ScoreAggregation.TOP_K_MEAN.value
        )
        custom_padding = self.padding_policy() is PaddingPolicy.CUSTOM and self.padding_policy_combo.isEnabled()
        self.custom_right_padding_spin.setEnabled(custom_padding)
        self.custom_bottom_padding_spin.setEnabled(custom_padding)

