"""Training configuration page."""

from __future__ import annotations

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

from app.core.model_registry import ModelExecutionMode, ModelRegistry, ModelSupportLevel
from app.core.threshold_calibrator import ThresholdMethod


class ConfigPage(QWidget):
    """Training configuration UI."""

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
        self.show_experimental_models_check = QCheckBox("Show experimental models")
        self._populate_models()
        self.model_support_label = QLabel()
        self.model_support_label.setObjectName("ModelSupport")
        self.model_support_label.setWordWrap(True)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["Auto", "CUDA", "CPU"])
        self.image_width_spin = QSpinBox()
        self.image_width_spin.setRange(32, 8192)
        self.image_width_spin.setValue(280)
        self.image_height_spin = QSpinBox()
        self.image_height_spin.setRange(32, 8192)
        self.image_height_spin.setValue(280)
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
        basic_form.addRow("Advanced", self.show_experimental_models_check)
        basic_form.addRow("Compatibility", self.model_support_label)
        basic_form.addRow("Device", self.device_combo)
        basic_form.addRow("AI Input Width", self.image_width_spin)
        basic_form.addRow("AI Input Height", self.image_height_spin)
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

        self.patchcore_group = QGroupBox("PatchCore Settings")
        patchcore_form = QFormLayout(self.patchcore_group)
        self.coreset_ratio_spin = QDoubleSpinBox()
        self.coreset_ratio_spin.setRange(0.01, 1.0)
        self.coreset_ratio_spin.setSingleStep(0.01)
        self.coreset_ratio_spin.setValue(0.1)
        self.neighbors_spin = QSpinBox()
        self.neighbors_spin.setRange(1, 99)
        self.neighbors_spin.setValue(9)
        patchcore_form.addRow("Coreset Sampling Ratio", self.coreset_ratio_spin)
        patchcore_form.addRow("Nearest Neighbors", self.neighbors_spin)
        root.addWidget(self.patchcore_group)

        self.resources_group = QGroupBox("Model Resources")
        self.resources_form = QFormLayout(self.resources_group)
        resource_row = QHBoxLayout()
        self.supplemental_path_edit = QLineEdit()
        self.browse_supplemental_button = QPushButton("Browse")
        resource_row.addWidget(self.supplemental_path_edit, stretch=1)
        resource_row.addWidget(self.browse_supplemental_button)
        self.zero_shot_class_name_edit = QLineEdit()
        self.supplemental_data_label = QLabel("Supplemental Model Data")
        self.zero_shot_class_name_label = QLabel("Zero-shot Class Name")
        self.resources_form.addRow(self.supplemental_data_label, resource_row)
        self.resources_form.addRow(self.zero_shot_class_name_label, self.zero_shot_class_name_edit)
        root.addWidget(self.resources_group)

        self.dinomaly_group = QGroupBox("Dinomaly DINOv2 Settings")
        dinomaly_form = QFormLayout(self.dinomaly_group)
        self.dinomaly_encoder_combo = QComboBox()
        self.dinomaly_encoder_combo.addItems(
            [
                "vit_small_patch14_reg4_dinov2",
                "vit_base_patch14_reg4_dinov2",
                "vit_large_patch14_reg4_dinov2",
            ]
        )
        self.dinomaly_decoder_depth_spin = QSpinBox()
        self.dinomaly_decoder_depth_spin.setRange(2, 32)
        self.dinomaly_decoder_depth_spin.setValue(8)
        self.dinomaly_dropout_spin = QDoubleSpinBox()
        self.dinomaly_dropout_spin.setRange(0.0, 0.99)
        self.dinomaly_dropout_spin.setSingleStep(0.05)
        self.dinomaly_dropout_spin.setValue(0.2)
        self.target_training_steps_spin = QSpinBox()
        self.target_training_steps_spin.setRange(1000, 1000000)
        self.target_training_steps_spin.setSingleStep(500)
        self.target_training_steps_spin.setValue(3000)
        self.dinomaly_context_recentering_check = QCheckBox()
        dinomaly_form.addRow("DINOv2 Encoder", self.dinomaly_encoder_combo)
        dinomaly_form.addRow("Decoder Depth", self.dinomaly_decoder_depth_spin)
        dinomaly_form.addRow("Bottleneck Dropout", self.dinomaly_dropout_spin)
        dinomaly_form.addRow("Target Training Steps", self.target_training_steps_spin)
        dinomaly_form.addRow("Context Recentering", self.dinomaly_context_recentering_check)
        root.addWidget(self.dinomaly_group)

        self.dinov3_group = QGroupBox("DINOv3 Experimental Settings")
        dinov3_form = QFormLayout(self.dinov3_group)
        self.dinov3_encoder_label = QLabel("Dinomaly DINOv3 Encoder")
        self.dinov3_encoder_combo = QComboBox()
        self.dinov3_encoder_combo.addItems(
            [
                "vit_small_patch16_dinov3.lvd1689m",
                "vit_base_patch16_dinov3.lvd1689m",
                "vit_large_patch16_dinov3.lvd1689m",
            ]
        )
        self.dinov3_feature_layers_label = QLabel("Feature Layers")
        self.dinov3_feature_layers_edit = QLineEdit()
        self.dinov3_feature_layers_edit.setPlaceholderText("Automatic runtime selection")
        self.superadd_encoder_label = QLabel("SuperADD DINOv3 Encoder")
        self.superadd_encoder_combo = QComboBox()
        self.superadd_encoder_combo.addItems(
            [
                "vit_huge_plus_patch16_dinov3.lvd1689m",
                "vit_large_patch16_dinov3.lvd1689m",
                "vit_base_patch16_dinov3.lvd1689m",
            ]
        )
        self.superadd_patch_size_label = QLabel("SuperADD Patch Size")
        self.superadd_patch_size_spin = QSpinBox()
        self.superadd_patch_size_spin.setRange(32, 4096)
        self.superadd_patch_size_spin.setSingleStep(16)
        self.superadd_patch_size_spin.setValue(448)
        self.superadd_patch_overlap_label = QLabel("SuperADD Patch Overlap")
        self.superadd_patch_overlap_spin = QSpinBox()
        self.superadd_patch_overlap_spin.setRange(1, 2048)
        self.superadd_patch_overlap_spin.setSingleStep(16)
        self.superadd_patch_overlap_spin.setValue(16)
        self.dinov3_experimental_label = QLabel(
            "Dinomaly DINOv3 is an application adapter. Runtime encoder metadata and token layout are verified before training."
        )
        self.dinov3_experimental_label.setWordWrap(True)
        dinov3_form.addRow(self.dinov3_encoder_label, self.dinov3_encoder_combo)
        dinov3_form.addRow(self.dinov3_feature_layers_label, self.dinov3_feature_layers_edit)
        dinov3_form.addRow(self.superadd_encoder_label, self.superadd_encoder_combo)
        dinov3_form.addRow(self.superadd_patch_size_label, self.superadd_patch_size_spin)
        dinov3_form.addRow(self.superadd_patch_overlap_label, self.superadd_patch_overlap_spin)
        dinov3_form.addRow(self.dinov3_experimental_label)
        root.addWidget(self.dinov3_group)

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
        threshold_form.addRow("Calibration Method", self.threshold_method_combo)
        threshold_form.addRow("Normal False Reject Target", self.threshold_fpr_combo)
        threshold_form.addRow("Custom Normal False Reject Target", self.threshold_fpr_spin)
        threshold_form.addRow("NG Recall Target", self.minimum_ng_recall_check)
        threshold_form.addRow("Required NG Recall", self.minimum_ng_recall_spin)
        threshold_form.addRow(self.normal_only_calibration_note)
        root.addWidget(self.threshold_group)

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
        self.show_experimental_models_check.toggled.connect(self._populate_models)
        self.threshold_fpr_combo.currentIndexChanged.connect(self._update_threshold_controls)
        self.minimum_ng_recall_check.toggled.connect(self._update_threshold_controls)
        self._update_model_support()
        self._update_threshold_controls()

    def set_estimated_training_steps(self, steps: int, epochs: int) -> None:
        """Show the model-adjusted optimizer work without changing layout width."""
        self.estimated_steps_label.setText(f"{steps:,} steps ({epochs:,} epochs)")

    def _populate_models(self) -> None:
        """Keep production-validated models prominent while retaining experimental access."""
        current_key = str(self.model_combo.currentData())
        definitions = self.model_registry.image_folder_models()
        if not self.show_experimental_models_check.isChecked():
            definitions = [
                definition
                for definition in definitions
                if definition.support_level is ModelSupportLevel.PRODUCTION_VALIDATED
            ]
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

    def _update_model_controls(self, model_key: str) -> None:
        """Show only inputs that apply to the selected Anomalib model."""
        is_dinomaly_dinov2 = model_key == "dinomaly_dinov2"
        is_dinomaly_dinov3 = model_key == "dinomaly_dinov3"
        is_superadd = model_key == "superadd_dinov3"
        self.patchcore_group.setVisible(model_key == "patchcore")
        self.dinomaly_group.setVisible(is_dinomaly_dinov2)
        self.dinov3_group.setVisible(is_dinomaly_dinov3 or is_superadd)
        self.dinov3_encoder_label.setVisible(is_dinomaly_dinov3)
        self.dinov3_encoder_combo.setVisible(is_dinomaly_dinov3)
        self.superadd_encoder_label.setVisible(is_superadd)
        self.superadd_encoder_combo.setVisible(is_superadd)
        self.superadd_patch_size_label.setVisible(is_superadd)
        self.superadd_patch_size_spin.setVisible(is_superadd)
        self.superadd_patch_overlap_label.setVisible(is_superadd)
        self.superadd_patch_overlap_spin.setVisible(is_superadd)
        self.dinov3_feature_layers_label.setVisible(is_dinomaly_dinov3 or is_superadd)
        self.dinov3_feature_layers_edit.setVisible(is_dinomaly_dinov3 or is_superadd)
        is_training_model = self._model_definitions[model_key].execution_mode is ModelExecutionMode.TRAIN
        self.trainer_group.setEnabled(is_training_model)
        self.trainer_group.setTitle("Trainer Settings" if is_training_model else "Trainer Settings (Not used for zero-shot evaluation)")
        uses_fixed_one_pass = model_key in {"patchcore", "superadd_dinov3"}
        self.max_epochs_spin.setValue(1 if uses_fixed_one_pass else self.max_epochs_spin.value())
        self.max_epochs_spin.setEnabled(not uses_fixed_one_pass and is_training_model)

        supplemental_models = {
            "draem": ("DRAEM Resources", "Required DTD texture dataset folder"),
            "efficientad": ("EfficientAD Resources", "Required ImageNet or Imagenette folder"),
            "cfm": ("CFM Resources", "Required PointMAE weights file"),
            "glass": ("GLASS Resources", "Optional anomaly-source image folder"),
        }
        supplemental_details = supplemental_models.get(model_key)
        uses_zero_shot_class = model_key == "winclip"
        self.resources_group.setVisible(supplemental_details is not None or uses_zero_shot_class)
        self.resources_form.setRowVisible(self.supplemental_data_label, supplemental_details is not None)
        self.resources_form.setRowVisible(self.zero_shot_class_name_label, uses_zero_shot_class)

        if supplemental_details is not None:
            group_title, placeholder = supplemental_details
            self.resources_group.setTitle(group_title)
            self.supplemental_path_edit.setPlaceholderText(placeholder)
        elif uses_zero_shot_class:
            self.resources_group.setTitle("WinCLIP Prompt")
            self.zero_shot_class_name_edit.setPlaceholderText("Optional object category, for example: bottle")

    def threshold_false_reject_rate(self) -> float:
        """Return the selected normal false-reject rate as a probability."""
        preset = self.threshold_fpr_combo.currentData()
        return float(preset) if preset is not None else self.threshold_fpr_spin.value() / 100

    def dinov3_feature_layers(self) -> tuple[int, ...]:
        """Parse optional explicit feature layers while leaving blank for runtime selection."""
        text = self.dinov3_feature_layers_edit.text().strip()
        if not text:
            return ()
        try:
            return tuple(int(value.strip()) for value in text.split(",") if value.strip())
        except ValueError as exc:
            raise ValueError("DINOv3 feature layers must be comma-separated integers.") from exc

    def _update_threshold_controls(self) -> None:
        """Enable only the optional calibration inputs selected by the user."""
        self.threshold_fpr_spin.setEnabled(self.threshold_fpr_combo.currentData() is None)
        self.minimum_ng_recall_spin.setEnabled(self.minimum_ng_recall_check.isChecked())

