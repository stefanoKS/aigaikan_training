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

from app.core.model_registry import ModelExecutionMode, ModelRegistry


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
        for definition in self.model_registry.all():
            self.model_combo.addItem(definition.display_name, definition.key)
            item = getattr(self.model_combo.model(), "item", lambda _index: None)(self.model_combo.count() - 1)
            if item is not None:
                item.setEnabled(definition.supports_image_folder)
                item.setToolTip(definition.requirement or "Supported image-folder model")
        self.model_support_label = QLabel()
        self.model_support_label.setObjectName("ModelSupport")
        self.model_support_label.setWordWrap(True)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["Auto", "CUDA", "CPU"])
        self.image_width_spin = QSpinBox()
        self.image_width_spin.setRange(32, 8192)
        self.image_width_spin.setValue(256)
        self.image_height_spin = QSpinBox()
        self.image_height_spin.setRange(32, 8192)
        self.image_height_spin.setValue(256)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 512)
        self.batch_size_spin.setValue(8)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(42)
        basic_form.addRow("Model", self.model_combo)
        basic_form.addRow("Compatibility", self.model_support_label)
        basic_form.addRow("Device", self.device_combo)
        basic_form.addRow("Image Width", self.image_width_spin)
        basic_form.addRow("Image Height", self.image_height_spin)
        basic_form.addRow("Batch Size", self.batch_size_spin)
        basic_form.addRow("Random Seed", self.seed_spin)
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
        trainer_form.addRow("Max Epochs", self.max_epochs_spin)
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

        self.dinomaly_group = QGroupBox("Dinomaly Settings")
        dinomaly_form = QFormLayout(self.dinomaly_group)
        self.dinomaly_encoder_combo = QComboBox()
        self.dinomaly_encoder_combo.addItems(
            [
                "vit_small_patch16_dinov3.lvd1689m",
                "vit_base_patch16_dinov3.lvd1689m",
                "vit_large_patch16_dinov3.lvd1689m",
            ]
        )
        self.dinomaly_decoder_depth_spin = QSpinBox()
        self.dinomaly_decoder_depth_spin.setRange(2, 32)
        self.dinomaly_decoder_depth_spin.setValue(8)
        self.dinomaly_dropout_spin = QDoubleSpinBox()
        self.dinomaly_dropout_spin.setRange(0.0, 0.99)
        self.dinomaly_dropout_spin.setSingleStep(0.05)
        self.dinomaly_dropout_spin.setValue(0.2)
        self.dinomaly_context_recentering_check = QCheckBox()
        dinomaly_form.addRow("DINOv3 Encoder", self.dinomaly_encoder_combo)
        dinomaly_form.addRow("Decoder Depth", self.dinomaly_decoder_depth_spin)
        dinomaly_form.addRow("Bottleneck Dropout", self.dinomaly_dropout_spin)
        dinomaly_form.addRow("Context Recentering", self.dinomaly_context_recentering_check)
        root.addWidget(self.dinomaly_group)

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
        details = [data_contract, execution]
        if definition.requirement:
            details.append(definition.requirement)
        self.model_support_label.setText(" | ".join(details))
        self._update_model_controls(definition.key)

    def _update_model_controls(self, model_key: str) -> None:
        """Show only inputs that apply to the selected Anomalib model."""
        self.patchcore_group.setVisible(model_key == "patchcore")
        self.dinomaly_group.setVisible(model_key == "dinomaly")
        is_training_model = self._model_definitions[model_key].execution_mode is ModelExecutionMode.TRAIN
        self.trainer_group.setEnabled(is_training_model)
        self.trainer_group.setTitle("Trainer Settings" if is_training_model else "Trainer Settings (Not used for zero-shot evaluation)")

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

