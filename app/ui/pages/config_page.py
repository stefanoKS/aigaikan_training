"""Training configuration page."""

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
)


class ConfigPage(QWidget):
    """Training configuration UI."""

    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)

        basic_group = QGroupBox("Training Configuration")
        basic_form = QFormLayout(basic_group)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["PatchCore"])
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
        basic_form.addRow("Device", self.device_combo)
        basic_form.addRow("Image Width", self.image_width_spin)
        basic_form.addRow("Image Height", self.image_height_spin)
        basic_form.addRow("Batch Size", self.batch_size_spin)
        basic_form.addRow("Random Seed", self.seed_spin)
        root.addWidget(basic_group)

        advanced_group = QGroupBox("Advanced Settings")
        advanced_group.setCheckable(True)
        advanced_group.setChecked(False)
        advanced_form = QFormLayout(advanced_group)
        self.coreset_ratio_spin = QDoubleSpinBox()
        self.coreset_ratio_spin.setRange(0.01, 1.0)
        self.coreset_ratio_spin.setSingleStep(0.01)
        self.coreset_ratio_spin.setValue(0.1)
        self.neighbors_spin = QSpinBox()
        self.neighbors_spin.setRange(1, 99)
        self.neighbors_spin.setValue(9)
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 64)
        self.workers_spin.setValue(0)
        self.cuda_available_label = QLabel("Unknown")
        self.gpu_name_label = QLabel("-")
        self.gpu_memory_label = QLabel("-")
        advanced_form.addRow("Coreset Sampling Ratio", self.coreset_ratio_spin)
        advanced_form.addRow("Nearest Neighbors", self.neighbors_spin)
        advanced_form.addRow("Data-loader Workers", self.workers_spin)
        advanced_form.addRow("CUDA Available", self.cuda_available_label)
        advanced_form.addRow("GPU Name", self.gpu_name_label)
        advanced_form.addRow("GPU Memory", self.gpu_memory_label)
        root.addWidget(advanced_group)

        button_row = QHBoxLayout()
        self.save_button = QPushButton("Save Configuration")
        self.load_button = QPushButton("Load Configuration")
        self.reset_button = QPushButton("Reset to Defaults")
        button_row.addWidget(self.save_button)
        button_row.addWidget(self.load_button)
        button_row.addWidget(self.reset_button)
        button_row.addStretch(1)
        root.addLayout(button_row)
        root.addStretch(1)

