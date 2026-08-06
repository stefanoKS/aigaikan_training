"""Inference page."""

from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)


class InferencePage(QWidget):
    """Inference UI."""

    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)

        controls = QHBoxLayout()
        self.load_run_button = QPushButton("Load Training Run")
        self.select_image_button = QPushButton("Select Image")
        self.select_folder_button = QPushButton("Select Folder")
        self.run_inference_button = QPushButton("Run Inference")
        self.export_csv_button = QPushButton("Export Inference CSV")
        controls.addWidget(self.load_run_button)
        controls.addWidget(self.select_image_button)
        controls.addWidget(self.select_folder_button)
        controls.addWidget(self.run_inference_button)
        controls.addWidget(self.export_csv_button)
        controls.addStretch(1)
        root.addLayout(controls)

        summary_group = QGroupBox("Inference Summary")
        summary_form = QFormLayout(summary_group)
        self.model_label = QLabel("-")
        self.score_label = QLabel("-")
        self.prediction_label = QLabel("-")
        self.threshold_label = QLabel("-")
        summary_form.addRow("Model", self.model_label)
        summary_form.addRow("Anomaly Score", self.score_label)
        summary_form.addRow("Predicted Result", self.prediction_label)
        summary_form.addRow("Threshold", self.threshold_label)
        root.addWidget(summary_group)

        self.results_table = QTableWidget(0, 5)
        self.results_table.setHorizontalHeaderLabels(["Source", "Prediction", "Score", "Threshold", "Overlay"])
        root.addWidget(self.results_table, stretch=1)

