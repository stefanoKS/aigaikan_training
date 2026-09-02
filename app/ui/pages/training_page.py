"""Training page."""

from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class TrainingPage(QWidget):
    """Training control UI."""

    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 28)
        root.setSpacing(16)

        button_row = QHBoxLayout()
        self.start_button = QPushButton("Start Training")
        self.start_button.setObjectName("PrimaryButton")
        self.cancel_button = QPushButton("Cancel Training")
        self.cancel_button.setObjectName("AlertButton")
        self.cancel_button.setEnabled(False)
        self.open_log_button = QPushButton("Open Log File")
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.cancel_button)
        button_row.addWidget(self.open_log_button)
        button_row.addStretch(1)
        root.addLayout(button_row)

        status_group = QGroupBox("Training Status")
        status_form = QFormLayout(status_group)
        self.current_stage_label = QLabel("-")
        self.elapsed_time_label = QLabel("00:00:00")
        self.active_model_label = QLabel("PatchCore")
        self.active_device_label = QLabel("Auto")
        self.dataset_counts_label = QLabel("-")
        self.stage_progress = QProgressBar()
        self.overall_progress = QProgressBar()
        status_form.addRow("Current Stage", self.current_stage_label)
        status_form.addRow("Stage Progress", self.stage_progress)
        status_form.addRow("Overall Progress", self.overall_progress)
        status_form.addRow("Elapsed Time", self.elapsed_time_label)
        status_form.addRow("Active Model", self.active_model_label)
        status_form.addRow("Active Device", self.active_device_label)
        status_form.addRow("Dataset Counts", self.dataset_counts_label)
        root.addWidget(status_group)

        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        root.addWidget(self.log_output, stretch=1)

    def append_log(self, level: str, message: str) -> None:
        """Append an event and keep the active training view on the newest line."""
        scrollbar = self.log_output.verticalScrollBar()
        follow_latest = scrollbar.value() >= scrollbar.maximum()
        self.log_output.appendPlainText(f"[{level.upper()}] {message}")
        if follow_latest:
            scrollbar.setValue(scrollbar.maximum())

