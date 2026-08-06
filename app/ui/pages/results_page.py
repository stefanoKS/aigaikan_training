"""Results page."""

from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
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
        self.export_model_button = QPushButton("Export Model")
        self.compare_button = QPushButton("Compare Runs")
        header.addWidget(QLabel("Filter"))
        header.addWidget(self.filter_combo)
        header.addStretch(1)
        header.addWidget(self.export_csv_button)
        header.addWidget(self.export_json_button)
        header.addWidget(self.open_folder_button)
        header.addWidget(self.export_model_button)
        header.addWidget(self.compare_button)
        root.addLayout(header)

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

    def set_metrics(self, metrics: dict[str, str]) -> None:
        """Populate the metrics table."""
        items = list(metrics.items())
        self.metrics_table.setRowCount(len(items))
        for row_index, (key, value) in enumerate(items):
            self.metrics_table.setItem(row_index, 0, QTableWidgetItem(key))
            self.metrics_table.setItem(row_index, 1, QTableWidgetItem(value))

