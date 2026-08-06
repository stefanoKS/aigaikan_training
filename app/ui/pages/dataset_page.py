"""Dataset management page."""

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
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class DatasetPage(QWidget):
    """Dataset management UI."""

    ROLES = (
        ("OK Training Images", "ok_train"),
        ("OK Test Images", "ok_test"),
        ("NG Test Images", "ng_test"),
        ("Optional Masks", "masks"),
    )

    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)

        header_row = QHBoxLayout()
        self.import_mode_combo = QComboBox()
        self.import_mode_combo.addItems(["Copy images into project", "Reference original folder"])
        self.validate_button = QPushButton("Validate Dataset")
        self.clear_button = QPushButton("Clear Dataset Selection")
        header_row.addWidget(QLabel("Import Mode"))
        header_row.addWidget(self.import_mode_combo)
        header_row.addStretch(1)
        header_row.addWidget(self.validate_button)
        header_row.addWidget(self.clear_button)
        root.addLayout(header_row)

        splitter = QSplitter()
        left_column = QWidget()
        left_layout = QVBoxLayout(left_column)
        self.role_widgets: dict[str, dict[str, QWidget]] = {}
        for title, key in self.ROLES:
            group = QGroupBox(title)
            form = QFormLayout(group)
            path_label = QLabel("-")
            count_label = QLabel("0")
            invalid_label = QLabel("0")
            resolution_label = QLabel("-")
            color_label = QLabel("-")
            import_button = QPushButton("Import")
            browse_button = QPushButton("Browse")
            thumb_label = QLabel("Thumbnail preview")
            form.addRow("Folder Path", path_label)
            form.addRow("Image Count", count_label)
            form.addRow("Invalid Images", invalid_label)
            form.addRow("Typical Resolution", resolution_label)
            form.addRow("Color Mode", color_label)
            button_row = QHBoxLayout()
            button_row.addWidget(import_button)
            button_row.addWidget(browse_button)
            form.addRow("Actions", button_row)
            form.addRow("Preview", thumb_label)
            left_layout.addWidget(group)
            self.role_widgets[key] = {
                "path": path_label,
                "count": count_label,
                "invalid": invalid_label,
                "resolution": resolution_label,
                "color": color_label,
                "import_button": import_button,
                "browse_button": browse_button,
                "preview": thumb_label,
            }
        left_layout.addStretch(1)
        splitter.addWidget(left_column)

        right_column = QWidget()
        right_layout = QVBoxLayout(right_column)
        self.validation_table = QTableWidget(0, 4)
        self.validation_table.setHorizontalHeaderLabels(["Level", "Role", "Message", "Path"])
        self.validation_summary = QTextEdit()
        self.validation_summary.setReadOnly(True)
        right_layout.addWidget(QLabel("Dataset Validation"))
        right_layout.addWidget(self.validation_table, stretch=2)
        right_layout.addWidget(QLabel("Summary"))
        right_layout.addWidget(self.validation_summary, stretch=1)
        splitter.addWidget(right_column)
        root.addWidget(splitter, stretch=1)

    def set_validation_rows(self, rows: list[tuple[str, str, str, str]]) -> None:
        """Populate validation rows."""
        self.validation_table.setRowCount(len(rows))
        for row_index, values in enumerate(rows):
            for column_index, value in enumerate(values):
                self.validation_table.setItem(row_index, column_index, QTableWidgetItem(value))

