"""Home/projects page."""

from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class HomePage(QWidget):
    """Project landing page."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 28)
        layout.setSpacing(16)

        action_row = QHBoxLayout()
        self.new_project_button = QPushButton("New Project")
        self.new_project_button.setObjectName("PrimaryButton")
        self.open_project_button = QPushButton("Open Project")
        self.save_project_button = QPushButton("Save Project")
        self.save_project_button.setEnabled(False)
        action_row.addWidget(self.new_project_button)
        action_row.addWidget(self.open_project_button)
        action_row.addWidget(self.save_project_button)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        summary_group = QGroupBox("Current Project")
        summary_form = QFormLayout(summary_group)
        self.project_name_label = QLabel("-")
        self.project_path_label = QLabel("-")
        self.created_date_label = QLabel("-")
        self.last_opened_label = QLabel("-")
        self.status_label = QLabel("Not trained")
        summary_form.addRow("Project Name", self.project_name_label)
        summary_form.addRow("Project Path", self.project_path_label)
        summary_form.addRow("Created", self.created_date_label)
        summary_form.addRow("Last Opened", self.last_opened_label)
        summary_form.addRow("Last Training Status", self.status_label)
        layout.addWidget(summary_group)

        recent_group = QGroupBox("Recent Projects")
        recent_layout = QVBoxLayout(recent_group)
        self.recent_projects_list = QListWidget()
        self.recent_projects_list.setObjectName("RecentProjectsList")
        recent_layout.addWidget(self.recent_projects_list)
        layout.addWidget(recent_group, stretch=1)

