"""Main application window."""

from __future__ import annotations

from typing import cast

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStackedWidget,
    QWidget,
)

from app.core.project_manager import ProjectManager
from app.core.settings_manager import SettingsManager
from app.ui.pages.config_page import ConfigPage
from app.ui.pages.dataset_page import DatasetPage
from app.ui.pages.home_page import HomePage
from app.ui.pages.inference_page import InferencePage
from app.ui.pages.results_page import ResultsPage
from app.ui.pages.training_page import TrainingPage
from app.ui.styles import APP_STYLE


class MainWindow(QMainWindow):
    """Main application shell with left navigation."""

    PAGE_DEFINITIONS = (
        ("Home / Projects", HomePage),
        ("Dataset", DatasetPage),
        ("Training Configuration", ConfigPage),
        ("Training", TrainingPage),
        ("Results", ResultsPage),
        ("Inference", InferencePage),
    )

    def __init__(self, settings_manager: SettingsManager, project_manager: ProjectManager) -> None:
        super().__init__()
        self.settings_manager = settings_manager
        self.project_manager = project_manager
        self.setWindowTitle("Anomalib Trainer")
        self.resize(1400, 900)
        self.setStyleSheet(APP_STYLE)

        splitter = QSplitter()
        self.navigation = QListWidget()
        self.navigation.setMaximumWidth(240)
        self.pages = QStackedWidget()
        self.page_instances: dict[str, QWidget] = {}

        for index, (title, page_type) in enumerate(self.PAGE_DEFINITIONS):
            self.navigation.addItem(QListWidgetItem(title))
            page = page_type()
            self.page_instances[title] = page
            self.pages.addWidget(page)
            if index == 0:
                self.navigation.setCurrentRow(0)

        self.navigation.currentRowChanged.connect(self.pages.setCurrentIndex)
        splitter.addWidget(self.navigation)
        splitter.addWidget(self.pages)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

    def show_dependency_error(self, message: str, details: str = "") -> None:
        """Show a friendly dependency error."""
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Critical)
        dialog.setWindowTitle("Missing Dependencies")
        dialog.setText(message)
        if details:
            dialog.setDetailedText(details)
        dialog.exec()

