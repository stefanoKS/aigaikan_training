"""Application entrypoint."""

from __future__ import annotations

import logging
import sys

from app.core.project_manager import ProjectManager
from app.core.settings_manager import SettingsManager


def configure_logging() -> None:
    """Configure root logging."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> int:
    """Start the desktop application."""
    configure_logging()
    try:
        from PySide6.QtCore import QSettings
        from PySide6.QtWidgets import QApplication
    except Exception as exc:
        logging.exception("PySide6 is unavailable")
        print(
            "PySide6 is not installed. Run scripts/setup.ps1 to create a supported environment.",
            file=sys.stderr,
        )
        print(str(exc), file=sys.stderr)
        return 1

    from app.ui.main_window import MainWindow

    QApplication.setOrganizationName("AnomalibTrainer")
    QApplication.setApplicationName("AnomalibTrainer")
    app = QApplication(sys.argv)
    app.setApplicationDisplayName("Anomalib Trainer")
    settings_manager = SettingsManager()
    settings_manager.app_data_directory().mkdir(parents=True, exist_ok=True)
    settings_manager.default_projects_directory().mkdir(parents=True, exist_ok=True)
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    window = MainWindow(
        settings_manager=settings_manager,
        project_manager=ProjectManager(settings_manager.default_projects_directory()),
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
