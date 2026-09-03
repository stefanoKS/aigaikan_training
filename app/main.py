"""Application entrypoint."""

from __future__ import annotations

import logging
import sys

from app.core.project_manager import ProjectManager
from app.core.settings_manager import SettingsManager


def configure_logging() -> None:
    """Configure root logging."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _install_dark_dialog_title_bars(app: object) -> None:
    """Apply the Windows dark title-bar attribute when Qt shows a dialog."""
    if sys.platform != "win32":
        return
    from ctypes import byref, c_int, c_void_p, sizeof, windll
    from PySide6.QtCore import QEvent, QObject
    from PySide6.QtWidgets import QDialog

    class DarkDialogTitleBarFilter(QObject):
        def eventFilter(self, watched: QObject, event: QEvent) -> bool:
            if isinstance(watched, QDialog) and event.type() in {QEvent.Type.Show, QEvent.Type.WinIdChange}:
                value = c_int(1)
                window_handle = c_void_p(int(watched.winId()))
                for attribute in (20, 19):
                    if windll.dwmapi.DwmSetWindowAttribute(window_handle, attribute, byref(value), sizeof(value)) == 0:
                        break
            return False

    dialog_filter = DarkDialogTitleBarFilter(app)
    app.installEventFilter(dialog_filter)


def main() -> int:
    """Start the desktop application."""
    configure_logging()
    try:
        from PySide6.QtCore import QSettings, Qt
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
    from app.ui.styles import APP_STYLE

    QApplication.setOrganizationName("AnomalibTrainer")
    QApplication.setApplicationName("AnomalibTrainer")
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeDialogs)
    app = QApplication(sys.argv)
    app.setApplicationDisplayName("Anomalib Trainer")
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLE)
    _install_dark_dialog_title_bars(app)
    settings_manager = SettingsManager()
    settings_manager.app_data_directory().mkdir(parents=True, exist_ok=True)
    settings_manager.default_projects_directory().mkdir(parents=True, exist_ok=True)
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    window = MainWindow(
        settings_manager=settings_manager,
        project_manager=ProjectManager(settings_manager.default_projects_directory()),
    )
    window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
