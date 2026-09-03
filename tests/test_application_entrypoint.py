"""Application entrypoint tests."""

from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace


def test_application_entrypoint_starts_main_window_maximized(tmp_path: Path, monkeypatch) -> None:
    """The desktop shell should use the available display at launch."""
    import app.main as application_main

    class FakeApplication:
        instance: "FakeApplication | None" = None

        @staticmethod
        def setOrganizationName(_name: str) -> None:
            pass

        @staticmethod
        def setApplicationName(_name: str) -> None:
            pass

        @staticmethod
        def setAttribute(_attribute: object) -> None:
            pass

        def __init__(self, _arguments: list[str]) -> None:
            type(self).instance = self

        def setApplicationDisplayName(self, _name: str) -> None:
            pass

        def setStyle(self, _name: str) -> None:
            pass

        def setStyleSheet(self, _stylesheet: str) -> None:
            pass

        def exec(self) -> int:
            return 0

    class FakeSettingsManager:
        def app_data_directory(self) -> Path:
            return tmp_path / "app-data"

        def default_projects_directory(self) -> Path:
            return tmp_path / "projects"

    class FakeProjectManager:
        def __init__(self, projects_directory: Path) -> None:
            self.projects_directory = projects_directory

    class FakeWindow:
        instance: "FakeWindow | None" = None

        def __init__(self, *, settings_manager: FakeSettingsManager, project_manager: FakeProjectManager) -> None:
            self.settings_manager = settings_manager
            self.project_manager = project_manager
            self.maximized = False
            type(self).instance = self

        def showMaximized(self) -> None:
            self.maximized = True

    core_module = ModuleType("PySide6.QtCore")
    core_module.QSettings = SimpleNamespace(
        Format=SimpleNamespace(IniFormat=object()),
        setDefaultFormat=lambda _format: None,
    )
    core_module.Qt = SimpleNamespace(ApplicationAttribute=SimpleNamespace(AA_DontUseNativeDialogs=object()))
    widgets_module = ModuleType("PySide6.QtWidgets")
    widgets_module.QApplication = FakeApplication
    main_window_module = ModuleType("app.ui.main_window")
    main_window_module.MainWindow = FakeWindow
    styles_module = ModuleType("app.ui.styles")
    styles_module.APP_STYLE = ""
    monkeypatch.setitem(sys.modules, "PySide6.QtCore", core_module)
    monkeypatch.setitem(sys.modules, "PySide6.QtWidgets", widgets_module)
    monkeypatch.setitem(sys.modules, "app.ui.main_window", main_window_module)
    monkeypatch.setitem(sys.modules, "app.ui.styles", styles_module)
    monkeypatch.setattr(application_main, "configure_logging", lambda: None)
    monkeypatch.setattr(application_main, "_install_dark_dialog_title_bars", lambda _application: None)
    monkeypatch.setattr(application_main, "SettingsManager", FakeSettingsManager)
    monkeypatch.setattr(application_main, "ProjectManager", FakeProjectManager)

    assert application_main.main() == 0
    assert FakeWindow.instance is not None
    assert FakeWindow.instance.maximized