"""Verify local installation prerequisites."""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path


def check_import(name: str) -> None:
    module = importlib.import_module(name)
    print(f"{name}: {getattr(module, '__version__', 'ok')}")


def main() -> int:
    if sys.version_info[:2] != (3, 11):
        raise SystemExit("Python 3.11 is required.")
    print(f"Python: {platform.python_version()}")
    for module_name in ("PySide6", "torch", "torchvision", "anomalib"):
        check_import(module_name)
    from PySide6.QtWidgets import QApplication

    app = QApplication([])
    app.quit()
    target = Path.home() / "Documents" / "AnomalibProjects"
    target.mkdir(parents=True, exist_ok=True)
    print(f"Project directory OK: {target}")
    weights_file = Path("weights") / "wide_resnet50_2-default.pth"
    if weights_file.exists():
        print(f"Weights OK: {weights_file}")
    else:
        print(f"Weights missing: {weights_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
