"""Environment inspection helpers."""

from __future__ import annotations

import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

from app.version import APP_VERSION


def _module_version(name: str) -> str:
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:
        return "not-installed"


def collect_environment_info(project_path: Path, seed: int) -> dict[str, str | bool | int]:
    """Collect environment metadata for persisted training runs."""
    cuda_available = False
    cuda_version = "not-available"
    gpu_name = "not-available"
    try:
        import torch

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            cuda_available = bool(torch.cuda.is_available())
            cuda_version = str(getattr(torch.version, "cuda", "not-available"))
            if cuda_available:
                gpu_name = str(torch.cuda.get_device_name(0))
    except Exception:
        pass
    return {
        "application_version": APP_VERSION,
        "python_version": platform.python_version(),
        "windows_version": platform.platform(),
        "anomalib_version": _module_version("anomalib"),
        "torch_version": _module_version("torch"),
        "torchvision_version": _module_version("torchvision"),
        "pyside6_version": _module_version("PySide6"),
        "cuda_available": cuda_available,
        "cuda_runtime_version": cuda_version,
        "gpu_name": gpu_name,
        "project_path": str(project_path),
        "training_date": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
        "python_executable": sys.executable,
    }
