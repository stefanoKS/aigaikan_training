"""Verify local installation prerequisites."""

from __future__ import annotations

import importlib
import platform
import sys
import warnings
from argparse import ArgumentParser
from pathlib import Path


def check_import(name: str) -> None:
    module = importlib.import_module(name)
    print(f"{name}: {getattr(module, '__version__', 'ok')}")


def _verify_cuda_support() -> None:
    """Confirm the installed PyTorch build includes the active GPU architecture."""
    import torch

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested, but PyTorch cannot access an NVIDIA GPU.")
        major, minor = torch.cuda.get_device_capability(0)
        architecture = f"sm_{major}{minor}"
        supported_architectures = torch.cuda.get_arch_list()
        if architecture not in supported_architectures:
            raise SystemExit(
                f"PyTorch {torch.__version__} does not support {architecture}. "
                f"Supported architectures: {', '.join(supported_architectures)}"
            )
        print(f"CUDA GPU OK: {torch.cuda.get_device_name(0)} ({architecture})")


def main(require_cuda: bool = False) -> int:
    if sys.version_info[:2] != (3, 11):
        raise SystemExit("Python 3.11 is required.")
    print(f"Python: {platform.python_version()}")
    for module_name in ("PySide6", "torch", "torchvision", "anomalib"):
        check_import(module_name)
    import anomalib
    from anomalib.models import Dinomaly, Padim, Patchcore

    if anomalib.__version__ != "2.6.0":
        raise SystemExit(f"Anomalib 2.6.0 is required; found {anomalib.__version__}.")
    print(f"Available models: {Patchcore.__name__}, {Padim.__name__}, {Dinomaly.__name__}")
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
    if require_cuda:
        _verify_cuda_support()
    return 0


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--require-cuda", action="store_true")
    arguments = parser.parse_args()
    raise SystemExit(main(require_cuda=arguments.require_cuda))
