"""Serialized Torch adapter for SuperADD's native decision-score contract."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
from torch import nn
from torchvision.transforms.v2.functional import to_dtype, to_image


class SuperADDDeploymentOutput(NamedTuple):
    """Raw SuperADD inference values kept outside Anomalib postprocessing."""

    decision_score: torch.Tensor
    anomaly_map: torch.Tensor


class SuperADDDeploymentAdapter(nn.Module):
    """Persist the trained SuperADD preprocessor and detector with raw outputs."""

    def __init__(self, trained_model: nn.Module) -> None:
        super().__init__()
        detector = getattr(trained_model, "model", None)
        if not isinstance(detector, nn.Module):
            raise ValueError("A trained SuperADD checkpoint must provide its detector module.")
        self.pre_processor = getattr(trained_model, "pre_processor", None)
        self.model = detector

    def forward(self, image: torch.Tensor) -> SuperADDDeploymentOutput:
        """Return native top-quantile score and continuous map before postprocessing."""
        prepared = self.pre_processor(image) if self.pre_processor else image
        prediction = self.model(prepared)
        decision_score = getattr(prediction, "pred_score", None)
        anomaly_map = getattr(prediction, "anomaly_map", None)
        if decision_score is None or anomaly_map is None:
            raise RuntimeError("The trained SuperADD detector did not return native score and anomaly map values.")
        return SuperADDDeploymentOutput(decision_score, anomaly_map)


class SuperADDDeploymentInferencer:
    """Load only a trusted SuperADD deployment adapter and preserve its raw output fields."""

    def __init__(self, model: SuperADDDeploymentAdapter, device: torch.device) -> None:
        self.model = model
        self.device = device

    @classmethod
    def load(
        cls,
        path: Path,
        device: str = "cpu",
        *,
        trust_newly_created_local_artifact: bool = False,
    ) -> "SuperADDDeploymentInferencer":
        """Load a generated local artifact without changing process trust environment state."""
        model_path = path.expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"SuperADD deployment model is missing: {model_path}")
        if not trust_newly_created_local_artifact and not _remote_code_is_trusted():
            raise ValueError(
                "Loading model.pt requires TRUST_REMOTE_CODE=1 because PyTorch pickle can execute code. "
                "Load only deployment artifacts from a trusted source."
            )
        resolved_device = _resolve_device(device)
        try:
            payload = torch.load(model_path, map_location=resolved_device, weights_only=False)
        except Exception as exc:
            raise ValueError(f"Could not load the SuperADD Torch deployment artifact: {model_path}") from exc
        model = payload.get("model") if isinstance(payload, Mapping) else None
        if not isinstance(model, SuperADDDeploymentAdapter):
            raise ValueError("model.pt does not contain the required SuperADD deployment adapter.")
        return cls(model.eval().to(resolved_device), resolved_device)

    def predict(self, image: np.ndarray) -> SuperADDDeploymentOutput:
        """Convert canonical RGB uint8 pixels like TorchInferencer without recreating scores."""
        values = np.asarray(image)
        if values.dtype != np.uint8 or values.ndim != 3 or values.shape[2] != 3:
            raise ValueError("SuperADD deployment inference requires an RGB HWC uint8 image.")
        tensor = to_dtype(to_image(values), torch.float32, scale=True).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model(tensor)


def _remote_code_is_trusted() -> bool:
    return os.environ.get("TRUST_REMOTE_CODE", "0").casefold() in {"1", "true"}


def _resolve_device(device: str) -> torch.device:
    if device not in {"cpu", "cuda"}:
        raise ValueError("SuperADD deployment device must be cpu or cuda.")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested for SuperADD deployment but is not available.")
    return torch.device(device)