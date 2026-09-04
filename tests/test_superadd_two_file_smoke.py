"""Opt-in real SuperADD two-file deployment smoke test."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from app.core.deployment_package import DeploymentPackage
from app.core.prediction_contract import SUPERADD_NATIVE_IMAGE_SCORE_SEMANTIC


@pytest.mark.anomalib_smoke
def test_real_superadd_two_file_package_loads_offline_when_explicitly_provided(monkeypatch) -> None:
    """Verify a manually evidenced SuperADD package without training-dir access or network access."""
    package_value = os.environ.get("SUPERADD_TWO_FILE_DEPLOYMENT")
    frame_value = os.environ.get("SUPERADD_TWO_FILE_FRAME")
    if not package_value or not frame_value:
        pytest.skip("Set SUPERADD_TWO_FILE_DEPLOYMENT and SUPERADD_TWO_FILE_FRAME to run real offline SuperADD deployment smoke.")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    with Image.open(frame_value) as image:
        raw = np.asarray(image)

    deployment = DeploymentPackage.load(Path(package_value), device="cpu")
    result = deployment.predict(raw)

    assert result.score_semantic == SUPERADD_NATIVE_IMAGE_SCORE_SEMANTIC
    assert np.isfinite(result.decision_score)
    assert np.isfinite(result.anomaly_map).all()