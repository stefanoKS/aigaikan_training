"""Opt-in real Anomalib SuperADD constructor smoke coverage."""

from __future__ import annotations

import os

import pytest

from app.core.model_registry import ModelRegistry
from app.models.training_config import TrainingConfig
from app.services.anomalib_service import AnomalibService


@pytest.mark.anomalib_smoke
@pytest.mark.parametrize(
    "backbone_id",
    (
        "vit_small_patch16_dinov3.lvd1689m",
        "vit_small_plus_patch16_dinov3.lvd1689m",
        "vit_base_patch16_dinov3.lvd1689m",
        "vit_huge_plus_patch16_dinov3.lvd1689m",
    ),
)
def test_real_anomalib_superadd_constructor_creation(backbone_id: str) -> None:
    """Exercise the installed constructor only when weights/runtime are deliberately available."""
    if os.environ.get("RUN_ANOMALIB_SMOKE") != "1":
        pytest.skip("Set RUN_ANOMALIB_SMOKE=1 with suitable DINOv3 weights and hardware to run real SuperADD constructors.")
    model = AnomalibService()._create_model(
        ModelRegistry().get("super_add"),
        TrainingConfig(model_name="super_add", superadd_backbone_id=backbone_id),
    )

    assert model is not None