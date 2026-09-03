"""SuperADD backbone configuration and factory tests."""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

from app.core.model_registry import ModelRegistry
from app.core.superadd_backbone_registry import LEGACY_HUGE_BACKBONE_ID, SuperAddBackboneRegistry
from app.models.training_config import TrainingConfig
from app.services.anomalib_service import AnomalibService
from app.ui.pages.config_page import ConfigPage


def test_curated_superadd_backbones_expose_stable_timm_identifiers() -> None:
    presets = SuperAddBackboneRegistry().all()

    assert [preset.identifier for preset in presets] == [
        "vit_small_patch16_dinov3.lvd1689m",
        "vit_small_plus_patch16_dinov3.lvd1689m",
        "vit_base_patch16_dinov3.lvd1689m",
        "vit_large_patch16_dinov3.lvd1689m",
        "vit_huge_plus_patch16_dinov3.lvd1689m",
    ]
    assert [preset.patch_size for preset in presets] == [16, 16, 16, 16, 16]


def test_superadd_registry_rejects_unavailable_or_arbitrary_backbones(monkeypatch) -> None:
    registry = SuperAddBackboneRegistry()
    monkeypatch.setattr(registry, "_available_backbone_ids", lambda: frozenset())

    with pytest.raises(ValueError, match="unavailable"):
        registry.validate("vit_small_patch16_dinov3.lvd1689m")
    with pytest.raises(ValueError, match="Unsupported"):
        registry.get("arbitrary-backbone")


def test_superadd_config_round_trip_and_historic_default() -> None:
    config = TrainingConfig(
        model_name="super_add",
        superadd_backbone_id="vit_small_plus_patch16_dinov3.lvd1689m",
        superadd_precision="float16",
    )

    restored = TrainingConfig.from_dict(config.to_dict())
    historic = TrainingConfig.from_dict({"model_name": "super_add"})

    assert restored.superadd_backbone_id == "vit_small_plus_patch16_dinov3.lvd1689m"
    assert restored.superadd_precision == "float16"
    assert historic.superadd_backbone_name == LEGACY_HUGE_BACKBONE_ID
    assert historic.superadd_precision == "float32"
    assert historic.model_profile()["layers"] == "automatic"
    assert historic.model_profile()["score_quantile"] == 0.001
    assert historic.model_profile()["memory_bank_limit"] == 100000
    assert historic.model_profile()["memory_bank_limit_source"] == "anomalib_default"


def test_superadd_factory_passes_the_selected_backbone_and_precision() -> None:
    class FakeSuperADD:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    config = TrainingConfig(
        model_name="super_add",
        superadd_backbone_id="vit_base_patch16_dinov3.lvd1689m",
        superadd_precision="float16",
    )
    model = AnomalibService()._create_super_add_model(
        FakeSuperADD,
        ModelRegistry().get("super_add"),
        config,
        None,
    )

    assert model.kwargs == {
        "backbone": "vit_base_patch16_dinov3.lvd1689m",
        "precision": "float16",
        "patch_size": 448,
        "patch_overlap": 16,
    }


def test_superadd_fp16_rejects_cpu_and_allows_cuda() -> None:
    config = TrainingConfig(model_name="super_add", superadd_precision="float16")

    with pytest.raises(ValueError, match="requires CUDA"):
        AnomalibService._validate_superadd_precision(config, "cpu")
    AnomalibService._validate_superadd_precision(config, "gpu")


def test_superadd_config_rejects_unavailable_backbone_and_invalid_precision(monkeypatch) -> None:
    config = TrainingConfig(model_name="super_add", superadd_backbone_id="vit_small_patch16_dinov3.lvd1689m")
    monkeypatch.setattr(SuperAddBackboneRegistry, "validate", lambda _self, _identifier: (_ for _ in ()).throw(ValueError("unavailable")))

    with pytest.raises(ValueError, match="unavailable"):
        config.validate()

    monkeypatch.undo()
    with pytest.raises(ValueError, match="precision"):
        TrainingConfig(model_name="super_add", superadd_precision="bf16").validate()


def test_superadd_settings_are_visible_and_disable_generic_aggregation(monkeypatch) -> None:
    application = QApplication.instance() or QApplication([])
    page = ConfigPage()
    monkeypatch.setattr(page.superadd_backbone_registry, "is_available", lambda _preset: True)
    page.show()
    page.model_combo.setCurrentIndex(page.model_combo.findData("super_add"))
    application.processEvents()

    page.set_superadd_settings(LEGACY_HUGE_BACKBONE_ID, "float16")

    assert page.superadd_group.isVisible()
    assert page.superadd_backbone_combo.currentData() == "vit_huge_plus_patch16_dinov3.lvd1689m"
    assert page.superadd_precision_combo.currentData() == "float16"
    assert page.superadd_feature_layers_label.text() == "Automatic"
    assert not page.score_aggregation_combo.isEnabled()
    assert not page.top_k_fraction_spin.isEnabled()
    assert page.superadd_score_aggregation_note.isVisible()
    page.close()