"""Tests for curated Dinomaly encoder selection."""

import sys
from types import ModuleType

from PySide6.QtWidgets import QApplication

from app.core.dinomaly_encoder_registry import DinomalyEncoderRegistry
from app.ui.pages.config_page import ConfigPage


def test_encoder_registry_marks_only_runtime_registered_presets_available(monkeypatch) -> None:
    registry = DinomalyEncoderRegistry()
    DinomalyEncoderRegistry._available_encoder_ids.cache_clear()
    fake_timm = ModuleType("timm")
    fake_timm.is_model = lambda identifier: identifier == "vit_small_patch14_reg4_dinov2"
    monkeypatch.setitem(sys.modules, "timm", fake_timm)

    assert registry.is_available(registry.get("vit_small_patch14_reg4_dinov2"))
    assert not registry.is_available(registry.get("vit_base_patch16_dinov3.lvd1689m"))

    DinomalyEncoderRegistry._available_encoder_ids.cache_clear()


def test_configuration_disables_unavailable_dinomaly_encoder(monkeypatch) -> None:
    application = QApplication.instance() or QApplication([])
    page = ConfigPage()
    unavailable_identifier = "vit_base_patch16_dinov3.lvd1689m"
    monkeypatch.setattr(
        page.dinomaly_encoder_registry,
        "is_available",
        lambda preset: preset.identifier != unavailable_identifier,
    )

    page.model_combo.setCurrentIndex(page.model_combo.findData("dinomaly_dinov3"))
    application.processEvents()

    unavailable_index = page.dinomaly_encoder_combo.findData(unavailable_identifier)
    unavailable_item = page.dinomaly_encoder_combo.model().item(unavailable_index)
    assert unavailable_item is not None
    assert not unavailable_item.isEnabled()
    assert page.dinomaly_encoder_combo.currentData() != unavailable_identifier
    page.close()