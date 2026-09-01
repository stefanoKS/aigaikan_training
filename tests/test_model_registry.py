"""Installed Anomalib model catalog coverage tests."""

import pytest

from app.core.model_registry import ModelRegistry, ModelSupportLevel


def test_registry_contains_only_the_supported_production_configurations() -> None:
    registry = ModelRegistry()

    assert [definition.key for definition in registry.all()] == [
        "patchcore",
        "padim",
        "dinomaly_dinov2",
        "dinomaly_dinov3",
    ]
    assert {definition.anomalib_class_name for definition in registry.official_anomalib_models()} == {
        "Patchcore",
        "Padim",
        "Dinomaly",
    }


def test_dinomaly_variants_have_distinct_encoder_identities_and_share_stock_implementation() -> None:
    registry = ModelRegistry()
    dinov2 = registry.get("dinomaly_dinov2")
    dinov3 = registry.get("dinomaly_dinov3")

    assert dinov2.model_variant == "dinomaly_dinov2"
    assert dinov3.model_variant == "dinomaly_dinov3"
    assert dinov2.anomalib_class_name == "Dinomaly"
    assert dinov3.anomalib_class_name == "Dinomaly"
    assert dinov2.official_anomalib_implementation
    assert dinov3.official_anomalib_implementation
    assert dinov3.support_level is ModelSupportLevel.SUPPORTED
    assert dinov2.encoder_family == "DINOv2"
    assert dinov3.encoder_family == "DINOv3"
    assert registry.get("Dinomaly").model_variant == "dinomaly_dinov2"