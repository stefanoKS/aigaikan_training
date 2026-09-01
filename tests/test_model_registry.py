"""Installed Anomalib model catalog coverage tests."""

import pytest

from app.core.model_registry import ModelRegistry
from app.core.model_registry import ModelSupportLevel


def test_registry_matches_installed_anomalib_model_exports() -> None:
    """The selectable catalog must cover every public model in the installed release."""
    try:
        import anomalib.models as anomalib_models
    except (ImportError, OSError, RuntimeError) as exc:
        pytest.skip(f"Installed Anomalib runtime could not initialize: {exc}")

    installed_models = set(anomalib_models.__all__)
    catalog_models = {definition.anomalib_class_name for definition in ModelRegistry().official_anomalib_models()}

    assert catalog_models == installed_models


def test_dinomaly_variants_and_superadd_have_distinct_algorithm_identities() -> None:
    registry = ModelRegistry()
    dinov2 = registry.get("dinomaly_dinov2")
    dinov3 = registry.get("dinomaly_dinov3")
    superadd = registry.get("superadd_dinov3")

    assert dinov2.model_variant == "dinomaly_dinov2"
    assert dinov3.model_variant == "dinomaly_dinov3"
    assert dinov2.anomalib_class_name == "Dinomaly"
    assert dinov2.official_anomalib_implementation
    assert dinov3.anomalib_class_name is None
    assert not dinov3.official_anomalib_implementation
    assert dinov3.support_level is ModelSupportLevel.EXPERIMENTAL
    assert superadd.algorithm == "SuperADD"
    assert superadd.model_variant == "superadd_dinov3"
    assert registry.get("Dinomaly").model_variant == "dinomaly_dinov2"