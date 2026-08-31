"""Installed Anomalib model catalog coverage tests."""

import pytest

from app.core.model_registry import ModelRegistry


def test_registry_matches_installed_anomalib_model_exports() -> None:
    """The selectable catalog must cover every public model in the installed release."""
    anomalib_models = pytest.importorskip("anomalib.models")

    installed_models = set(anomalib_models.__all__)
    catalog_models = {definition.anomalib_class_name for definition in ModelRegistry().all()}

    assert catalog_models == installed_models