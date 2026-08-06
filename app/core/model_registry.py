"""Model registry for supported Anomalib models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelDefinition:
    """Descriptor for a supported model."""

    key: str
    display_name: str
    anomalib_class_name: str
    supports_export: bool = True


class ModelRegistry:
    """Registry for available models."""

    def __init__(self) -> None:
        self._models = {
            "patchcore": ModelDefinition(
                key="patchcore",
                display_name="PatchCore",
                anomalib_class_name="Patchcore",
            )
        }

    def all(self) -> list[ModelDefinition]:
        """Return all registered models."""
        return list(self._models.values())

    def get(self, key: str) -> ModelDefinition:
        """Get a model definition by key."""
        return self._models[key.lower()]

