"""Curated, runtime-validated timm encoders for Dinomaly configurations."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True, slots=True)
class DinomalyEncoderPreset:
    """One intentional Dinomaly encoder choice with a stable persisted ID."""

    identifier: str
    display_name: str
    family: str
    patch_size: int


class DinomalyEncoderRegistry:
    """Expose only tested DINO encoder presets and their installed-runtime availability."""

    _PRESETS = (
        DinomalyEncoderPreset("vit_small_patch14_reg4_dinov2", "DINOv2 Small", "DINOv2", 14),
        DinomalyEncoderPreset("vit_base_patch14_reg4_dinov2", "DINOv2 Base", "DINOv2", 14),
        DinomalyEncoderPreset("vit_large_patch14_reg4_dinov2", "DINOv2 Large", "DINOv2", 14),
        DinomalyEncoderPreset("vit_small_patch16_dinov3.lvd1689m", "DINOv3 Small", "DINOv3", 16),
        DinomalyEncoderPreset("vit_base_patch16_dinov3.lvd1689m", "DINOv3 Base", "DINOv3", 16),
        DinomalyEncoderPreset("vit_large_patch16_dinov3.lvd1689m", "DINOv3 Large", "DINOv3", 16),
    )

    def all(self, family: str | None = None) -> tuple[DinomalyEncoderPreset, ...]:
        """Return every curated preset, optionally restricted to one DINO family."""
        return tuple(preset for preset in self._PRESETS if family is None or preset.family == family)

    def get(self, identifier: str) -> DinomalyEncoderPreset:
        """Return one curated encoder by its permanent timm identifier."""
        for preset in self._PRESETS:
            if preset.identifier == identifier:
                return preset
        raise ValueError(f"Unsupported Dinomaly encoder: {identifier}")

    def is_available(self, preset: DinomalyEncoderPreset) -> bool:
        """Report whether the pinned timm environment currently resolves the exact preset ID."""
        return preset.identifier in self._available_encoder_ids()

    def validate_for_family(self, identifier: str, family: str) -> DinomalyEncoderPreset:
        """Require a curated, installed encoder that matches the selected Dinomaly variant."""
        preset = self.get(identifier)
        if preset.family != family:
            raise ValueError(f"{preset.display_name} is incompatible with the selected {family} Dinomaly model.")
        if not self.is_available(preset):
            raise ValueError(f"The selected {preset.display_name} encoder is unavailable in the installed timm runtime.")
        return preset

    @staticmethod
    @lru_cache(maxsize=1)
    def _available_encoder_ids() -> frozenset[str]:
        try:
            import timm

            return frozenset(preset.identifier for preset in DinomalyEncoderRegistry._PRESETS if timm.is_model(preset.identifier))
        except Exception:
            return frozenset()