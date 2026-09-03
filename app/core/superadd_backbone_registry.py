"""Curated, runtime-validated DINOv3 backbones for SuperADD."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


LEGACY_HUGE_BACKBONE_ID = "vit_huge_plus_patch16_dinov3"


@dataclass(frozen=True, slots=True)
class SuperAddBackbonePreset:
    """One supported SuperADD backbone with a stable persisted timm identifier."""

    identifier: str
    display_name: str
    guidance: str
    patch_size: int = 16


class SuperAddBackboneRegistry:
    """Expose only intentional DINOv3 choices that the installed timm can resolve."""

    _PRESETS = (
        SuperAddBackbonePreset("vit_small_patch16_dinov3.lvd1689m", "DINOv3 Small", "Fastest candidate"),
        SuperAddBackbonePreset("vit_small_plus_patch16_dinov3.lvd1689m", "DINOv3 Small+", "Recommended real-time candidate"),
        SuperAddBackbonePreset("vit_base_patch16_dinov3.lvd1689m", "DINOv3 Base", "Balanced quality/latency candidate"),
        SuperAddBackbonePreset("vit_large_patch16_dinov3.lvd1689m", "DINOv3 Large", "Slow candidate"),
        SuperAddBackbonePreset("vit_huge_plus_patch16_dinov3.lvd1689m", "DINOv3 Huge+", "Current/reference configuration; expected to be very slow"),
    )

    def all(self) -> tuple[SuperAddBackbonePreset, ...]:
        """Return every curated SuperADD backbone in operator-facing order."""
        return self._PRESETS

    def get(self, identifier: str) -> SuperAddBackbonePreset:
        """Return a curated backbone or the narrowly-scoped historic Huge+ identifier."""
        for preset in self._PRESETS:
            if preset.identifier == identifier:
                return preset
        if identifier == LEGACY_HUGE_BACKBONE_ID:
            return SuperAddBackbonePreset(
                LEGACY_HUGE_BACKBONE_ID,
                "DINOv3 Huge+",
                "Historical/current reference configuration; expected to be very slow",
            )
        raise ValueError(f"Unsupported SuperADD backbone: {identifier}")

    def is_available(self, preset: SuperAddBackbonePreset) -> bool:
        """Report whether the pinned timm runtime resolves the selected identifier."""
        return preset.identifier in self._available_backbone_ids()

    def validate(self, identifier: str) -> SuperAddBackbonePreset:
        """Require an intentional, installed SuperADD backbone."""
        preset = self.get(identifier)
        if not self.is_available(preset):
            raise ValueError(f"The selected {preset.display_name} backbone is unavailable in the installed timm runtime.")
        return preset

    @staticmethod
    @lru_cache(maxsize=1)
    def _available_backbone_ids() -> frozenset[str]:
        try:
            import timm

            identifiers = [preset.identifier for preset in SuperAddBackboneRegistry._PRESETS]
            identifiers.append(LEGACY_HUGE_BACKBONE_ID)
            return frozenset(identifier for identifier in identifiers if timm.is_model(identifier))
        except Exception:
            return frozenset()