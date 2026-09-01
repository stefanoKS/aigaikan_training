"""Unit tests for DINOv3 adapter contract inspection without loading PyTorch."""

from types import SimpleNamespace

import pytest

from app.services.dinomaly_dinov3_adapter import DinoV3AdapterError, inspect_dinov3_encoder


def _encoder(*, patch_size: tuple[int, int] = (16, 16), depth: int = 12) -> SimpleNamespace:
    return SimpleNamespace(
        patch_embed=SimpleNamespace(patch_size=patch_size),
        embed_dim=384,
        blocks=[SimpleNamespace(attn=SimpleNamespace(num_heads=6)) for _ in range(depth)],
        forward_intermediates=lambda *args, **kwargs: [],
        pretrained_cfg={"input_size": (3, 512, 512), "mean": (0.4, 0.5, 0.6), "std": (0.2, 0.3, 0.4)},
    )


def test_encoder_metadata_is_derived_from_runtime_properties() -> None:
    metadata = inspect_dinov3_encoder(_encoder(), "vit_small_patch16_dinov3.lvd1689m")

    assert metadata.patch_size == 16
    assert metadata.embedding_dim == 384
    assert metadata.depth == 12
    assert metadata.num_heads == 6
    assert metadata.feature_layers == (0, 4, 7, 11)
    assert metadata.normalization_mean == (0.4, 0.5, 0.6)


def test_explicit_feature_layers_are_checked_against_runtime_depth() -> None:
    metadata = inspect_dinov3_encoder(_encoder(depth=5), "vit_small_patch16_dinov3.lvd1689m", (1, 4))

    assert metadata.feature_layers == (1, 4)
    with pytest.raises(DinoV3AdapterError, match="between 0 and 4"):
        inspect_dinov3_encoder(_encoder(depth=5), "vit_small_patch16_dinov3.lvd1689m", (5,))


def test_adapter_rejects_incompatible_runtime_patch_geometry() -> None:
    with pytest.raises(DinoV3AdapterError, match="square"):
        inspect_dinov3_encoder(_encoder(patch_size=(16, 14)), "vit_small_patch16_dinov3.lvd1689m")