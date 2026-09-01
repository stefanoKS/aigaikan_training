"""Tests for training configuration persistence and validation."""

from __future__ import annotations

from app.models.training_config import DeviceMode, TrainingConfig


def test_training_config_round_trip() -> None:
    config = TrainingConfig(
        model_name="dinomaly_dinov3",
        device=DeviceMode.CPU,
        batch_size=4,
        max_epochs=15,
        validation_every_n_epochs=3,
        gradient_clip_val=0.5,
        accumulate_grad_batches=2,
    )
    payload = config.to_dict()
    restored = TrainingConfig.from_dict(payload)
    assert restored.device is DeviceMode.CPU
    assert restored.max_epochs == 15
    assert restored.target_training_steps == 3000
    assert restored.validation_every_n_epochs == 3
    assert restored.gradient_clip_val == 0.5
    assert restored.accumulate_grad_batches == 2
    assert restored.model_name == "dinomaly_dinov3"
    assert restored.dinomaly_decoder_depth == 8
    assert restored.dinomaly_encoder_name == "vit_base_patch16_dinov3.lvd1689m"
    assert payload["model_profile"]["preprocessing"] == "anomalib-native"
    restored.validate()


def test_patchcore_defaults_to_fixed_production_profile() -> None:
    config = TrainingConfig()

    assert config.model_profile()["backbone"] == "wide_resnet50_2"
    assert config.model_profile()["layers"] == ["layer2", "layer3"]
    assert config.batch_size == 8
    assert config.recommended_epochs(training_image_count=250) == 1
    config.validate()


def test_sparse_persisted_config_uses_current_model_profile() -> None:
    config = TrainingConfig.from_dict({})

    assert config.model_profile()["preprocessing"] == "anomalib-native"


def test_dinomaly_calculates_epochs_from_training_image_count() -> None:
    config = TrainingConfig(model_name="dinomaly_dinov2", batch_size=8, max_epochs=1)
    config.apply_model_defaults(training_image_count=80)

    assert config.max_epochs == 300
    assert config.estimated_training_steps(training_image_count=80) == 3000


def test_legacy_dinomaly_dinov3_encoder_migrates_to_its_explicit_variant() -> None:
    config = TrainingConfig.from_dict(
        {"model_name": "Dinomaly", "dinomaly_encoder": "vit_small_patch16_dinov3"}
    )

    assert config.model_name == "dinomaly_dinov3"
    assert config.is_dinomaly_dinov3
    assert config.dinomaly_encoder_name == "vit_base_patch16_dinov3.lvd1689m"


def test_dinomaly_rejects_overrides_of_the_stock_profile() -> None:
    config = TrainingConfig(model_name="dinomaly_dinov2", dinomaly_decoder_depth=12)

    try:
        config.validate()
    except ValueError as exc:
        assert "stock profile" in str(exc)
    else:
        raise AssertionError("Expected validation failure")


def test_invalid_batch_size_raises() -> None:
    config = TrainingConfig(batch_size=0)
    try:
        config.validate()
    except ValueError as exc:
        assert "Batch size" in str(exc)
    else:
        raise AssertionError("Expected validation failure")

