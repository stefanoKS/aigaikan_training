"""Tests for training configuration persistence and validation."""

from __future__ import annotations

from app.models.training_config import DeviceMode, TrainingConfig


def test_training_config_round_trip() -> None:
    config = TrainingConfig(
        model_name="Dinomaly",
        device=DeviceMode.CPU,
        image_width=512,
        image_height=320,
        batch_size=4,
        max_epochs=15,
        validation_every_n_epochs=3,
        gradient_clip_val=0.5,
        accumulate_grad_batches=2,
        dinomaly_encoder="vit_small_patch16_dinov3.lvd1689m",
        dinomaly_decoder_depth=12,
        supplemental_data_path="C:/datasets/imagenette",
        zero_shot_class_name="widget",
    )
    payload = config.to_dict()
    restored = TrainingConfig.from_dict(payload)
    assert restored.device is DeviceMode.CPU
    assert restored.image_width == 512
    assert restored.max_epochs == 15
    assert restored.target_training_steps == 3000
    assert restored.validation_every_n_epochs == 3
    assert restored.gradient_clip_val == 0.5
    assert restored.accumulate_grad_batches == 2
    assert restored.model_name == "Dinomaly"
    assert restored.dinomaly_encoder == "vit_small_patch16_dinov3.lvd1689m"
    assert restored.dinomaly_decoder_depth == 12
    assert restored.supplemental_data_path == "C:/datasets/imagenette"
    assert restored.zero_shot_class_name == "widget"
    restored.validate()


def test_patchcore_defaults_to_one_epoch_and_280_pixel_input() -> None:
    config = TrainingConfig()

    assert config.model_input_size == (280, 280)
    assert config.recommended_epochs(training_image_count=250) == 1
    config.validate()


def test_sparse_persisted_config_uses_current_input_size_defaults() -> None:
    config = TrainingConfig.from_dict({})

    assert config.model_input_size == (280, 280)


def test_dinomaly_calculates_epochs_from_training_image_count() -> None:
    config = TrainingConfig(model_name="Dinomaly", batch_size=8, max_epochs=1)
    config.apply_model_defaults(training_image_count=80)

    assert config.max_epochs == 300
    assert config.estimated_training_steps(training_image_count=80) == 3000


def test_invalid_batch_size_raises() -> None:
    config = TrainingConfig(batch_size=0)
    try:
        config.validate()
    except ValueError as exc:
        assert "Batch size" in str(exc)
    else:
        raise AssertionError("Expected validation failure")

