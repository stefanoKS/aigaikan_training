"""Tests for training configuration persistence and validation."""

from __future__ import annotations

from app.models.training_config import DeviceMode, TrainingConfig


def test_training_config_round_trip() -> None:
    config = TrainingConfig(device=DeviceMode.CPU, image_width=512, image_height=320, batch_size=4)
    payload = config.to_dict()
    restored = TrainingConfig.from_dict(payload)
    assert restored.device is DeviceMode.CPU
    assert restored.image_width == 512
    restored.validate()


def test_invalid_batch_size_raises() -> None:
    config = TrainingConfig(batch_size=0)
    try:
        config.validate()
    except ValueError as exc:
        assert "Batch size" in str(exc)
    else:
        raise AssertionError("Expected validation failure")

