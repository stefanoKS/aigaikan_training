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
    assert restored.target_training_steps is None
    assert restored.validation_every_n_epochs == 3
    assert restored.gradient_clip_val == 0.5
    assert restored.accumulate_grad_batches == 2
    assert restored.model_name == "dinomaly_dinov3"
    assert restored.dinomaly_decoder_depth == 8
    assert restored.dinomaly_encoder_name == "vit_base_patch16_dinov3.lvd1689m"
    assert payload["model_profile"]["preprocessing"] == {
        "resize_size": [448, 448],
        "center_crop_size": [384, 384],
        "encoder_patch_size": 16,
    }
    restored.validate()


def test_pixel_threshold_operating_point_is_independent_and_opt_in() -> None:
    config = TrainingConfig(pixel_threshold_enabled=True, pixel_threshold=2.5)

    restored = TrainingConfig.from_dict(config.to_dict())

    assert restored.pixel_operating_point.active_threshold == 2.5
    assert restored.pixel_operating_point.to_dict()["comparator"] == "greater_than_or_equal"
    assert TrainingConfig.from_dict({}).pixel_operating_point.active_threshold is None


def test_patchcore_defaults_to_fixed_production_profile() -> None:
    config = TrainingConfig()

    assert config.model_profile()["backbone"] == "wide_resnet50_2"
    assert config.model_profile()["layers"] == ["layer2", "layer3"]
    assert config.batch_size == 8
    assert config.recommended_epochs(training_image_count=250) == 1
    config.validate()


def test_padim_uses_the_one_epoch_anomalib_trainer_contract() -> None:
    config = TrainingConfig(model_name="padim")

    assert config.max_epochs == 1
    config.max_epochs = 5
    config.apply_model_defaults(training_image_count=250)
    assert config.max_epochs == 1


def test_sparse_persisted_config_uses_current_model_profile() -> None:
    config = TrainingConfig.from_dict({})

    assert config.model_profile()["preprocessing"] == "anomalib-native"


def test_dinomaly_auto_steps_follow_the_single_class_baseline() -> None:
    config = TrainingConfig(model_name="dinomaly_dinov2", batch_size=8, max_epochs=1)
    config.apply_model_defaults(training_image_count=80)

    assert config.resolved_dinomaly_training_steps(training_image_count=80) == 5000
    assert config.estimated_training_steps(training_image_count=80) == 5000


def test_dinomaly_step_override_remains_available() -> None:
    config = TrainingConfig(model_name="dinomaly_dinov2", target_training_steps=7500)

    assert config.resolved_dinomaly_training_steps(training_image_count=80) == 7500


def test_dinomaly_persists_a_curated_encoder_selection() -> None:
    config = TrainingConfig(
        model_name="dinomaly_dinov2",
        dinomaly_encoder_id="vit_large_patch14_reg4_dinov2",
    )

    restored = TrainingConfig.from_dict(config.to_dict())

    assert restored.dinomaly_encoder_name == "vit_large_patch14_reg4_dinov2"
    restored.validate()


def test_dinomaly_preprocessing_defaults_match_the_selected_encoder_patch_size() -> None:
    assert TrainingConfig(model_name="dinomaly_dinov2").model_profile()["preprocessing"] == "anomalib-native"
    assert TrainingConfig(model_name="dinomaly_dinov3").model_profile()["preprocessing"] == {
        "resize_size": [448, 448],
        "center_crop_size": [384, 384],
        "encoder_patch_size": 16,
    }


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


def test_anomaly_dino_uses_one_memory_bank_collection_pass() -> None:
    config = TrainingConfig(model_name="anomaly_dino", max_epochs=12)

    assert config.max_epochs == 1
    assert config.model_profile()["coreset_subsampling"] is True
    assert config.model_profile()["sampling_ratio"] == 0.1
    config.max_epochs = 2
    try:
        config.validate()
    except ValueError as exc:
        assert "AnomalyDINO" in str(exc)
    else:
        raise AssertionError("Expected validation failure")


def test_efficient_ad_requires_one_image_training_batches() -> None:
    config = TrainingConfig(model_name="efficient_ad", batch_size=12)

    assert config.batch_size == 1
    assert config.model_profile()["batch_size"] == 1
    config.batch_size = 2
    try:
        config.validate()
    except ValueError as exc:
        assert "EfficientAD" in str(exc)
    else:
        raise AssertionError("Expected validation failure")


def test_super_add_uses_its_stock_single_memory_bank_pass() -> None:
    config = TrainingConfig(model_name="super_add", max_epochs=12)

    assert config.max_epochs == 1
    assert config.model_profile()["max_epochs"] == 1
    config.max_epochs = 2
    try:
        config.validate()
    except ValueError as exc:
        assert "SuperADD" in str(exc)
    else:
        raise AssertionError("Expected validation failure")

