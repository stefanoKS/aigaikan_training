"""Tests for the lazy Anomalib integration boundary."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

from app.core.model_registry import ModelRegistry
from app.models.dataset_config import DatasetConfig, DatasetRole
from app.models.inspection_region import InspectionRegionConfig
from app.models.training_config import DeviceMode, TrainingConfig
from app.services.anomalib_service import AnomalibService


def test_dinomaly_uses_selected_folders_and_omits_missing_masks(tmp_path: Path, monkeypatch) -> None:
    """Referenced folders must work without requiring a mask directory."""
    folder_calls: dict[str, object] = {}

    class FakeFolder:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            folder_calls.update(kwargs)

    class FakeEngine:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class FakePatchcore:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        @staticmethod
        def configure_pre_processor(**kwargs) -> dict[str, object]:
            return kwargs

    class FakeDinomaly:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        @staticmethod
        def configure_pre_processor(**kwargs) -> dict[str, object]:
            return kwargs

    anomalib_module = ModuleType("anomalib")
    anomalib_data = ModuleType("anomalib.data")
    anomalib_engine = ModuleType("anomalib.engine")
    anomalib_models = ModuleType("anomalib.models")
    anomalib_data.Folder = FakeFolder
    anomalib_engine.Engine = FakeEngine
    anomalib_models.Patchcore = FakePatchcore
    anomalib_models.Dinomaly = FakeDinomaly
    monkeypatch.setitem(sys.modules, "anomalib", anomalib_module)
    monkeypatch.setitem(sys.modules, "anomalib.data", anomalib_data)
    monkeypatch.setitem(sys.modules, "anomalib.engine", anomalib_engine)
    monkeypatch.setitem(sys.modules, "anomalib.models", anomalib_models)

    ok_folder = tmp_path / "ok"
    ng_folder = tmp_path / "ng"
    ok_folder.mkdir()
    ng_folder.mkdir()
    dataset = DatasetConfig()
    dataset.folders[DatasetRole.OK_TRAIN].path = str(ok_folder)
    dataset.folders[DatasetRole.NG_TEST].path = str(ng_folder)
    dataset.folders[DatasetRole.MASKS].path = str(tmp_path / "missing_masks")
    config = TrainingConfig(model_name="dinomaly_dinov2", device=DeviceMode.CPU)

    components = AnomalibService().create_components(dataset, config)

    assert isinstance(components["model"], FakeDinomaly)
    assert folder_calls["normal_dir"] == ok_folder.resolve()
    assert folder_calls["abnormal_dir"] == ng_folder.resolve()
    assert folder_calls["mask_dir"] is None
    assert folder_calls["test_split_mode"] == "from_dir"
    assert folder_calls["val_split_mode"] == "none"


def test_padim_uses_the_fixed_stock_profile(tmp_path: Path, monkeypatch) -> None:
    """PaDiM must use the fixed supported profile and native preprocessing."""
    class FakeFolder:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class FakeEngine:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class FakePadim:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    anomalib_data = ModuleType("anomalib.data")
    anomalib_engine = ModuleType("anomalib.engine")
    anomalib_models = ModuleType("anomalib.models")
    anomalib_data.Folder = FakeFolder
    anomalib_engine.Engine = FakeEngine
    anomalib_models.Padim = FakePadim
    monkeypatch.setitem(sys.modules, "anomalib.data", anomalib_data)
    monkeypatch.setitem(sys.modules, "anomalib.engine", anomalib_engine)
    monkeypatch.setitem(sys.modules, "anomalib.models", anomalib_models)

    ok_folder = tmp_path / "ok"
    ng_folder = tmp_path / "ng"
    ok_folder.mkdir()
    ng_folder.mkdir()
    dataset = DatasetConfig()
    dataset.folders[DatasetRole.OK_TRAIN].path = str(ok_folder)
    dataset.folders[DatasetRole.NG_TEST].path = str(ng_folder)

    components = AnomalibService().create_components(
        dataset,
        TrainingConfig(model_name="padim", device=DeviceMode.CPU),
        run_directory=tmp_path / "run",
        callbacks=["progress-callback"],
    )

    assert isinstance(components["model"], FakePadim)
    assert components["engine"].kwargs["default_root_dir"] == str(tmp_path / "run")
    assert components["engine"].kwargs["callbacks"] == ["progress-callback"]
    assert components["engine"].kwargs == {
        "accelerator": "cpu",
        "devices": 1,
        "default_root_dir": str(tmp_path / "run"),
        "enable_progress_bar": False,
        "max_epochs": 1,
        "callbacks": ["progress-callback"],
    }
    assert components["model"].kwargs == {
        "backbone": "resnet18",
        "layers": ["layer1", "layer2", "layer3"],
        "pre_trained": True,
    }


def test_patchcore_uses_fixed_profile_and_native_preprocessing(tmp_path: Path, monkeypatch) -> None:
    """PatchCore must use its supported profile without an app preprocessor override."""
    class FakeFolder:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class FakeEngine:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class FakePatchcore:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    anomalib_data = ModuleType("anomalib.data")
    anomalib_engine = ModuleType("anomalib.engine")
    anomalib_models = ModuleType("anomalib.models")
    anomalib_data.Folder = FakeFolder
    anomalib_engine.Engine = FakeEngine
    anomalib_models.Patchcore = FakePatchcore
    monkeypatch.setitem(sys.modules, "anomalib.data", anomalib_data)
    monkeypatch.setitem(sys.modules, "anomalib.engine", anomalib_engine)
    monkeypatch.setitem(sys.modules, "anomalib.models", anomalib_models)

    ok_folder = tmp_path / "ok"
    ng_folder = tmp_path / "ng"
    ok_folder.mkdir()
    ng_folder.mkdir()
    dataset = DatasetConfig()
    dataset.folders[DatasetRole.OK_TRAIN].path = str(ok_folder)
    dataset.folders[DatasetRole.NG_TEST].path = str(ng_folder)
    config = TrainingConfig(model_name="patchcore", device=DeviceMode.CPU)

    components = AnomalibService().create_components(dataset, config)

    assert components["model"].kwargs == {
        "backbone": "wide_resnet50_2",
        "layers": ["layer2", "layer3"],
        "num_neighbors": 9,
        "coreset_sampling_ratio": 0.1,
        "pre_trained": True,
    }


@pytest.mark.parametrize(
    ("model_name", "encoder_name"),
    (
        ("dinomaly_dinov2", "vit_base_patch14_reg4_dinov2"),
        ("dinomaly_dinov3", "vit_base_patch16_dinov3.lvd1689m"),
    ),
)
def test_dinomaly_variants_use_stock_class_and_explicit_encoder(
    model_name: str,
    encoder_name: str,
    monkeypatch,
) -> None:
    class FakeDinomaly:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        @staticmethod
        def configure_pre_processor(**kwargs) -> dict[str, object]:
            return kwargs

    anomalib_models = ModuleType("anomalib.models")
    anomalib_models.Dinomaly = FakeDinomaly
    monkeypatch.setitem(sys.modules, "anomalib.models", anomalib_models)

    model = AnomalibService()._create_model(
        ModelRegistry().get(model_name),
        TrainingConfig(model_name=model_name, device=DeviceMode.CPU),
    )

    assert isinstance(model, FakeDinomaly)
    assert model.kwargs == {
        "encoder_name": encoder_name,
        "decoder_depth": 8,
        "bottleneck_dropout": 0.2,
        "use_context_recentering": False,
    }


def test_dinomaly_preserves_native_trainer_arguments_except_max_steps(tmp_path: Path, monkeypatch) -> None:
    class FakeFolder:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class FakeEngine:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class FakeDinomaly:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    anomalib_data = ModuleType("anomalib.data")
    anomalib_engine = ModuleType("anomalib.engine")
    anomalib_models = ModuleType("anomalib.models")
    anomalib_data.Folder = FakeFolder
    anomalib_engine.Engine = FakeEngine
    anomalib_models.Dinomaly = FakeDinomaly
    monkeypatch.setitem(sys.modules, "anomalib.data", anomalib_data)
    monkeypatch.setitem(sys.modules, "anomalib.engine", anomalib_engine)
    monkeypatch.setitem(sys.modules, "anomalib.models", anomalib_models)

    ok_folder = tmp_path / "ok"
    ok_folder.mkdir()
    for index in range(9):
        (ok_folder / f"ok_{index}.png").write_bytes(b"image")
    dataset = DatasetConfig()
    dataset.folders[DatasetRole.OK_TRAIN].path = str(ok_folder)

    components = AnomalibService().create_components(
        dataset,
        TrainingConfig(model_name="dinomaly_dinov2", device=DeviceMode.CPU),
        run_directory=tmp_path / "run",
        callbacks=["progress-callback"],
    )

    assert components["engine"].kwargs == {
        "accelerator": "cpu",
        "devices": 1,
        "default_root_dir": str(tmp_path / "run"),
        "enable_progress_bar": False,
        "max_steps": 5000,
        "callbacks": ["progress-callback"],
    }


def test_inference_components_disable_console_progress(tmp_path: Path, monkeypatch) -> None:
    class FakeEngine:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class FakePatchcore:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    anomalib_engine = ModuleType("anomalib.engine")
    anomalib_models = ModuleType("anomalib.models")
    anomalib_engine.Engine = FakeEngine
    anomalib_models.Patchcore = FakePatchcore
    monkeypatch.setitem(sys.modules, "anomalib.engine", anomalib_engine)
    monkeypatch.setitem(sys.modules, "anomalib.models", anomalib_models)

    components = AnomalibService().create_inference_components(
        TrainingConfig(model_name="patchcore", device=DeviceMode.CPU),
        tmp_path / "inference",
    )

    assert components["engine"].kwargs == {
        "accelerator": "cpu",
        "devices": 1,
        "default_root_dir": str(tmp_path / "inference"),
        "enable_progress_bar": False,
    }


def test_calibration_datamodule_never_splits_the_final_test_subset(tmp_path: Path, monkeypatch) -> None:
    """Calibration reuses its own held-out snapshot rather than splitting it again."""
    class FakeFolder:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    anomalib_data = ModuleType("anomalib.data")
    anomalib_data.Folder = FakeFolder
    monkeypatch.setitem(sys.modules, "anomalib.data", anomalib_data)
    for folder_name in ("ok_train", "ok_test", "ng_test"):
        (tmp_path / folder_name).mkdir()
    dataset = DatasetConfig()
    dataset.folders[DatasetRole.OK_TRAIN].path = str(tmp_path / "ok_train")
    dataset.folders[DatasetRole.OK_TEST].path = str(tmp_path / "ok_test")
    dataset.folders[DatasetRole.NG_TEST].path = str(tmp_path / "ng_test")

    datamodule = AnomalibService().create_datamodule(
        dataset,
        TrainingConfig(device=DeviceMode.CPU),
        calibration_mode=True,
    )

    assert datamodule.kwargs["test_split_mode"] == "from_dir"
    assert datamodule.kwargs["val_split_mode"] == "same_as_test"


def test_enabled_inspection_roi_is_applied_to_every_datamodule_stage(tmp_path: Path, monkeypatch) -> None:
    class FakeFolder:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    anomalib_data = ModuleType("anomalib.data")
    anomalib_data.Folder = FakeFolder
    monkeypatch.setitem(sys.modules, "anomalib.data", anomalib_data)
    for folder_name in ("ok_train", "ok_test"):
        (tmp_path / folder_name).mkdir()
    dataset = DatasetConfig()
    dataset.folders[DatasetRole.OK_TRAIN].path = str(tmp_path / "ok_train")
    dataset.folders[DatasetRole.OK_TEST].path = str(tmp_path / "ok_test")
    roi = InspectionRegionConfig(
        enabled=True,
        source_width=64,
        source_height=64,
        points_px=((4, 4), (59, 4), (59, 59), (4, 59)),
    )

    datamodule = AnomalibService().create_datamodule(
        dataset,
        TrainingConfig(device=DeviceMode.CPU),
        calibration_mode=True,
        inspection_region=roi,
    )

    transform = datamodule.kwargs["augmentations"]
    assert transform.config == roi


def test_retired_models_are_rejected_before_training() -> None:
    with pytest.raises(ValueError, match="Unsupported production model"):
        TrainingConfig(model_name="superadd_dinov3").validate()


def test_auto_device_falls_back_when_pytorch_lacks_the_gpu_architecture(monkeypatch) -> None:
    """Auto must not choose a CUDA device that the installed PyTorch cannot execute on."""
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_capability(_device: int) -> tuple[int, int]:
            return 12, 0

        @staticmethod
        def get_arch_list() -> list[str]:
            return ["sm_80", "sm_90"]

        @staticmethod
        def get_device_name(_device: int) -> str:
            return "Test GPU"

    fake_torch = ModuleType("torch")
    fake_torch.cuda = FakeCuda()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    service = AnomalibService()

    assert service.resolve_device(DeviceMode.AUTO) == "cpu"
    with pytest.raises(RuntimeError, match="sm_120"):
        service.resolve_device(DeviceMode.CUDA)


def test_inspect_api_rejects_anomalib_versions_other_than_2_6_0(monkeypatch) -> None:
    anomalib_module = ModuleType("anomalib")
    anomalib_module.__version__ = "2.5.1"
    anomalib_data = ModuleType("anomalib.data")
    anomalib_data.Folder = object
    anomalib_engine = ModuleType("anomalib.engine")
    anomalib_engine.Engine = object
    anomalib_models = ModuleType("anomalib.models")
    monkeypatch.setitem(sys.modules, "anomalib", anomalib_module)
    monkeypatch.setitem(sys.modules, "anomalib.data", anomalib_data)
    monkeypatch.setitem(sys.modules, "anomalib.engine", anomalib_engine)
    monkeypatch.setitem(sys.modules, "anomalib.models", anomalib_models)

    info = AnomalibService().inspect_api()

    assert not info.available
    assert "2.6.0 is required" in info.notes