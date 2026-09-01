"""Tests for the lazy Anomalib integration boundary."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

from app.core.model_registry import ModelRegistry
from app.models.dataset_config import DatasetConfig, DatasetRole
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
    config = TrainingConfig(model_name="Dinomaly", device=DeviceMode.CPU)

    components = AnomalibService().create_components(dataset, config)

    assert isinstance(components["model"], FakeDinomaly)
    assert folder_calls["normal_dir"] == ok_folder.resolve()
    assert folder_calls["abnormal_dir"] == ng_folder.resolve()
    assert folder_calls["mask_dir"] is None
    assert folder_calls["test_split_mode"] == "from_dir"
    assert folder_calls["val_split_mode"] == "none"


def test_generic_image_model_uses_the_catalog_class(tmp_path: Path, monkeypatch) -> None:
    """An image-folder model outside the custom configuration paths is supported."""
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
    assert components["engine"].kwargs["max_epochs"] == 1
    assert components["engine"].kwargs["check_val_every_n_epoch"] == 1


def test_patchcore_uses_anomalib_owned_model_input_preprocessing(tmp_path: Path, monkeypatch) -> None:
    """The UI model-input dimensions must configure PatchCore's preprocessor once."""
    class FakeFolder:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class FakeEngine:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class FakePatchcore:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        @staticmethod
        def configure_pre_processor(**kwargs) -> dict[str, object]:
            return kwargs

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
    config = TrainingConfig(model_name="patchcore", image_width=280, image_height=280, device=DeviceMode.CPU)

    components = AnomalibService().create_components(dataset, config)

    assert components["model"].kwargs["pre_processor"] == {"image_size": (280, 280)}


def test_dinomaly_dinov3_dispatches_to_the_application_adapter(monkeypatch) -> None:
    import app.services.dinomaly_dinov3_adapter as adapter

    sentinel = object()
    monkeypatch.setattr(adapter, "create_dinomaly_dinov3_model", lambda _config: sentinel)

    model = AnomalibService()._create_model(
        ModelRegistry().get("dinomaly_dinov3"),
        TrainingConfig(model_name="dinomaly_dinov3", device=DeviceMode.CPU),
    )

    assert model is sentinel


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


def test_video_model_is_rejected_for_image_folder_projects() -> None:
    """Video-only models need a future video-project workflow rather than Folder data."""
    try:
        AnomalibService().create_components(DatasetConfig(), TrainingConfig(model_name="aivad"))
    except ValueError as exc:
        assert "video dataset" in str(exc)
    else:
        raise AssertionError("Expected the video model to be rejected")


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


@pytest.mark.parametrize("model_name", ("draem", "efficientad", "cfm"))
def test_required_model_resources_are_reported_before_training(model_name: str) -> None:
    """Models with external assets explain the missing configuration clearly."""
    definition = ModelRegistry().get(model_name)

    with pytest.raises(ValueError, match="requires Supplemental Model Data"):
        AnomalibService()._model_kwargs(definition, TrainingConfig(model_name=model_name))