"""Anomalib integration helpers with lazy imports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

from app.core.model_registry import ModelDefinition, ModelExecutionMode, ModelRegistry
from app.models.dataset_config import DatasetConfig, DatasetRole
from app.models.training_config import DeviceMode, TrainingConfig


@dataclass(slots=True)
class AnomalibApiInfo:
    """Describes the discovered Anomalib API surface."""

    available: bool
    version: str
    patchcore_import: str
    dinomaly_import: str
    folder_datamodule_import: str
    engine_import: str
    model_count: int = 0
    notes: str = ""


class AnomalibService:
    """Wrap installed Anomalib APIs behind a stable interface."""

    def __init__(self, model_registry: ModelRegistry | None = None) -> None:
        self.model_registry = model_registry or ModelRegistry()

    def inspect_api(self) -> AnomalibApiInfo:
        """Inspect the installed Anomalib API."""
        try:
            import anomalib
            from anomalib.data import Folder  # noqa: F401
            from anomalib.engine import Engine  # noqa: F401
            import anomalib.models as anomalib_models
        except Exception as exc:
            return AnomalibApiInfo(
                available=False,
                version="not-installed",
                patchcore_import="anomalib.models.Patchcore",
                dinomaly_import="anomalib.models.Dinomaly",
                folder_datamodule_import="anomalib.data.Folder",
                engine_import="anomalib.engine.Engine",
                notes=str(exc),
            )
        missing_models = [
            definition.anomalib_class_name
            for definition in self.model_registry.official_anomalib_models()
            if not hasattr(anomalib_models, definition.anomalib_class_name)
        ]
        if missing_models:
            return AnomalibApiInfo(
                available=False,
                version=str(getattr(anomalib, "__version__", "unknown")),
                patchcore_import="anomalib.models.Patchcore",
                dinomaly_import="anomalib.models.Dinomaly",
                folder_datamodule_import="anomalib.data.Folder",
                engine_import="anomalib.engine.Engine",
                notes=f"Installed Anomalib is missing: {', '.join(missing_models)}",
            )
        return AnomalibApiInfo(
            available=True,
            version=str(getattr(anomalib, "__version__", "unknown")),
            patchcore_import="anomalib.models.Patchcore",
            dinomaly_import="anomalib.models.Dinomaly",
            folder_datamodule_import="anomalib.data.Folder",
            engine_import="anomalib.engine.Engine",
            model_count=len(self.model_registry.all()),
        )

    def resolve_device(self, requested: DeviceMode) -> str:
        """Resolve the effective training device."""
        if requested is DeviceMode.CPU:
            return "cpu"
        try:
            import torch

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                if torch.cuda.is_available() and self._cuda_architecture_is_supported(torch):
                    return "gpu"
                if requested is DeviceMode.CUDA:
                    raise RuntimeError(self._cuda_support_message(torch))
        except RuntimeError:
            raise
        except Exception:
            if requested is DeviceMode.CUDA:
                raise RuntimeError("CUDA was selected, but PyTorch could not initialize a supported CUDA device.")
        return "cpu"

    @staticmethod
    def _cuda_architecture_is_supported(torch: Any) -> bool:
        """Return whether the installed PyTorch build includes the active GPU architecture."""
        try:
            major, minor = torch.cuda.get_device_capability(0)
            return f"sm_{major}{minor}" in torch.cuda.get_arch_list()
        except Exception:
            return False

    @staticmethod
    def _cuda_support_message(torch: Any) -> str:
        """Describe why an explicitly selected CUDA device cannot run training."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                major, minor = torch.cuda.get_device_capability(0)
                device_name = torch.cuda.get_device_name(0)
                supported_architectures = ", ".join(torch.cuda.get_arch_list())
            return (
                f"{device_name} requires CUDA architecture sm_{major}{minor}, but the installed PyTorch build supports "
                f"{supported_architectures}. Install a newer PyTorch CUDA build or select CPU."
            )
        except Exception:
            return "CUDA was selected, but the installed PyTorch build does not support the active GPU."

    def create_components(
        self,
        dataset: DatasetConfig,
        config: TrainingConfig,
        run_directory: Path | None = None,
        callbacks: list[Any] | None = None,
        calibration_mode: bool = False,
    ) -> dict[str, Any]:
        """Instantiate current Anomalib components from the project configuration."""
        config.validate()
        definition = self.model_registry.get(config.model_name)
        if not definition.supports_image_folder:
            raise ValueError(
                f"{definition.display_name} requires a video dataset and cannot run in an image-folder project."
            )
        from anomalib.data import Folder
        from anomalib.engine import Engine

        ok_train = self._required_folder(dataset, DatasetRole.OK_TRAIN)
        model = self._create_model(definition, config)
        device = self.resolve_device(config.device)
        if device == "gpu":
            self._configure_gpu_precision()

        datamodule = self.create_datamodule(dataset, config, calibration_mode=calibration_mode)
        engine_kwargs: dict[str, Any] = {
            "max_epochs": config.max_epochs,
            "check_val_every_n_epoch": config.validation_every_n_epochs,
            "gradient_clip_val": config.gradient_clip_val,
            "accumulate_grad_batches": config.accumulate_grad_batches,
            "accelerator": device,
            "devices": 1,
            "default_root_dir": str(run_directory or config.resolved_output_dir(ok_train.parent)),
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "logger": False,
        }
        if config.is_dinomaly:
            engine_kwargs["max_steps"] = config.target_training_steps
        if callbacks:
            engine_kwargs["callbacks"] = callbacks
        engine = Engine(
            **engine_kwargs,
        )
        return {
            "model": model,
            "datamodule": datamodule,
            "engine": engine,
            "definition": definition,
            "device": device,
            "device_note": self._device_note(config.device, device),
        }

    def create_datamodule(
        self,
        dataset: DatasetConfig,
        config: TrainingConfig,
        *,
        calibration_mode: bool,
    ) -> Any:
        """Create an explicit Folder datamodule without Anomalib-side random splitting."""
        from anomalib.data import Folder

        ok_train = self._required_folder(dataset, DatasetRole.OK_TRAIN)
        ng_test = self._optional_folder(dataset, DatasetRole.NG_TEST)
        ok_test = self._optional_folder(dataset, DatasetRole.OK_TEST)
        masks = self._optional_folder(dataset, DatasetRole.MASKS)
        return Folder(
            name="custom",
            normal_dir=ok_train,
            abnormal_dir=ng_test,
            normal_test_dir=ok_test,
            mask_dir=masks,
            train_batch_size=config.batch_size,
            eval_batch_size=config.batch_size,
            num_workers=config.num_workers,
            test_split_mode="from_dir",
            val_split_mode="same_as_test" if calibration_mode else "none",
            seed=config.split_seed,
        )

    def create_inference_components(self, config: TrainingConfig, output_directory: Path) -> dict[str, Any]:
        """Create a model and engine for prediction from a saved checkpoint."""
        config.validate()
        definition = self.model_registry.get(config.model_name)
        if not definition.supports_image_folder:
            raise ValueError(
                f"{definition.display_name} requires a video dataset and cannot run on image files."
            )
        from anomalib.engine import Engine

        device = self.resolve_device(config.device)
        if device == "gpu":
            self._configure_gpu_precision()
        engine = Engine(
            accelerator=device,
            devices=1,
            default_root_dir=str(output_directory),
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )
        return {
            "model": self._create_model(definition, config),
            "engine": engine,
            "definition": definition,
            "device": device,
            "device_note": self._device_note(config.device, device),
        }

    @staticmethod
    def _configure_gpu_precision() -> None:
        """Use Tensor Cores efficiently on the selected CUDA device."""
        try:
            import torch

            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    def _device_note(self, requested: DeviceMode, resolved: str) -> str:
        """Explain when Auto falls back from a detected but unsupported CUDA device."""
        if requested is not DeviceMode.AUTO or resolved != "cpu":
            return ""
        try:
            import torch

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                if torch.cuda.is_available() and not self._cuda_architecture_is_supported(torch):
                    return f"{self._cuda_support_message(torch)} Auto selected CPU."
        except Exception:
            pass
        return ""

    def _create_model(self, definition: ModelDefinition, config: TrainingConfig) -> Any:
        """Instantiate a model with its supported project-level options."""
        if definition.key == "dinomaly_dinov3":
            from app.services.dinomaly_dinov3_adapter import create_dinomaly_dinov3_model

            return create_dinomaly_dinov3_model(config)
        if definition.anomalib_class_name is None:
            raise RuntimeError(f"No model factory is registered for {definition.display_name}.")
        import anomalib.models as anomalib_models

        model_class = getattr(anomalib_models, definition.anomalib_class_name, None)
        if model_class is None:
            raise RuntimeError(
                f"Installed Anomalib does not export {definition.anomalib_class_name}; rerun setup."
            )
        if definition.key == "patchcore":
            return model_class(
                backbone=config.backbone,
                layers=list(config.layers),
                num_neighbors=config.num_neighbors,
                coreset_sampling_ratio=config.coreset_sampling_ratio,
                pre_trained=True,
                pre_processor=model_class.configure_pre_processor(image_size=config.model_input_size),
            )
        if definition.key == "dinomaly_dinov2":
            image_size = config.model_input_size
            return model_class(
                encoder_name=config.dinomaly_encoder,
                decoder_depth=config.dinomaly_decoder_depth,
                bottleneck_dropout=config.dinomaly_bottleneck_dropout,
                use_context_recentering=config.dinomaly_context_recentering,
                pre_processor=model_class.configure_pre_processor(
                    image_size=image_size,
                    crop_size=min(image_size),
                ),
            )
        if definition.key == "superadd_dinov3":
            return model_class(
                backbone=config.superadd_encoder,
                layers=list(config.dinov3_feature_layers) or None,
                patch_size=config.superadd_patch_size,
                patch_overlap=config.superadd_patch_overlap,
                pre_processor=model_class.configure_pre_processor(image_size=config.model_input_size),
            )

        model_kwargs = self._model_kwargs(definition, config)
        return model_class(**model_kwargs)

    def _model_kwargs(self, definition: ModelDefinition, config: TrainingConfig) -> dict[str, Any]:
        """Build optional model-specific arguments from project configuration."""
        if definition.key == "draem":
            return {"dtd_dir": self._required_supplemental_path(config, definition, expect_file=False)}
        if definition.key == "efficientad":
            return {"imagenet_dir": self._required_supplemental_path(config, definition, expect_file=False)}
        if definition.key == "cfm":
            return {"pointmae_weights": self._required_supplemental_path(config, definition, expect_file=True)}
        if definition.key == "glass" and config.supplemental_data_path:
            return {"anomaly_source_path": self._existing_supplemental_path(config, definition)}
        if definition.key == "winclip" and config.zero_shot_class_name.strip():
            return {"class_name": config.zero_shot_class_name.strip()}
        return {}

    @staticmethod
    def _required_supplemental_path(
        config: TrainingConfig,
        definition: ModelDefinition,
        expect_file: bool,
    ) -> Path:
        path = AnomalibService._existing_supplemental_path(config, definition)
        if expect_file and not path.is_file():
            raise ValueError(f"{definition.display_name} requires a supplemental weights file.")
        if not expect_file and not path.is_dir():
            raise ValueError(f"{definition.display_name} requires a supplemental data folder.")
        return path

    @staticmethod
    def _existing_supplemental_path(config: TrainingConfig, definition: ModelDefinition) -> Path:
        if not config.supplemental_data_path.strip():
            raise ValueError(
                f"{definition.display_name} requires Supplemental Model Data. {definition.requirement}"
            )
        path = Path(config.supplemental_data_path).expanduser()
        if not path.exists():
            raise ValueError(f"Supplemental Model Data does not exist: {path}")
        return path.resolve()

    def _required_folder(self, dataset: DatasetConfig, role: DatasetRole) -> Path:
        path = self._optional_folder(dataset, role)
        if path is None:
            raise ValueError(f"The {role.value} folder is required and must exist")
        return path

    @staticmethod
    def _optional_folder(dataset: DatasetConfig, role: DatasetRole) -> Path | None:
        path = dataset.folders[role].resolved_path()
        if path is None or not path.is_dir():
            return None
        return path.resolve()
