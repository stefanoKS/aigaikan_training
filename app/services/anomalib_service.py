"""Anomalib integration helpers with lazy imports."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

from app.core.model_registry import ModelDefinition, ModelExecutionMode, ModelRegistry
from app.models.dataset_config import DatasetConfig, DatasetRole, SUPPORTED_IMAGE_EXTENSIONS
from app.models.inspection_region import InspectionRegionConfig
from app.models.preprocessing_config import ResolvedPreprocessingPlan
from app.models.training_config import DeviceMode, TrainingConfig

REQUIRED_ANOMALIB_VERSION = "2.6.0"
DINOMALY_DINOV3_RESIZE_SIZE = (448, 448)
DINOMALY_DINOV3_CROP_SIZE = 384


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


@dataclass(frozen=True, slots=True)
class _ModelAdapter:
    """One verified Anomalib constructor behind an application-owned model ID."""

    factory: Callable[[Any, ModelDefinition, TrainingConfig, ResolvedPreprocessingPlan | None], Any]


class AnomalibService:
    """Wrap installed Anomalib APIs behind a stable interface."""

    def __init__(self, model_registry: ModelRegistry | None = None) -> None:
        self.model_registry = model_registry or ModelRegistry()
        self._model_adapters = {
            "patchcore": _ModelAdapter(self._create_patchcore_model),
            "padim": _ModelAdapter(self._create_padim_model),
            "dinomaly_dinov2": _ModelAdapter(self._create_dinomaly_model),
            "dinomaly_dinov3": _ModelAdapter(self._create_dinomaly_model),
            "anomaly_dino": _ModelAdapter(self._create_anomaly_dino_model),
            "super_add": _ModelAdapter(self._create_super_add_model),
            "efficient_ad": _ModelAdapter(self._create_efficient_ad_model),
            "supersimplenet": _ModelAdapter(self._create_supersimplenet_model),
        }

    def inspect_api(self) -> AnomalibApiInfo:
        """Inspect the installed Anomalib API."""
        try:
            import anomalib
            from anomalib.data import Folder  # noqa: F401
            self._anomalib_engine()
            anomalib_models = self._anomalib_models()
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
        version = str(getattr(anomalib, "__version__", "unknown"))
        if version != REQUIRED_ANOMALIB_VERSION:
            return AnomalibApiInfo(
                available=False,
                version=version,
                patchcore_import="anomalib.models.Patchcore",
                dinomaly_import="anomalib.models.Dinomaly",
                folder_datamodule_import="anomalib.data.Folder",
                engine_import="anomalib.engine.Engine",
                notes=f"Anomalib {REQUIRED_ANOMALIB_VERSION} is required; found {version}.",
            )
        missing_models = [
            definition.anomalib_class_name
            for definition in self.model_registry.official_anomalib_models()
            if not hasattr(anomalib_models, definition.anomalib_class_name)
        ]
        if missing_models:
            return AnomalibApiInfo(
                available=False,
                version=version,
                patchcore_import="anomalib.models.Patchcore",
                dinomaly_import="anomalib.models.Dinomaly",
                folder_datamodule_import="anomalib.data.Folder",
                engine_import="anomalib.engine.Engine",
                notes=f"Installed Anomalib is missing: {', '.join(missing_models)}",
            )
        return AnomalibApiInfo(
            available=True,
            version=version,
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
        inspection_region: InspectionRegionConfig | None = None,
        preprocessing_plan: ResolvedPreprocessingPlan | None = None,
    ) -> dict[str, Any]:
        """Instantiate current Anomalib components from the project configuration."""
        ok_train = self._required_folder(dataset, DatasetRole.OK_TRAIN)
        training_image_count = sum(
            path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            for path in ok_train.rglob("*")
        )
        config.apply_model_defaults(training_image_count)
        config.validate()
        definition = self.model_registry.get(config.model_name)
        if not definition.supports_image_folder:
            raise ValueError(
                f"{definition.display_name} requires a video dataset and cannot run in an image-folder project."
            )
        Engine = self._anomalib_engine()
        self._seed_everything(config.random_seed)

        model = self._create_model(definition, config, preprocessing_plan)
        device = self.resolve_device(config.device)
        if device == "gpu":
            self._configure_gpu_precision()

        datamodule = self.create_datamodule(
            dataset,
            config,
            calibration_mode=calibration_mode,
            inspection_region=inspection_region,
            preprocessing_plan=preprocessing_plan,
        )
        engine_kwargs: dict[str, Any] = {
            "accelerator": device,
            "devices": 1,
            "default_root_dir": str(run_directory or config.resolved_output_dir(ok_train.parent)),
            "enable_progress_bar": False,
            "check_val_every_n_epoch": config.validation_every_n_epochs,
            "gradient_clip_val": config.gradient_clip_val,
            "accumulate_grad_batches": config.accumulate_grad_batches,
            "deterministic": True,
        }
        if config.uses_fixed_one_pass:
            engine_kwargs["max_epochs"] = 1
        if config.is_dinomaly:
            engine_kwargs["max_steps"] = config.resolved_dinomaly_training_steps(training_image_count)
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
        inspection_region: InspectionRegionConfig | None = None,
        preprocessing_plan: ResolvedPreprocessingPlan | None = None,
    ) -> Any:
        """Create an explicit Folder datamodule without Anomalib-side random splitting."""
        from anomalib.data import Folder

        ok_train = self._required_folder(dataset, DatasetRole.OK_TRAIN)
        ng_test = self._optional_folder(dataset, DatasetRole.NG_TEST)
        ok_test = self._optional_folder(dataset, DatasetRole.OK_TEST)
        masks = self._optional_folder(dataset, DatasetRole.MASKS)
        inspection_transform = None if preprocessing_plan is not None else self._inspection_transform(inspection_region)
        return Folder(
            name="custom",
            normal_dir=ok_train,
            abnormal_dir=ng_test,
            normal_test_dir=ok_test,
            mask_dir=masks,
            train_batch_size=config.batch_size,
            eval_batch_size=config.batch_size,
            num_workers=config.num_workers,
            augmentations=inspection_transform,
            test_split_mode="from_dir",
            val_split_mode="same_as_test" if calibration_mode else "none",
            seed=config.split_seed,
        )

    @staticmethod
    def _inspection_transform(inspection_region: InspectionRegionConfig | None) -> Any | None:
        """Create the one fixed ROI transform shared by every Anomalib dataset stage."""
        if inspection_region is None or not inspection_region.enabled:
            return None
        from app.core.inspection_region import InspectionRegionProcessor

        return InspectionRegionProcessor(inspection_region)

    def create_inference_components(
        self,
        config: TrainingConfig,
        output_directory: Path,
        preprocessing_plan: ResolvedPreprocessingPlan | None = None,
    ) -> dict[str, Any]:
        """Create a model and engine for prediction from a saved checkpoint."""
        config.validate()
        definition = self.model_registry.get(config.model_name)
        if not definition.supports_image_folder:
            raise ValueError(
                f"{definition.display_name} requires a video dataset and cannot run on image files."
            )
        Engine = self._anomalib_engine()

        device = self.resolve_device(config.device)
        if device == "gpu":
            self._configure_gpu_precision()
        engine = Engine(
            accelerator=device,
            devices=1,
            default_root_dir=str(output_directory),
            enable_progress_bar=False,
        )
        return {
            "model": self._create_model(definition, config, preprocessing_plan),
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

    @staticmethod
    def _seed_everything(seed: int) -> None:
        """Seed every random source used by the application and Lightning data workers."""
        import random

        import numpy as np
        import torch
        from lightning.pytorch import seed_everything

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        seed_everything(seed, workers=True, verbose=False)

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

    def _create_model(
        self,
        definition: ModelDefinition,
        config: TrainingConfig,
        preprocessing_plan: ResolvedPreprocessingPlan | None = None,
    ) -> Any:
        """Instantiate a model with its supported project-level options."""
        anomalib_models = self._anomalib_models()
        model_class = getattr(anomalib_models, definition.anomalib_class_name, None)
        if model_class is None:
            raise RuntimeError(
                f"Installed Anomalib does not export {definition.anomalib_class_name}; rerun setup."
            )
        if preprocessing_plan is not None:
            preprocessing_plan.validate()
            if preprocessing_plan.model_id != definition.key:
                raise ValueError(
                    f"Preprocessing plan is for {preprocessing_plan.model_id}, not selected model {definition.key}."
                )
        try:
            adapter = self._model_adapters[definition.key]
        except KeyError as exc:
            raise RuntimeError(f"No model factory is registered for {definition.display_name}.") from exc
        return adapter.factory(model_class, definition, config, preprocessing_plan)

    def _create_patchcore_model(
        self,
        model_class: Any,
        definition: ModelDefinition,
        _config: TrainingConfig,
        preprocessing_plan: ResolvedPreprocessingPlan | None,
    ) -> Any:
        return self._instantiate_model(
            model_class,
            definition,
            preprocessing_plan,
            {
                "backbone": "wide_resnet50_2",
                "layers": ["layer2", "layer3"],
                "num_neighbors": 9,
                "coreset_sampling_ratio": 0.1,
                "pre_trained": True,
            },
        )

    def _create_padim_model(
        self,
        model_class: Any,
        definition: ModelDefinition,
        _config: TrainingConfig,
        preprocessing_plan: ResolvedPreprocessingPlan | None,
    ) -> Any:
        return self._instantiate_model(
            model_class,
            definition,
            preprocessing_plan,
            {"backbone": "resnet18", "layers": ["layer1", "layer2", "layer3"], "pre_trained": True},
        )

    def _create_dinomaly_model(
        self,
        model_class: Any,
        definition: ModelDefinition,
        config: TrainingConfig,
        preprocessing_plan: ResolvedPreprocessingPlan | None,
    ) -> Any:
        model_kwargs = {
            "encoder_name": config.dinomaly_encoder_name,
            "decoder_depth": 8,
            "bottleneck_dropout": 0.2,
            "use_context_recentering": False,
        }
        if preprocessing_plan is None and config.is_dinomaly_dinov3:
            model_kwargs["pre_processor"] = model_class.configure_pre_processor(
                image_size=DINOMALY_DINOV3_RESIZE_SIZE,
                crop_size=DINOMALY_DINOV3_CROP_SIZE,
            )
        return self._instantiate_model(model_class, definition, preprocessing_plan, model_kwargs)

    def _create_anomaly_dino_model(
        self,
        model_class: Any,
        definition: ModelDefinition,
        _config: TrainingConfig,
        preprocessing_plan: ResolvedPreprocessingPlan | None,
    ) -> Any:
        return self._instantiate_model(
            model_class,
            definition,
            preprocessing_plan,
            {"num_neighbours": 1, "encoder_name": "vit_small_patch14_dinov2", "sampling_ratio": 0.1},
        )

    def _create_super_add_model(
        self,
        model_class: Any,
        definition: ModelDefinition,
        _config: TrainingConfig,
        preprocessing_plan: ResolvedPreprocessingPlan | None,
    ) -> Any:
        return self._instantiate_model(
            model_class,
            definition,
            preprocessing_plan,
            {"backbone": "vit_huge_plus_patch16_dinov3", "patch_size": 448, "patch_overlap": 16},
        )

    def _create_efficient_ad_model(
        self,
        model_class: Any,
        definition: ModelDefinition,
        _config: TrainingConfig,
        preprocessing_plan: ResolvedPreprocessingPlan | None,
    ) -> Any:
        return self._instantiate_model(model_class, definition, preprocessing_plan, {})

    def _create_supersimplenet_model(
        self,
        model_class: Any,
        definition: ModelDefinition,
        _config: TrainingConfig,
        preprocessing_plan: ResolvedPreprocessingPlan | None,
    ) -> Any:
        return self._instantiate_model(
            model_class,
            definition,
            preprocessing_plan,
            {"backbone": "wide_resnet50_2.tv_in1k", "layers": ["layer2", "layer3"]},
        )

    def _instantiate_model(
        self,
        model_class: Any,
        definition: ModelDefinition,
        preprocessing_plan: ResolvedPreprocessingPlan | None,
        model_kwargs: dict[str, Any],
    ) -> Any:
        if preprocessing_plan is not None:
            model_kwargs["pre_processor"] = self._create_v2_pre_processor(
                model_class,
                definition,
                preprocessing_plan,
            )
        return model_class(**model_kwargs)

    @staticmethod
    def _create_v2_pre_processor(
        model_class: Any,
        definition: ModelDefinition,
        preprocessing_plan: ResolvedPreprocessingPlan,
    ) -> Any:
        """Create a rectangular no-crop preprocessor matching a prepared v2 model input."""
        width, height = preprocessing_plan.model_input_size
        image_size = (height, width)
        if definition.key not in {"dinomaly_dinov2", "dinomaly_dinov3", "anomaly_dino"}:
            return model_class.configure_pre_processor(image_size=image_size)
        from anomalib.pre_processing import PreProcessor
        from torchvision.transforms.v2 import Compose, Normalize, Resize

        return PreProcessor(
            transform=Compose(
                [
                    Resize(image_size),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        )

    @staticmethod
    def _anomalib_models() -> Any:
        """Import model classes without surfacing unrelated Dinomaly deprecation notices."""
        with warnings.catch_warnings():
            AnomalibService._filter_known_import_warnings()
            import anomalib.models as anomalib_models
        return anomalib_models

    @staticmethod
    def _anomalib_engine() -> Any:
        """Import the prediction engine without unrelated dependency deprecation notices."""
        with warnings.catch_warnings():
            AnomalibService._filter_known_import_warnings()
            from anomalib.engine import Engine
        return Engine

    @staticmethod
    def _filter_known_import_warnings() -> None:
        """Hide Anomalib 2.6.0 dependency warnings unrelated to the selected model."""
        warnings.filterwarnings(
            "ignore",
            message="Importing from timm.models.layers is deprecated, please import via timm.layers",
            category=FutureWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="The anomalib.models.components.dinov2 package is deprecated.*",
            category=FutureWarning,
        )

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
