"""Anomalib integration helpers with lazy imports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.models.training_config import DeviceMode, TrainingConfig


@dataclass(slots=True)
class AnomalibApiInfo:
    """Describes the discovered Anomalib API surface."""

    available: bool
    version: str
    patchcore_import: str
    folder_datamodule_import: str
    engine_import: str
    notes: str = ""


class AnomalibService:
    """Wrap installed Anomalib APIs behind a stable interface."""

    def inspect_api(self) -> AnomalibApiInfo:
        """Inspect the installed Anomalib API."""
        try:
            import anomalib
            from anomalib.data import Folder  # noqa: F401
            from anomalib.engine import Engine  # noqa: F401
            from anomalib.models import Patchcore  # noqa: F401
        except Exception as exc:
            return AnomalibApiInfo(
                available=False,
                version="not-installed",
                patchcore_import="anomalib.models.Patchcore",
                folder_datamodule_import="anomalib.data.Folder",
                engine_import="anomalib.engine.Engine",
                notes=str(exc),
            )
        return AnomalibApiInfo(
            available=True,
            version=str(getattr(anomalib, "__version__", "unknown")),
            patchcore_import="anomalib.models.Patchcore",
            folder_datamodule_import="anomalib.data.Folder",
            engine_import="anomalib.engine.Engine",
        )

    def resolve_device(self, requested: DeviceMode) -> str:
        """Resolve the effective training device."""
        if requested is DeviceMode.CPU:
            return "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def create_patchcore_components(
        self,
        dataset_root: Path,
        ok_train: Path,
        ng_test: Path,
        ok_test: Path,
        masks: Path | None,
        config: TrainingConfig,
    ) -> dict[str, Any]:
        """Instantiate a datamodule, model, and engine for the installed API."""
        from anomalib.data import Folder
        from anomalib.engine import Engine
        from anomalib.models import Patchcore

        model = Patchcore(
            backbone=config.backbone,
            layers=list(config.layers),
            num_neighbors=config.num_neighbors,
            coreset_sampling_ratio=config.coreset_sampling_ratio,
            pre_trained=True,
        )
        datamodule = Folder(
            name="custom",
            root=str(dataset_root),
            normal_dir=ok_train.name,
            abnormal_dir=ng_test.name,
            normal_test_dir=ok_test.name,
            mask_dir=masks.name if masks else None,
            train_batch_size=config.batch_size,
            eval_batch_size=config.batch_size,
            num_workers=config.num_workers,
        )
        engine = Engine(
            max_epochs=1,
            accelerator=self.resolve_device(config.device),
            devices=1,
            default_root_dir=str(config.resolved_output_dir(dataset_root.parent)),
        )
        return {"model": model, "datamodule": datamodule, "engine": engine}
