"""Training worker entrypoint that communicates through JSON Lines."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from app.core.environment_info import collect_environment_info
from app.core.model_registry import ModelExecutionMode
from app.core.project_manager import ProjectManager
from app.core.result_parser import ResultParser
from app.models.project_config import ProjectConfig
from app.models.training_run import TrainingRun
from app.services.anomalib_service import AnomalibService

LOGGER = logging.getLogger(__name__)

STAGES = [
    "Validating dataset",
    "Preparing datamodule",
    "Loading model",
    "Extracting normal features",
    "Building anomaly model",
    "Evaluating test images",
    "Generating visualizations",
    "Saving results",
]


class TrainingProgressReporter:
    """Report Lightning batch progress through the worker JSON Lines protocol."""

    def __init__(self, emitter: Callable[[dict[str, object]], None]) -> None:
        self._emitter = emitter

    @staticmethod
    def _batch_total(value: object) -> int:
        if isinstance(value, (list, tuple)):
            return max(sum(TrainingProgressReporter._batch_total(item) for item in value), 1)
        try:
            return max(int(value), 1)
        except (TypeError, ValueError, OverflowError):
            return 1

    def _start_stage(self, name: str, total: object) -> None:
        self._emitter({"type": "stage", "name": name})
        self._emitter({"type": "stage_progress", "current": 0, "total": self._batch_total(total)})

    def _update_stage_progress(self, batch_index: int, total: object) -> None:
        self._emitter(
            {
                "type": "stage_progress",
                "current": batch_index + 1,
                "total": self._batch_total(total),
            }
        )

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        self._start_stage("Training model", trainer.num_training_batches)

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._update_stage_progress(batch_idx, trainer.num_training_batches)

    def on_validation_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        self._start_stage("Calibrating model", trainer.num_val_batches)

    def on_validation_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._update_stage_progress(batch_idx, trainer.num_val_batches)

    def on_test_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        self._start_stage("Evaluating test images", trainer.num_test_batches)

    def on_test_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._update_stage_progress(batch_idx, trainer.num_test_batches)


def create_training_progress_callback(emitter: Callable[[dict[str, object]], None]) -> Any:
    """Create a Lightning callback only after the worker is ready to train."""
    from lightning.pytorch.callbacks import Callback

    class LightningTrainingProgressCallback(TrainingProgressReporter, Callback):
        def __init__(self, callback_emitter: Callable[[dict[str, object]], None]) -> None:
            Callback.__init__(self)
            TrainingProgressReporter.__init__(self, callback_emitter)

    return LightningTrainingProgressCallback(emitter)


def emit(message: dict[str, object]) -> None:
    """Emit a JSON line to stdout."""
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def run(project_file: Path) -> int:
    """Run training for the given project."""
    manager = ProjectManager(project_file.parent.parent)
    project = manager.load_project(project_file)
    service = AnomalibService()
    api_info = service.inspect_api()
    if not api_info.available:
        emit(
            {
                "type": "error",
                "message": "Anomalib dependencies are not installed.",
                "details": api_info.notes,
            }
        )
        return 1

    run_dir = manager.create_run_directory(project, project.training.model_name)
    emit({"type": "stage", "name": STAGES[0]})
    emit({"type": "progress", "current": 1, "total": len(STAGES)})
    emit({"type": "log", "level": "info", "message": f"Loaded project {project.name}"})

    environment = collect_environment_info(Path(project.project_path), project.training.random_seed)
    (run_dir / "model").mkdir(parents=True, exist_ok=True)
    (run_dir / "visualizations").mkdir(parents=True, exist_ok=True)
    (run_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(project.training.to_dict(), indent=2), encoding="utf-8")
    result_parser = ResultParser()

    try:
        emit({"type": "stage", "name": STAGES[1]})
        emit({"type": "progress", "current": 2, "total": len(STAGES)})
        progress_callback = create_training_progress_callback(emit)
        components = service.create_components(
            dataset=project.dataset,
            config=project.training,
            run_directory=run_dir,
            callbacks=[progress_callback],
        )
        device_note = str(components["device_note"])
        if device_note:
            emit({"type": "log", "level": "warning", "message": device_note})
        emit({"type": "log", "level": "info", "message": f"Using {components['device']} device"})
        emit({"type": "stage", "name": STAGES[2]})
        emit({"type": "progress", "current": 3, "total": len(STAGES)})
        definition = components["definition"]
        training_duration = 0.0
        if definition.execution_mode is ModelExecutionMode.TRAIN:
            emit({"type": "log", "level": "info", "message": "Starting Anomalib Engine.fit"})
            training_started = perf_counter()
            components["engine"].fit(model=components["model"], datamodule=components["datamodule"])
            training_duration = perf_counter() - training_started
        else:
            emit(
                {
                    "type": "log",
                    "level": "info",
                    "message": f"{definition.display_name} is zero-shot; skipping Engine.fit",
                }
            )
        emit({"type": "stage", "name": STAGES[5]})
        emit({"type": "progress", "current": 6, "total": len(STAGES)})
        evaluation_started = perf_counter()
        metrics = components["engine"].test(model=components["model"], datamodule=components["datamodule"])
        evaluation_duration = perf_counter() - evaluation_started
        metric_payload = metrics[0] if metrics else {}
        run_metrics: dict[str, float | str | None] = {}
        for name, value in metric_payload.items():
            if hasattr(value, "item"):
                value = value.item()
            if isinstance(value, (int, float)):
                run_metrics[result_parser.normalize_metric_name(str(name))] = value
                emit({"type": "metric", "name": name, "value": value})
        result_parser.write_training_run(
            run_dir / "results.json",
            TrainingRun(
                run_name=run_dir.name,
                run_dir=str(run_dir),
                model_name=definition.display_name,
                device=str(components["device"]),
                run_date=str(environment.get("training_date", "")),
                training_duration_seconds=training_duration,
                evaluation_duration_seconds=evaluation_duration,
                metrics=run_metrics,
            ),
        )
        emit({"type": "stage", "name": STAGES[7]})
        emit({"type": "progress", "current": 8, "total": len(STAGES)})
        emit({"type": "completed", "result_dir": str(run_dir)})
        return 0
    except Exception:
        LOGGER.exception("Training failed")
        emit(
            {
                "type": "error",
                "message": "Training failed",
                "details": traceback.format_exc(),
            }
        )
        return 1


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-file", required=True)
    args = parser.parse_args()
    return run(Path(args.project_file))


if __name__ == "__main__":
    raise SystemExit(main())

