"""Training worker entrypoint that communicates through JSON Lines."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path

from app.core.anomalib_service import AnomalibService
from app.core.environment_info import collect_environment_info
from app.core.project_manager import ProjectManager
from app.models.project_config import ProjectConfig

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

    run_dir = manager.create_run_directory(project, "patchcore")
    emit({"type": "stage", "name": STAGES[0]})
    emit({"type": "progress", "current": 1, "total": len(STAGES)})
    emit({"type": "log", "level": "info", "message": f"Loaded project {project.name}"})

    environment = collect_environment_info(Path(project.project_path), project.training.random_seed)
    (run_dir / "model").mkdir(parents=True, exist_ok=True)
    (run_dir / "visualizations").mkdir(parents=True, exist_ok=True)
    (run_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(project.training.to_dict(), indent=2), encoding="utf-8")

    try:
        dataset_root = Path(project.project_path) / "dataset"
        emit({"type": "stage", "name": STAGES[1]})
        emit({"type": "progress", "current": 2, "total": len(STAGES)})
        components = service.create_patchcore_components(
            dataset_root=dataset_root,
            ok_train=dataset_root / "ok_train",
            ng_test=dataset_root / "ng_test",
            ok_test=dataset_root / "ok_test",
            masks=(dataset_root / "masks"),
            config=project.training,
        )
        emit({"type": "stage", "name": STAGES[2]})
        emit({"type": "progress", "current": 3, "total": len(STAGES)})
        emit({"type": "log", "level": "info", "message": "Starting Anomalib Engine.fit"})
        components["engine"].fit(model=components["model"], datamodule=components["datamodule"])
        emit({"type": "stage", "name": STAGES[5]})
        emit({"type": "progress", "current": 6, "total": len(STAGES)})
        metrics = components["engine"].test(model=components["model"], datamodule=components["datamodule"])
        metric_payload = metrics[0] if metrics else {}
        for name, value in metric_payload.items():
            if isinstance(value, (int, float)):
                emit({"type": "metric", "name": name, "value": value})
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

