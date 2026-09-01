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
from app.core.dataset_manifest import build_dataset_manifest, build_effective_split, stage_effective_split, write_dataset_manifest
from app.core.model_registry import ModelExecutionMode
from app.core.prediction_adapter import iter_anomalib_predictions
from app.core.project_manager import ProjectManager
from app.core.quality_metrics import calculate_quality_metrics
from app.core.result_parser import ResultParser
from app.core.run_artifacts import resolve_canonical_checkpoint, write_evaluation_revision, write_run_manifest
from app.core.threshold_calibrator import CalibrationSample, ThresholdCalibrationConfig, ThresholdCalibrator
from app.models.prediction_result import PredictionResult
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


def calibration_samples_from_predictions(
    output: Any,
    source_path_by_staged_path: dict[Path, Path],
) -> list[CalibrationSample]:
    """Convert only held-out calibration predictions into evidence for threshold selection."""
    samples: list[CalibrationSample] = []
    for prediction in iter_anomalib_predictions(output):
        if prediction.image_path not in source_path_by_staged_path:
            raise ValueError(f"Calibration prediction path is not part of the staged dataset: {prediction.image_path}")
        role = prediction.image_path.parent.name
        if role == "validation_ok":
            label = "OK"
        elif role == "validation_ng":
            label = "NG"
        else:
            raise ValueError(f"Calibration prediction path has an unexpected staged role: {prediction.image_path}")
        samples.append(CalibrationSample(score=prediction.score, label=label))
    return samples


def _final_test_predictions(
    output: Any,
    source_path_by_staged_path: dict[Path, Path],
    threshold: float,
) -> list[PredictionResult]:
    """Build final-test rows using the application-calibrated deployment threshold."""
    predictions: list[PredictionResult] = []
    for anomalib_prediction in iter_anomalib_predictions(output):
        staged_path = anomalib_prediction.image_path
        source_path = source_path_by_staged_path.get(staged_path)
        if source_path is None:
            raise ValueError(f"Final-test prediction path is not part of the staged dataset: {staged_path}")
        dataset_role = staged_path.parent.name
        if dataset_role == "final_test_ok":
            ground_truth = "OK"
        elif dataset_role == "final_test_ng":
            ground_truth = "NG"
        else:
            raise ValueError(f"Final-test prediction path has an unexpected staged role: {staged_path}")
        predictions.append(
            PredictionResult(
                source_path=str(source_path),
                predicted_label="NG" if anomalib_prediction.score >= threshold else "OK",
                ground_truth_label=ground_truth,
                anomaly_score=anomalib_prediction.score,
                threshold=threshold,
                original_image=str(source_path),
                dataset_role=dataset_role,
            )
        )
    return predictions


def _model_provenance(definition: Any, config: Any, model: Any) -> dict[str, object]:
    """Describe model identity and encoder provenance without conflating DINO families."""
    payload: dict[str, object] = {
        "algorithm": definition.algorithm or definition.anomalib_class_name or definition.display_name,
        "model_variant": definition.model_variant or definition.key,
        "encoder_family": definition.encoder_family or None,
        "official_anomalib_implementation": definition.official_anomalib_implementation,
        "profile": config.model_profile(),
    }
    if definition.key == "dinomaly_dinov2":
        payload["encoder"] = {"family": "DINOv2", "name": config.dinomaly_encoder_name}
    elif definition.key == "dinomaly_dinov3":
        payload["encoder"] = {"family": "DINOv3", "name": config.dinomaly_encoder_name}
    return payload


def _reset_gpu_peak_memory(device: str) -> None:
    """Reset GPU peak accounting only when a CUDA prediction is actually selected."""
    if device != "gpu":
        return
    try:
        import torch

        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _peak_gpu_memory_mb(device: str) -> float | None:
    """Return measured CUDA peak memory when the runtime provides it."""
    if device != "gpu":
        return None
    try:
        import torch

        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        return None


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

    (run_dir / "model").mkdir(parents=True, exist_ok=True)
    (run_dir / "visualizations").mkdir(parents=True, exist_ok=True)
    result_parser = ResultParser()

    try:
        effective_split = build_effective_split(project.dataset, project.training.split_seed)
        project.training.apply_model_defaults(len(effective_split.training_ok))
        manifest = build_dataset_manifest(effective_split.roles(), Path(project.project_path))
        write_dataset_manifest(run_dir / "dataset_manifest.json", manifest)
        calibration_manifest = build_dataset_manifest(
            {
                "validation_ok": effective_split.validation_ok,
                "validation_ng": effective_split.validation_ng,
            },
            Path(project.project_path),
        )
        write_dataset_manifest(run_dir / "calibration_manifest.json", calibration_manifest)
        final_test_manifest = build_dataset_manifest(
            {
                "final_test_ok": effective_split.final_test_ok,
                "final_test_ng": effective_split.final_test_ng,
            },
            Path(project.project_path),
        )
        write_dataset_manifest(run_dir / "final_test_manifest.json", final_test_manifest)
        staged_dataset = stage_effective_split(effective_split, project.dataset, run_dir / "dataset_snapshot")
        environment = collect_environment_info(Path(project.project_path), project.training.random_seed)
        (run_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
        (run_dir / "config.json").write_text(json.dumps(project.training.to_dict(), indent=2), encoding="utf-8")
        emit(
            {
                "type": "log",
                "level": "info",
                "message": f"Staged deterministic split: {effective_split.counts()}",
            }
        )
        emit({"type": "stage", "name": STAGES[1]})
        emit({"type": "progress", "current": 2, "total": len(STAGES)})
        progress_callback = create_training_progress_callback(emit)
        components = service.create_components(
            dataset=staged_dataset.training_config,
            config=project.training,
            run_directory=run_dir,
            callbacks=[progress_callback],
            calibration_mode=True,
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
        canonical_checkpoint = resolve_canonical_checkpoint(components["engine"])
        emit(
            {
                "type": "log",
                "level": "info",
                "message": f"Using Anomalib canonical checkpoint: {canonical_checkpoint.path.name}",
            }
        )
        calibration_predictions_output = components["engine"].predict(
            model=components["model"],
            datamodule=components["datamodule"],
            return_predictions=True,
            ckpt_path=str(canonical_checkpoint.path),
        )
        calibration_samples = calibration_samples_from_predictions(
            calibration_predictions_output,
            staged_dataset.source_path_by_staged_path,
        )
        expected_calibration_count = sum(effective_split.counts()["validation"].values())
        if len(calibration_samples) != expected_calibration_count:
            raise RuntimeError(
                f"Calibration prediction count mismatch: expected {expected_calibration_count}, "
                f"received {len(calibration_samples)}."
            )
        calibration_result = ThresholdCalibrator().calibrate(
            calibration_samples,
            ThresholdCalibrationConfig(
                method=project.training.threshold_method,
                target_normal_false_reject_rate=project.training.target_normal_false_reject_rate,
                minimum_required_ng_recall=project.training.minimum_required_ng_recall,
            ),
        )
        decision_threshold = calibration_result.threshold_value
        threshold_metadata = calibration_result.to_dict()
        threshold_metadata["calibration_manifest_sha256"] = calibration_manifest["manifest_sha256"]
        threshold_metadata["calibration_manifest_path"] = str(run_dir / "calibration_manifest.json")
        emit(
            {
                "type": "log",
                "level": "warning" if calibration_result.warning else "info",
                "message": (
                    f"Calibrated threshold {decision_threshold:.6g} with {calibration_result.threshold_method} "
                    f"from {calibration_result.calibration_sample_count} held-out validation images."
                ),
            }
        )
        emit({"type": "stage", "name": STAGES[5]})
        emit({"type": "progress", "current": 6, "total": len(STAGES)})
        evaluation_started = perf_counter()
        final_test_datamodule = service.create_datamodule(
            staged_dataset.final_test_config,
            project.training,
            calibration_mode=False,
        )
        _reset_gpu_peak_memory(str(components["device"]))
        emit({"type": "stage", "name": STAGES[6]})
        final_predictions_output = components["engine"].predict(
            model=components["model"],
            datamodule=final_test_datamodule,
            return_predictions=True,
            ckpt_path=str(canonical_checkpoint.path),
        )
        evaluation_duration = perf_counter() - evaluation_started
        predictions = _final_test_predictions(
            final_predictions_output,
            staged_dataset.source_path_by_staged_path,
            decision_threshold,
        )
        expected_prediction_count = sum(effective_split.counts()["final_test"].values())
        if len(predictions) != expected_prediction_count:
            raise RuntimeError(
                f"Final-test prediction count mismatch: expected {expected_prediction_count}, received {len(predictions)}."
            )
        mean_inference_latency_ms = evaluation_duration * 1000 / len(predictions)
        peak_gpu_memory_mb = _peak_gpu_memory_mb(str(components["device"]))
        run_metrics: dict[str, float | str | None] = {
            "Mean Inference Latency (ms/image)": mean_inference_latency_ms,
            "P95 Inference Latency (ms/image)": "NOT MEASURED (aggregate prediction timing)",
            "Peak GPU Memory (MB)": peak_gpu_memory_mb if peak_gpu_memory_mb is not None else "NOT MEASURED",
            "Model Size (bytes)": canonical_checkpoint.path.stat().st_size,
        }
        result_parser.export_predictions_csv(run_dir / "predictions.csv", predictions)
        quality_report = calculate_quality_metrics(predictions)
        run_metrics.update(quality_report.metrics)
        run_metrics["Quality Status"] = quality_report.status
        run_metrics["Threshold Method"] = calibration_result.threshold_method
        run_metrics["Calibration Image Count"] = calibration_result.calibration_sample_count
        run_metrics["Calibration Normal Image Count"] = calibration_result.normal_calibration_sample_count
        run_metrics["Calibration NG Image Count"] = calibration_result.abnormal_calibration_sample_count
        run_metrics["Calibration Target False Reject Rate"] = calibration_result.target_false_reject_rate
        run_metrics["Calibration Observed False Reject Rate"] = calibration_result.observed_calibration_false_reject_rate
        run_metrics["Calibration OK Score P05"] = calibration_result.normal_score_quantiles.get("p05")
        run_metrics["Calibration OK Score P50"] = calibration_result.normal_score_quantiles.get("p50")
        run_metrics["Calibration OK Score P95"] = calibration_result.normal_score_quantiles.get("p95")
        run_metrics["Calibration OK Score IQR"] = calibration_result.normal_score_iqr
        if calibration_result.abnormal_score_quantiles:
            run_metrics["Calibration NG Score P05"] = calibration_result.abnormal_score_quantiles.get("p05")
            run_metrics["Calibration NG Score P50"] = calibration_result.abnormal_score_quantiles.get("p50")
            run_metrics["Calibration NG Score P95"] = calibration_result.abnormal_score_quantiles.get("p95")
            run_metrics["Calibration NG Score IQR"] = calibration_result.abnormal_score_iqr
        if quality_report.warning:
            run_metrics["Quality Evidence Warning"] = quality_report.warning
        for name, value in quality_report.metrics.items():
            if isinstance(value, (int, float)):
                emit({"type": "metric", "name": name, "value": value})
        revision_path = write_evaluation_revision(
            run_dir,
            canonical_checkpoint=canonical_checkpoint,
            calibration_manifest_sha256=str(calibration_manifest["manifest_sha256"]),
            final_test_manifest_sha256=str(final_test_manifest["manifest_sha256"]),
            threshold_metadata=threshold_metadata,
            evaluation_metrics=quality_report.metrics,
        )
        threshold_metadata["threshold_revision"] = revision_path.stem
        run_metrics["Threshold Revision"] = revision_path.stem
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
                final_checkpoint_path=str(canonical_checkpoint.path),
                final_checkpoint_sha256=canonical_checkpoint.sha256,
                dataset_manifest_sha256=str(manifest["manifest_sha256"]),
                calibration_manifest_sha256=str(calibration_manifest["manifest_sha256"]),
                final_test_manifest_sha256=str(final_test_manifest["manifest_sha256"]),
                evaluation_revision_id=revision_path.stem,
                model_variant=definition.model_variant or definition.key,
                encoder_family=definition.encoder_family,
                threshold_metadata=threshold_metadata,
                mean_inference_latency_ms=mean_inference_latency_ms,
                peak_gpu_memory_mb=peak_gpu_memory_mb,
                quality_status=quality_report.status,
                metrics=run_metrics,
                predictions=predictions,
            ),
        )
        write_run_manifest(
            run_dir / "run_manifest.json",
            canonical_checkpoint=canonical_checkpoint,
            dataset_manifest_sha256=str(manifest["manifest_sha256"]),
            split_counts=effective_split.counts(),
            threshold=decision_threshold,
            threshold_metadata=threshold_metadata,
            extra={
                "model": _model_provenance(definition, project.training, components["model"]),
                "config_path": str(run_dir / "config.json"),
                "environment_path": str(run_dir / "environment.json"),
                "calibration_manifest_path": str(run_dir / "calibration_manifest.json"),
                "final_test_manifest_path": str(run_dir / "final_test_manifest.json"),
                "evaluation_revision_path": str(revision_path),
                "predictions_path": str(run_dir / "predictions.csv"),
                "quality_status": quality_report.status,
            },
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

