"""Checkpoint-preserving recalibration and reevaluation service."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.dataset_manifest import build_dataset_manifest, write_dataset_manifest
from app.core.prediction_adapter import iter_anomalib_predictions
from app.core.quality_metrics import QualityReport, calculate_quality_metrics
from app.core.result_parser import ResultParser
from app.core.run_artifacts import (
    CanonicalCheckpoint,
    read_canonical_checkpoint,
    write_evaluation_revision,
)
from app.core.threshold_calibrator import CalibrationSample, ThresholdCalibrationConfig, ThresholdCalibrator
from app.models.dataset_config import DatasetConfig, DatasetRole
from app.models.prediction_result import PredictionResult
from app.models.training_config import TrainingConfig
from app.services.anomalib_service import AnomalibService


@dataclass(frozen=True, slots=True)
class EvaluationDirectories:
    """New held-out calibration and final-test directories used by one revision."""

    calibration_ok: Path
    final_test_ok: Path
    calibration_ng: Path | None = None
    final_test_ng: Path | None = None


@dataclass(frozen=True, slots=True)
class EvaluationRevisionResult:
    """Persisted outcome of one reevaluation that leaves the model unchanged."""

    revision_path: Path
    canonical_checkpoint: CanonicalCheckpoint
    threshold_metadata: dict[str, object]
    quality_report: QualityReport
    predictions_path: Path


class EvaluationRevisionService:
    """Evaluate a canonical trained model on new data without ever calling Engine.fit."""

    def __init__(self, anomalib_service: AnomalibService | None = None) -> None:
        self.anomalib_service = anomalib_service or AnomalibService()

    def reevaluate(
        self,
        run_directory: Path,
        directories: EvaluationDirectories,
        threshold_config: ThresholdCalibrationConfig,
    ) -> EvaluationRevisionResult:
        """Calibrate and evaluate a verified existing checkpoint on separate new folders."""
        run_directory = run_directory.expanduser().resolve()
        canonical_checkpoint = read_canonical_checkpoint(run_directory)
        config = self._load_training_config(run_directory)
        resolved = self._validate_directories(directories)
        revisions_directory = run_directory / "evaluation_revisions"
        revisions_directory.mkdir(parents=True, exist_ok=True)
        components = self.anomalib_service.create_inference_components(config, revisions_directory)
        calibration_datamodule = self.anomalib_service.create_datamodule(
            self._dataset_config(resolved.calibration_ok, resolved.calibration_ok, resolved.calibration_ng),
            config,
            calibration_mode=False,
        )
        calibration_output = components["engine"].predict(
            model=components["model"],
            datamodule=calibration_datamodule,
            return_predictions=True,
            ckpt_path=str(canonical_checkpoint.path),
        )
        calibration_samples = self._calibration_samples(
            calibration_output,
            normal_directory=resolved.calibration_ok,
            abnormal_directory=resolved.calibration_ng,
        )
        calibration_result = ThresholdCalibrator().calibrate(calibration_samples, threshold_config)
        calibration_manifest = build_dataset_manifest(
            {
                "calibration_ok": self._images(resolved.calibration_ok),
                "calibration_ng": self._images(resolved.calibration_ng),
            }
        )
        final_test_datamodule = self.anomalib_service.create_datamodule(
            self._dataset_config(resolved.calibration_ok, resolved.final_test_ok, resolved.final_test_ng),
            config,
            calibration_mode=False,
        )
        final_predictions = self._final_predictions(
            components["engine"].predict(
                model=components["model"],
                datamodule=final_test_datamodule,
                return_predictions=True,
                ckpt_path=str(canonical_checkpoint.path),
            ),
            normal_directory=resolved.final_test_ok,
            abnormal_directory=resolved.final_test_ng,
            threshold=calibration_result.threshold_value,
        )
        quality_report = calculate_quality_metrics(final_predictions)
        final_test_manifest = build_dataset_manifest(
            {
                "final_test_ok": self._images(resolved.final_test_ok),
                "final_test_ng": self._images(resolved.final_test_ng),
            }
        )
        threshold_metadata = calibration_result.to_dict()
        threshold_metadata["calibration_manifest_sha256"] = calibration_manifest["manifest_sha256"]
        revision_path = write_evaluation_revision(
            run_directory,
            canonical_checkpoint=canonical_checkpoint,
            calibration_manifest_sha256=str(calibration_manifest["manifest_sha256"]),
            final_test_manifest_sha256=str(final_test_manifest["manifest_sha256"]),
            threshold_metadata=threshold_metadata,
            evaluation_metrics=quality_report.metrics,
        )
        threshold_metadata["threshold_revision"] = revision_path.stem
        revision_directory = revision_path.parent
        write_dataset_manifest(revision_directory / f"{revision_path.stem}_calibration_manifest.json", calibration_manifest)
        write_dataset_manifest(revision_directory / f"{revision_path.stem}_final_test_manifest.json", final_test_manifest)
        predictions_path = revision_directory / f"{revision_path.stem}_predictions.csv"
        ResultParser().export_predictions_csv(predictions_path, final_predictions)
        return EvaluationRevisionResult(
            revision_path=revision_path,
            canonical_checkpoint=canonical_checkpoint,
            threshold_metadata=threshold_metadata,
            quality_report=quality_report,
            predictions_path=predictions_path,
        )

    @staticmethod
    def _load_training_config(run_directory: Path) -> TrainingConfig:
        path = run_directory / "config.json"
        if not path.is_file():
            raise FileNotFoundError(f"Training configuration not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Training configuration must be a JSON object.")
        return TrainingConfig.from_dict(payload)

    @staticmethod
    def _validate_directories(directories: EvaluationDirectories) -> EvaluationDirectories:
        for description, path in (
            ("Calibration OK", directories.calibration_ok),
            ("Final-test OK", directories.final_test_ok),
            ("Calibration NG", directories.calibration_ng),
            ("Final-test NG", directories.final_test_ng),
        ):
            if path is not None and (not path.is_dir() or not EvaluationRevisionService._images(path)):
                raise ValueError(f"{description} directory must exist and contain images: {path}")
        return EvaluationDirectories(
            calibration_ok=directories.calibration_ok.resolve(),
            calibration_ng=directories.calibration_ng.resolve() if directories.calibration_ng else None,
            final_test_ok=directories.final_test_ok.resolve(),
            final_test_ng=directories.final_test_ng.resolve() if directories.final_test_ng else None,
        )

    @staticmethod
    def _dataset_config(normal_anchor: Path, normal_test: Path, abnormal_test: Path | None) -> DatasetConfig:
        config = DatasetConfig()
        config.folders[DatasetRole.OK_TRAIN].path = str(normal_anchor)
        config.folders[DatasetRole.OK_TEST].path = str(normal_test)
        if abnormal_test is not None:
            config.folders[DatasetRole.NG_TEST].path = str(abnormal_test)
        return config

    @staticmethod
    def _images(directory: Path | None) -> list[Path]:
        if directory is None:
            return []
        return sorted(
            (path.resolve() for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}),
            key=lambda path: str(path).casefold(),
        )

    @staticmethod
    def _calibration_samples(
        output: Any,
        *,
        normal_directory: Path,
        abnormal_directory: Path | None,
    ) -> list[CalibrationSample]:
        labels = {path: "OK" for path in EvaluationRevisionService._images(normal_directory)}
        labels.update({path: "NG" for path in EvaluationRevisionService._images(abnormal_directory)})
        samples: list[CalibrationSample] = []
        for prediction in iter_anomalib_predictions(output):
            try:
                label = labels[prediction.image_path]
            except KeyError as exc:
                raise ValueError(f"Calibration prediction path is outside the selected revision folders: {prediction.image_path}") from exc
            samples.append(CalibrationSample(score=prediction.score, label=label))
        if len(samples) != len(labels):
            raise RuntimeError(f"Calibration prediction count mismatch: expected {len(labels)}, received {len(samples)}.")
        return samples

    @staticmethod
    def _final_predictions(
        output: Any,
        *,
        normal_directory: Path,
        abnormal_directory: Path | None,
        threshold: float,
    ) -> list[PredictionResult]:
        labels = {path: ("OK", "final_test_ok") for path in EvaluationRevisionService._images(normal_directory)}
        labels.update({path: ("NG", "final_test_ng") for path in EvaluationRevisionService._images(abnormal_directory)})
        results: list[PredictionResult] = []
        for prediction in iter_anomalib_predictions(output):
            try:
                ground_truth, role = labels[prediction.image_path]
            except KeyError as exc:
                raise ValueError(f"Final-test prediction path is outside the selected revision folders: {prediction.image_path}") from exc
            results.append(
                PredictionResult(
                    source_path=str(prediction.image_path),
                    original_image=str(prediction.image_path),
                    predicted_label="NG" if prediction.score >= threshold else "OK",
                    ground_truth_label=ground_truth,
                    anomaly_score=prediction.score,
                    threshold=threshold,
                    dataset_role=role,
                )
            )
        if len(results) != len(labels):
            raise RuntimeError(f"Final-test prediction count mismatch: expected {len(labels)}, received {len(results)}.")
        return results