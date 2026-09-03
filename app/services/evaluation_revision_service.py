"""Checkpoint-preserving recalibration and reevaluation service."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from app.core.dataset_manifest import build_dataset_manifest, write_dataset_manifest
from app.core.decision_score import resolve_decision_score
from app.core.inspection_region import InspectionRegionProcessor, validate_inspection_region_sources
from app.core.prediction_artifacts import PredictionArtifacts, inspection_region_metadata, save_prediction_artifacts
from app.core.prediction_adapter import explicitly_postprocessed_predict, iter_anomalib_predictions, iter_preprocessed_predictions
from app.core.prediction_contract import PREDICTION_CONTRACT_VERSION, RAW_SCORE_SEMANTIC
from app.core.preprocessing_pipeline import PreprocessingPipeline
from app.core.quality_metrics import FinalTestAcceptancePolicy, QualityReport, calculate_quality_metrics
from app.core.result_parser import ResultParser
from app.core.score_diagnostics import summarize_prediction_score_ranges
from app.core.run_artifacts import (
    CanonicalCheckpoint,
    read_canonical_checkpoint,
    next_evaluation_revision_id,
    read_persisted_pixel_operating_point,
    read_verified_inspection_region,
    read_verified_preprocessing_plan,
    write_evaluation_revision,
)
from app.core.threshold_calibrator import CalibrationSample, ThresholdCalibrationConfig, ThresholdCalibrator
from app.core.threshold_contract import PixelThresholdOperatingPoint
from app.models.dataset_config import DatasetConfig, DatasetRole
from app.models.prediction_result import PredictionResult
from app.models.preprocessing_config import PreprocessingTile
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
        pixel_operating_point: PixelThresholdOperatingPoint | None = None,
    ) -> EvaluationRevisionResult:
        """Calibrate and evaluate a verified existing checkpoint on separate new folders."""
        run_directory = run_directory.expanduser().resolve()
        canonical_checkpoint = read_canonical_checkpoint(run_directory)
        pixel_operating_point = pixel_operating_point or read_persisted_pixel_operating_point(run_directory)
        pixel_operating_point.validate()
        pixel_threshold = pixel_operating_point.active_threshold
        config = self._load_training_config(run_directory)
        inspection_region = read_verified_inspection_region(run_directory)
        preprocessing_plan = read_verified_preprocessing_plan(run_directory)
        preprocessing_pipeline = (
            PreprocessingPipeline(inspection_region, preprocessing_plan) if preprocessing_plan is not None else None
        )
        resolved = self._validate_directories(directories)
        validate_inspection_region_sources(
            inspection_region,
            [
                *self._images(resolved.calibration_ok),
                *self._images(resolved.calibration_ng),
                *self._images(resolved.final_test_ok),
                *self._images(resolved.final_test_ng),
            ],
        )
        revisions_directory = run_directory / "evaluation_revisions"
        revisions_directory.mkdir(parents=True, exist_ok=True)
        revision_id = next_evaluation_revision_id(run_directory)
        artifact_directory = revisions_directory / f"{revision_id}_prediction_artifacts"
        components = (
            self.anomalib_service.create_inference_components(config, revisions_directory, preprocessing_plan)
            if preprocessing_plan is not None
            else self.anomalib_service.create_inference_components(config, revisions_directory)
        )
        if preprocessing_pipeline is None:
            calibration_datamodule = self.anomalib_service.create_datamodule(
                self._dataset_config(resolved.calibration_ok, resolved.calibration_ok, resolved.calibration_ng),
                config,
                calibration_mode=False,
                inspection_region=inspection_region,
            )
            calibration_samples = self._calibration_samples(
                explicitly_postprocessed_predict(
                    components["engine"],
                    model=components["model"],
                    datamodule=calibration_datamodule,
                    return_predictions=True,
                    ckpt_path=str(canonical_checkpoint.path),
                ),
                normal_directory=resolved.calibration_ok,
                abnormal_directory=resolved.calibration_ng,
            )
        else:
            calibration_predictions = self._preprocessed_predictions(
                    components,
                    canonical_checkpoint.path,
                    [*self._images(resolved.calibration_ok), *self._images(resolved.calibration_ng)],
                    preprocessing_pipeline,
                )
            calibration_samples = self._calibration_samples_from_scores(
                (
                    (prediction.source_path, prediction.score, prediction.score_semantic)
                    for prediction in calibration_predictions
                ),
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
        if preprocessing_pipeline is None:
            final_test_datamodule = self.anomalib_service.create_datamodule(
                self._dataset_config(resolved.calibration_ok, resolved.final_test_ok, resolved.final_test_ng),
                config,
                calibration_mode=False,
                inspection_region=inspection_region,
            )
            final_predictions = self._final_predictions(
                explicitly_postprocessed_predict(
                    components["engine"],
                    model=components["model"],
                    datamodule=final_test_datamodule,
                    return_predictions=True,
                    ckpt_path=str(canonical_checkpoint.path),
                ),
                normal_directory=resolved.final_test_ok,
                abnormal_directory=resolved.final_test_ng,
                threshold=calibration_result.threshold_value,
                artifact_directory=artifact_directory,
                inspection_region=inspection_region,
                pixel_threshold=pixel_threshold,
            )
        else:
            final_test_predictions = self._preprocessed_predictions(
                    components,
                    canonical_checkpoint.path,
                    [*self._images(resolved.final_test_ok), *self._images(resolved.final_test_ng)],
                    preprocessing_pipeline,
                )
            final_predictions = self._final_predictions_from_preprocessed(
                final_test_predictions,
                normal_directory=resolved.final_test_ok,
                abnormal_directory=resolved.final_test_ng,
                threshold=calibration_result.threshold_value,
                artifact_directory=artifact_directory,
                inspection_region=inspection_region,
                preprocessing_pipeline=preprocessing_pipeline,
                pixel_threshold=pixel_threshold,
            )
        quality_report = calculate_quality_metrics(
            final_predictions,
            FinalTestAcceptancePolicy(
                maximum_false_reject_rate=config.maximum_final_test_false_reject_rate,
                minimum_ok_test_images=config.minimum_final_test_ok_images,
                minimum_ng_test_images=config.minimum_final_test_ng_images,
            ),
        )
        final_test_manifest = build_dataset_manifest(
            {
                "final_test_ok": self._images(resolved.final_test_ok),
                "final_test_ng": self._images(resolved.final_test_ng),
            }
        )
        threshold_metadata = calibration_result.to_dict()
        threshold_metadata["final_test_score_ranges"] = summarize_prediction_score_ranges(final_predictions)
        threshold_metadata["calibration_manifest_sha256"] = calibration_manifest["manifest_sha256"]
        revision_path = write_evaluation_revision(
            run_directory,
            canonical_checkpoint=canonical_checkpoint,
            calibration_manifest_sha256=str(calibration_manifest["manifest_sha256"]),
            final_test_manifest_sha256=str(final_test_manifest["manifest_sha256"]),
            threshold_metadata=threshold_metadata,
            evaluation_metrics=quality_report.metrics,
            revision_id=revision_id,
            pixel_operating_point=pixel_operating_point.to_dict(),
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
        return EvaluationRevisionService._calibration_samples_from_scores(
            (
                (
                    prediction.image_path,
                    resolve_decision_score(
                        None,
                        postprocessed_image_score=prediction.score,
                        raw_image_score=prediction.raw_image_score,
                    ).value,
                    resolve_decision_score(
                        None,
                        postprocessed_image_score=prediction.score,
                        raw_image_score=prediction.raw_image_score,
                    ).semantic,
                )
                for prediction in iter_anomalib_predictions(output)
            ),
            normal_directory=normal_directory,
            abnormal_directory=abnormal_directory,
        )

    @staticmethod
    def _calibration_samples_from_scores(
        predictions: Any,
        *,
        normal_directory: Path,
        abnormal_directory: Path | None,
    ) -> list[CalibrationSample]:
        labels = {path: "OK" for path in EvaluationRevisionService._images(normal_directory)}
        labels.update({path: "NG" for path in EvaluationRevisionService._images(abnormal_directory)})
        samples: list[CalibrationSample] = []
        for prediction in predictions:
            prediction_path, score, *semantic = prediction
            try:
                label = labels[prediction_path]
            except KeyError as exc:
                raise ValueError(f"Calibration prediction path is outside the selected revision folders: {prediction_path}") from exc
            samples.append(
                CalibrationSample(
                    score=score,
                    label=label,
                    score_semantic=str(semantic[0]) if semantic else "anomalib_postprocessed_pred_score_v1",
                )
            )
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
        artifact_directory: Path | None = None,
        inspection_region=None,
        pixel_threshold: float | None = None,
    ) -> list[PredictionResult]:
        labels = {path: ("OK", "final_test_ok") for path in EvaluationRevisionService._images(normal_directory)}
        labels.update({path: ("NG", "final_test_ng") for path in EvaluationRevisionService._images(abnormal_directory)})
        region_metadata = inspection_region_metadata(inspection_region) if inspection_region is not None else {}
        results: list[PredictionResult] = []
        for prediction in iter_anomalib_predictions(output):
            decision_score = resolve_decision_score(
                None,
                postprocessed_image_score=prediction.score,
                raw_image_score=prediction.raw_image_score,
            )
            try:
                ground_truth, role = labels[prediction.image_path]
            except KeyError as exc:
                raise ValueError(f"Final-test prediction path is outside the selected revision folders: {prediction.image_path}") from exc
            artifacts = EvaluationRevisionService._prediction_artifacts(
                prediction.image_path,
                prediction.anomaly_map,
                artifact_directory,
                len(results),
                inspection_region=inspection_region,
                pixel_threshold=pixel_threshold,
                raw_anomaly_map=prediction.raw_anomaly_map,
            )
            results.append(
                PredictionResult(
                    source_path=str(prediction.image_path),
                    original_image=str(prediction.image_path),
                    predicted_label="NG" if decision_score.value >= threshold else "OK",
                    ground_truth_label=ground_truth,
                    anomaly_score=decision_score.value,
                    threshold=threshold,
                    dataset_role=role,
                    native_image_score=decision_score.value,
                    native_tile_scores=[decision_score.value],
                    score_semantic=decision_score.semantic,
                    raw_image_score=prediction.raw_image_score,
                    raw_score_semantic=RAW_SCORE_SEMANTIC if prediction.raw_image_score is not None else "",
                    raw_anomaly_map=artifacts.raw_anomaly_map,
                    postprocessed_image_score=decision_score.value,
                    postprocessed_score_semantic=decision_score.semantic,
                    postprocessed_anomaly_map=artifacts.continuous_anomaly_map,
                    prediction_contract_version=PREDICTION_CONTRACT_VERSION,
                    continuous_anomaly_map=artifacts.continuous_anomaly_map,
                    anomaly_map=artifacts.heatmap_image,
                    overlay_image=artifacts.overlay_image,
                    binary_mask=artifacts.binary_mask,
                    contour_overlay_image=artifacts.contour_overlay_image,
                    pixel_threshold=artifacts.pixel_threshold,
                    pixel_threshold_comparator=artifacts.pixel_threshold_comparator,
                    pixel_threshold_semantic=artifacts.pixel_threshold_semantic,
                    map_display_normalization=artifacts.display_normalization or {},
                    region_metadata=region_metadata,
                )
            )
        if len(results) != len(labels):
            raise RuntimeError(f"Final-test prediction count mismatch: expected {len(labels)}, received {len(results)}.")
        return results

    @staticmethod
    def _final_predictions_from_preprocessed(
        predictions: list[Any],
        *,
        normal_directory: Path,
        abnormal_directory: Path | None,
        threshold: float,
        artifact_directory: Path | None,
        inspection_region,
        preprocessing_pipeline: PreprocessingPipeline,
        pixel_threshold: float | None,
    ) -> list[PredictionResult]:
        labels = {path: ("OK", "final_test_ok") for path in EvaluationRevisionService._images(normal_directory)}
        labels.update({path: ("NG", "final_test_ng") for path in EvaluationRevisionService._images(abnormal_directory)})
        region_metadata = inspection_region_metadata(inspection_region)
        results: list[PredictionResult] = []
        for prediction in predictions:
            try:
                ground_truth, role = labels[prediction.source_path]
            except KeyError as exc:
                raise ValueError(f"Final-test prediction path is outside the selected revision folders: {prediction.source_path}") from exc
            artifacts = EvaluationRevisionService._prediction_artifacts(
                prediction.source_path,
                prediction.anomaly_map,
                artifact_directory,
                len(results),
                preprocessing_pipeline=preprocessing_pipeline,
                pixel_threshold=pixel_threshold,
                valid_roi_mask=prediction.valid_roi_mask,
                raw_anomaly_map=prediction.raw_anomaly_map,
            )
            results.append(
                PredictionResult(
                    source_path=str(prediction.source_path),
                    original_image=str(prediction.source_path),
                    predicted_label="NG" if prediction.score >= threshold else "OK",
                    ground_truth_label=ground_truth,
                    anomaly_score=prediction.score,
                    threshold=threshold,
                    dataset_role=role,
                    native_image_score=prediction.native_image_score,
                    native_tile_scores=list(prediction.native_tile_scores),
                    score_semantic=prediction.score_semantic,
                    raw_image_score=prediction.raw_image_score,
                    raw_score_semantic=RAW_SCORE_SEMANTIC if prediction.raw_image_score is not None else "",
                    raw_anomaly_map=artifacts.raw_anomaly_map,
                    postprocessed_image_score=prediction.score,
                    postprocessed_score_semantic=prediction.score_semantic,
                    postprocessed_anomaly_map=artifacts.continuous_anomaly_map,
                    prediction_contract_version=PREDICTION_CONTRACT_VERSION,
                    continuous_anomaly_map=artifacts.continuous_anomaly_map,
                    anomaly_map=artifacts.heatmap_image,
                    overlay_image=artifacts.overlay_image,
                    binary_mask=artifacts.binary_mask,
                    contour_overlay_image=artifacts.contour_overlay_image,
                    pixel_threshold=artifacts.pixel_threshold,
                    pixel_threshold_comparator=artifacts.pixel_threshold_comparator,
                    pixel_threshold_semantic=artifacts.pixel_threshold_semantic,
                    map_display_normalization=artifacts.display_normalization or {},
                    region_metadata=region_metadata,
                )
            )
        if len(results) != len(labels):
            raise RuntimeError(f"Final-test prediction count mismatch: expected {len(labels)}, received {len(results)}.")
        return results

    @staticmethod
    def _prediction_artifacts(
        source_path: Path,
        anomaly_map: Any,
        artifact_directory: Path | None,
        index: int,
        *,
        inspection_region=None,
        preprocessing_pipeline: PreprocessingPipeline | None = None,
        pixel_threshold: float | None = None,
        valid_roi_mask: Any = None,
        raw_anomaly_map: Any = None,
    ) -> PredictionArtifacts:
        if artifact_directory is None:
            return PredictionArtifacts()
        rectified_image = None
        if preprocessing_pipeline is not None:
            _prepared, rectified_image = preprocessing_pipeline.prepare_path_with_rectified(source_path)
        elif inspection_region is not None and inspection_region.enabled:
            rectified_image = InspectionRegionProcessor(inspection_region).apply_path(source_path)
        return save_prediction_artifacts(
            source_path,
            anomaly_map,
            artifact_directory,
            index,
            rectified_image=rectified_image,
            pixel_threshold=pixel_threshold,
            valid_roi_mask=valid_roi_mask,
            raw_anomaly_map=raw_anomaly_map,
        )

    @staticmethod
    def _final_predictions_from_scores(
        predictions: Any,
        *,
        normal_directory: Path,
        abnormal_directory: Path | None,
        threshold: float,
    ) -> list[PredictionResult]:
        return EvaluationRevisionService._final_predictions_from_scores_legacy(
            predictions,
            normal_directory=normal_directory,
            abnormal_directory=abnormal_directory,
            threshold=threshold,
        )

    @staticmethod
    def _final_predictions_from_scores_legacy(
        predictions: Any,
        *,
        normal_directory: Path,
        abnormal_directory: Path | None,
        threshold: float,
    ) -> list[PredictionResult]:
        labels = {path: ("OK", "final_test_ok") for path in EvaluationRevisionService._images(normal_directory)}
        labels.update({path: ("NG", "final_test_ng") for path in EvaluationRevisionService._images(abnormal_directory)})
        results: list[PredictionResult] = []
        for prediction_path, score in predictions:
            try:
                ground_truth, role = labels[prediction_path]
            except KeyError as exc:
                raise ValueError(f"Final-test prediction path is outside the selected revision folders: {prediction_path}") from exc
            results.append(
                PredictionResult(
                    source_path=str(prediction_path),
                    original_image=str(prediction_path),
                    predicted_label="NG" if score >= threshold else "OK",
                    ground_truth_label=ground_truth,
                    anomaly_score=score,
                    threshold=threshold,
                    dataset_role=role,
                )
            )
        if len(results) != len(labels):
            raise RuntimeError(f"Final-test prediction count mismatch: expected {len(labels)}, received {len(results)}.")
        return results

    @staticmethod
    def _preprocessed_predictions(
        components: dict[str, Any],
        checkpoint_path: Path,
        source_paths: list[Path],
        preprocessing_pipeline: PreprocessingPipeline,
    ) -> list[Any]:
        """Run a saved preprocessing plan and retain source scores, maps, and provenance."""
        from PIL import Image
        from anomalib.data import PredictDataset

        with TemporaryDirectory(prefix="aigaikan-evaluation-v2-") as temporary_directory:
            prepared_directory = Path(temporary_directory) / "prepared"
            prepared_directory.mkdir()
            source_path_by_staged_path: dict[Path, Path] = {}
            preprocessing_tile_by_staged_path: dict[Path, PreprocessingTile] = {}
            for source_index, source_path in enumerate(source_paths):
                for prepared in preprocessing_pipeline.prepare_path(source_path):
                    staged_path = (
                        prepared_directory / f"{source_index:06d}_tile{prepared.tile.index:02d}_{source_path.stem}.png"
                    ).resolve()
                    Image.fromarray(prepared.image_rgb, "RGB").save(staged_path)
                    source_path_by_staged_path[staged_path] = source_path
                    preprocessing_tile_by_staged_path[staged_path] = prepared.tile
            output = explicitly_postprocessed_predict(
                components["engine"],
                model=components["model"],
                dataset=PredictDataset(prepared_directory),
                return_predictions=True,
                ckpt_path=str(checkpoint_path),
            )
            return list(
                iter_preprocessed_predictions(
                    output,
                    source_path_by_staged_path,
                    preprocessing_tile_by_staged_path,
                    preprocessing_pipeline,
                )
            )