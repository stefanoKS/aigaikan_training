"""Immutable threshold revisions generated from persisted postprocessed maps."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from app.core.dataset_manifest import sha256_file
from app.core.inspection_region import InspectionRegionProcessor
from app.core.prediction_artifacts import save_mask_artifacts
from app.core.preprocessing_pipeline import PreprocessingPipeline
from app.core.result_parser import ResultParser
from app.core.run_artifacts import (
    read_canonical_checkpoint,
    read_persisted_pixel_operating_point,
    read_persisted_threshold_metadata,
    read_verified_inspection_region,
    read_verified_preprocessing_plan,
)
from app.core.threshold_contract import ImageThresholdOperatingPoint, PixelThresholdOperatingPoint
from app.models.prediction_result import PredictionResult

THRESHOLD_REVISION_VERSION = 2
ACTIVE_THRESHOLD_REVISION_VERSION = 1


@dataclass(frozen=True, slots=True)
class ThresholdRevisionResult:
    """One immutable decision revision and its regenerated result records."""

    revision_path: Path
    predictions_path: Path
    image_operating_point: ImageThresholdOperatingPoint
    pixel_operating_point: PixelThresholdOperatingPoint
    source: str = "operator_override"
    base_calibrated_threshold: float | None = None
    created_at: str = ""
    operator_note: str = ""


@dataclass(frozen=True, slots=True)
class DecisionThresholdPreview:
    """Persisted-score-only preview of an operator proposed deployment threshold."""

    calibrated_threshold: float
    active_threshold: float
    proposed_threshold: float
    score_semantic: str
    ok_to_ng_changes: int
    ng_to_ok_changes: int
    false_reject_rate: float | None
    ng_recall: float | None
    outside_calibration_range: bool


class ThresholdRevisionService:
    """Regenerate decisions and map artifacts without loading or executing a model."""

    def create_revision(
        self,
        run_directory: Path,
        image_operating_point: ImageThresholdOperatingPoint,
        pixel_operating_point: PixelThresholdOperatingPoint | None = None,
        operator_note: str = "",
    ) -> ThresholdRevisionResult:
        """Create and activate a new immutable revision from a completed run's postprocessed maps."""
        run_directory = run_directory.expanduser().resolve()
        image_operating_point.validate()
        active_revision = self.read_active_revision(run_directory)
        pixel_operating_point = pixel_operating_point or (
            active_revision.pixel_operating_point if active_revision is not None else read_persisted_pixel_operating_point(run_directory)
        )
        pixel_operating_point.validate()
        threshold_metadata = read_persisted_threshold_metadata(run_directory)
        persisted_semantic = threshold_metadata.get("score_semantic")
        if persisted_semantic and persisted_semantic != image_operating_point.score_semantic:
            raise ValueError("New image threshold score semantic does not match the saved run decision domain.")
        if "\x00" in operator_note:
            raise ValueError("Operator note must not contain NUL characters.")

        predictions = self._load_revision_base_predictions(run_directory, active_revision)
        revision_id = self._next_revision_id(run_directory)
        revision_directory = run_directory / "threshold_revisions"
        artifact_directory = revision_directory / f"{revision_id}_prediction_artifacts"
        inspection_region = read_verified_inspection_region(run_directory)
        plan = read_verified_preprocessing_plan(run_directory)
        pipeline = PreprocessingPipeline(inspection_region, plan) if plan is not None else None
        revised_predictions = [
            replace(
                self._revise_prediction(
                prediction,
                image_operating_point,
                pixel_operating_point,
                artifact_directory,
                index,
                inspection_region,
                pipeline,
                ),
                decision_revision_id=revision_id,
            )
            for index, prediction in enumerate(predictions)
        ]
        revision_directory.mkdir(parents=True, exist_ok=True)
        predictions_path = revision_directory / f"{revision_id}_predictions.csv"
        ResultParser().export_predictions_csv(predictions_path, revised_predictions)
        canonical_checkpoint = read_canonical_checkpoint(run_directory)
        payload = {
            "version": THRESHOLD_REVISION_VERSION,
            "revision_id": revision_id,
            "canonical_checkpoint": {"path": str(canonical_checkpoint.path), "sha256": canonical_checkpoint.sha256},
            "image_operating_point": image_operating_point.to_dict(),
            "pixel_operating_point": pixel_operating_point.to_dict(),
            "source_results": "results.json",
            "source_results_sha256": sha256_file(run_directory / "results.json"),
            "predictions_file": predictions_path.name,
            "prediction_count": len(revised_predictions),
            "source": "operator_override",
            "base_calibrated_threshold": float(threshold_metadata.get("threshold_raw", threshold_metadata["threshold_value"])),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "operator_note": operator_note,
            "score_semantic": image_operating_point.score_semantic,
        }
        revision_path = revision_directory / f"{revision_id}.json"
        self._atomic_write_json(revision_path, payload)
        self._activate_revision(run_directory, revision_path)
        return ThresholdRevisionResult(
            revision_path,
            predictions_path,
            image_operating_point,
            pixel_operating_point,
            source="operator_override",
            base_calibrated_threshold=float(payload["base_calibrated_threshold"]),
            created_at=str(payload["created_at"]),
            operator_note=operator_note,
        )

    def preview_decision_threshold(
        self,
        run_directory: Path,
        proposed_threshold: float,
        score_semantic: str,
    ) -> DecisionThresholdPreview:
        """Preview a decision-only revision from persisted scores without model execution."""
        image_point = ImageThresholdOperatingPoint(proposed_threshold, score_semantic)
        image_point.validate()
        run_directory = run_directory.expanduser().resolve()
        metadata = read_persisted_threshold_metadata(run_directory)
        if metadata.get("score_semantic") != score_semantic:
            raise ValueError("Proposed deployment threshold score semantic does not match the saved run decision domain.")
        predictions = self._load_predictions(run_directory)
        active_revision = self.read_active_revision(run_directory)
        active_threshold = active_revision.image_operating_point.threshold if active_revision is not None else float(metadata["threshold_value"])
        old_labels = ["NG" if prediction.anomaly_score >= active_threshold else "OK" for prediction in predictions]
        new_labels = ["NG" if prediction.anomaly_score >= proposed_threshold else "OK" for prediction in predictions]
        actual_ok = [label for label, prediction in zip(new_labels, predictions, strict=True) if prediction.ground_truth_label.upper() == "OK"]
        actual_ng = [label for label, prediction in zip(new_labels, predictions, strict=True) if prediction.ground_truth_label.upper() == "NG"]
        calibration_values = self._calibration_score_range(metadata)
        return DecisionThresholdPreview(
            calibrated_threshold=float(metadata.get("threshold_raw", metadata["threshold_value"])),
            active_threshold=active_threshold,
            proposed_threshold=proposed_threshold,
            score_semantic=score_semantic,
            ok_to_ng_changes=sum(old == "OK" and new == "NG" for old, new in zip(old_labels, new_labels, strict=True)),
            ng_to_ok_changes=sum(old == "NG" and new == "OK" for old, new in zip(old_labels, new_labels, strict=True)),
            false_reject_rate=(sum(label == "NG" for label in actual_ok) / len(actual_ok)) if actual_ok else None,
            ng_recall=(sum(label == "NG" for label in actual_ng) / len(actual_ng)) if actual_ng else None,
            outside_calibration_range=(
                bool(calibration_values)
                and (proposed_threshold < min(calibration_values) or proposed_threshold > max(calibration_values))
            ),
        )

    def activate_revision(self, run_directory: Path, revision_id: str) -> ThresholdRevisionResult:
        """Select one existing immutable revision without altering its contents."""
        run_directory = run_directory.expanduser().resolve()
        if not self._is_revision_id(revision_id):
            raise ValueError("Threshold revision ID must use the form threshold-NNN.")
        revision_path = run_directory / "threshold_revisions" / f"{revision_id}.json"
        if not revision_path.is_file():
            raise FileNotFoundError(f"Threshold revision not found: {revision_path}")
        result = self._read_revision(revision_path)
        self._activate_revision(run_directory, revision_path)
        return result

    @staticmethod
    def read_active_revision(run_directory: Path) -> ThresholdRevisionResult | None:
        """Read the checksummed selected revision, or return ``None`` for legacy run-manifest operation."""
        run_directory = run_directory.expanduser().resolve()
        pointer_path = run_directory / "active_threshold_revision.json"
        if not pointer_path.is_file():
            return None
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        if pointer.get("version") != ACTIVE_THRESHOLD_REVISION_VERSION:
            raise ValueError("Unsupported active threshold revision pointer version.")
        revision_file = pointer.get("revision_file")
        if not isinstance(revision_file, str) or not revision_file.endswith(".json"):
            raise ValueError("Active threshold revision pointer has an invalid revision filename.")
        revision_id = revision_file.removesuffix(".json")
        if not ThresholdRevisionService._is_revision_id(revision_id):
            raise ValueError("Active threshold revision pointer has an invalid revision filename.")
        revision_path = run_directory / "threshold_revisions" / revision_file
        if not revision_path.is_file() or sha256_file(revision_path) != pointer.get("revision_sha256"):
            raise ValueError("Active threshold revision pointer does not match an immutable revision.")
        return ThresholdRevisionService._read_revision(revision_path)

    @staticmethod
    def _read_revision(revision_path: Path) -> ThresholdRevisionResult:
        payload = json.loads(revision_path.read_text(encoding="utf-8"))
        if payload.get("version") not in {1, THRESHOLD_REVISION_VERSION}:
            raise ValueError("Unsupported threshold revision version.")
        if payload.get("revision_id") != revision_path.stem:
            raise ValueError("Threshold revision contents do not match its immutable filename.")
        image_point = ImageThresholdOperatingPoint.from_dict(payload.get("image_operating_point", {}))
        pixel_point = PixelThresholdOperatingPoint.from_dict(payload.get("pixel_operating_point", {}))
        expected_predictions_file = f"{revision_path.stem}_predictions.csv"
        if payload.get("predictions_file") != expected_predictions_file:
            raise ValueError("Threshold revision predictions filename is invalid.")
        predictions_path = revision_path.parent / expected_predictions_file
        if not predictions_path.is_file():
            raise ValueError("Active threshold revision predictions are missing.")
        source = "operator_override" if payload.get("version") == THRESHOLD_REVISION_VERSION else "calibrated"
        return ThresholdRevisionResult(
            revision_path,
            predictions_path,
            image_point,
            pixel_point,
            source=str(payload.get("source", source)),
            base_calibrated_threshold=ThresholdRevisionService._optional_float(payload.get("base_calibrated_threshold")),
            created_at=str(payload.get("created_at", "")),
            operator_note=str(payload.get("operator_note", "")),
        )

    @staticmethod
    def _load_predictions(run_directory: Path) -> list[PredictionResult]:
        results_path = run_directory / "results.json"
        if not results_path.is_file():
            raise FileNotFoundError(f"Completed run results are missing: {results_path}")
        predictions = ResultParser().read_training_run(results_path).predictions
        if not predictions:
            raise ValueError("Completed run has no persisted predictions to revise.")
        return predictions

    @staticmethod
    def _load_revision_base_predictions(
        run_directory: Path,
        active_revision: ThresholdRevisionResult | None,
    ) -> list[PredictionResult]:
        if active_revision is None:
            return ThresholdRevisionService._load_predictions(run_directory)
        predictions = ResultParser().read_predictions_csv(active_revision.predictions_path)
        if not predictions:
            raise ValueError("Active threshold revision has no persisted predictions to revise.")
        return predictions

    @staticmethod
    def _next_revision_id(run_directory: Path) -> str:
        revisions = run_directory / "threshold_revisions"
        existing_ids = (
            [
                int(path.stem.removeprefix("threshold-"))
                for path in revisions.glob("threshold-*.json")
                if ThresholdRevisionService._is_revision_id(path.stem)
            ]
            if revisions.is_dir()
            else []
        )
        return f"threshold-{max(existing_ids, default=0) + 1:03d}"

    @staticmethod
    def _is_revision_id(value: str) -> bool:
        return value.startswith("threshold-") and value[10:].isdigit()

    @staticmethod
    def _revise_prediction(
        prediction: PredictionResult,
        image_point: ImageThresholdOperatingPoint,
        pixel_point: PixelThresholdOperatingPoint,
        artifact_directory: Path,
        index: int,
        inspection_region,
        pipeline: PreprocessingPipeline | None,
    ) -> PredictionResult:
        score_semantic = prediction.score_semantic
        if score_semantic != image_point.score_semantic:
            raise ValueError(f"Prediction score semantic does not match the new threshold: {prediction.source_path}")
        score = prediction.anomaly_score
        if ThresholdRevisionService._same_pixel_policy(prediction, pixel_point):
            return replace(
                prediction,
                threshold=image_point.threshold,
                predicted_label="NG" if score >= image_point.threshold else "OK",
            )
        map_path = Path(prediction.postprocessed_anomaly_map or prediction.continuous_anomaly_map)
        if not map_path.is_file():
            raise FileNotFoundError(f"Postprocessed anomaly map is missing: {map_path}")
        with np.load(map_path, allow_pickle=False) as stored:
            anomaly_map = stored["anomaly_map"]
            valid_mask = stored["valid_roi_mask"] if "valid_roi_mask" in stored.files else None
        source_path = Path(prediction.source_path)
        rectified_image = ThresholdRevisionService._rectified_image(source_path, inspection_region, pipeline)
        artifacts = save_mask_artifacts(
            source_path,
            anomaly_map,
            artifact_directory,
            index,
            rectified_image=rectified_image,
            pixel_threshold=pixel_point.active_threshold,
            valid_roi_mask=valid_mask,
        )
        return replace(
            prediction,
            anomaly_score=score,
            threshold=image_point.threshold,
            predicted_label="NG" if score >= image_point.threshold else "OK",
            binary_mask=artifacts.binary_mask,
            contour_overlay_image=artifacts.contour_overlay_image,
            pixel_threshold=artifacts.pixel_threshold,
            pixel_threshold_comparator=artifacts.pixel_threshold_comparator,
            pixel_threshold_semantic=artifacts.pixel_threshold_semantic,
        )

    @staticmethod
    def _same_pixel_policy(prediction: PredictionResult, point: PixelThresholdOperatingPoint) -> bool:
        return prediction.pixel_threshold == point.active_threshold and (
            not point.enabled
            or (
                prediction.pixel_threshold_comparator == "greater_than_or_equal"
                and prediction.pixel_threshold_semantic == "continuous_anomaly_map_gte_v1"
            )
        )

    @staticmethod
    def _calibration_score_range(metadata: dict[str, object]) -> tuple[float, ...]:
        values: list[float] = []
        for key in ("normal_score_quantiles", "abnormal_score_quantiles"):
            quantiles = metadata.get(key)
            if isinstance(quantiles, dict):
                for value in quantiles.values():
                    try:
                        values.append(float(value))
                    except (TypeError, ValueError):
                        continue
        return tuple(values)

    @staticmethod
    def _optional_float(value: object) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Threshold revision base calibrated threshold must be numeric.") from exc

    @staticmethod
    def _load_raw_map(path_value: str) -> np.ndarray | None:
        if not path_value:
            return None
        path = Path(path_value)
        if not path.is_file():
            raise FileNotFoundError(f"Raw anomaly map is missing: {path}")
        with np.load(path, allow_pickle=False) as stored:
            return stored["anomaly_map"]

    @staticmethod
    def _rectified_image(source_path: Path, inspection_region, pipeline: PreprocessingPipeline | None) -> np.ndarray | None:
        if pipeline is not None:
            _prepared, rectified = pipeline.prepare_path_with_rectified(source_path)
            return rectified
        if inspection_region.enabled:
            return InspectionRegionProcessor(inspection_region).apply_path(source_path)
        return None

    @staticmethod
    def _activate_revision(run_directory: Path, revision_path: Path) -> None:
        pointer = {
            "version": ACTIVE_THRESHOLD_REVISION_VERSION,
            "revision_file": revision_path.name,
            "revision_sha256": sha256_file(revision_path),
        }
        ThresholdRevisionService._atomic_write_json(run_directory / "active_threshold_revision.json", pointer)

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            temporary_path = Path(handle.name)
        temporary_path.replace(path)