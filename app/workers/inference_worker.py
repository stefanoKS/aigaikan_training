"""Inference worker entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from lightning.pytorch.callbacks import BasePredictionWriter

from app.core.inspection_region import InspectionRegionProcessor
from app.core.prediction_artifacts import inspection_region_metadata, save_prediction_artifacts
from app.core.prediction_adapter import PreprocessedPredictionAccumulator, iter_anomalib_predictions
from app.core.preprocessing_pipeline import PreprocessingPipeline
from app.core.result_parser import ResultParser
from app.core.run_artifacts import (
    read_canonical_checkpoint,
    read_persisted_pixel_operating_point,
    read_persisted_threshold,
    read_verified_inspection_region,
    read_verified_preprocessing_plan,
)
from app.models.prediction_result import PredictionResult
from app.models.preprocessing_config import PreprocessingTile
from app.models.training_config import TrainingConfig
from app.services.anomalib_service import AnomalibService

INFERENCE_BATCH_SIZE = 8

def configure_worker_stdio() -> None:
    """Use UTF-8 JSON Lines streams when the Windows locale is not Unicode-safe."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="backslashreplace")


def emit(message: dict[str, object]) -> None:
    """Emit a JSON line."""
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _discover_images(input_path: Path) -> tuple[Path, ...]:
    """Use Anomalib's prediction input rules for UI progress and engine parity."""
    from anomalib.data.utils import get_image_filenames

    return tuple(Path(path).expanduser().resolve() for path in get_image_filenames(input_path))


def _count_images(input_path: Path) -> int:
    """Return the count Anomalib itself will attempt to predict."""
    return len(_discover_images(input_path))


def _expected_source_path(predicted_path: Path, expected_paths: set[Path]) -> Path | None:
    """Resolve a prediction to its selected raw path, including Windows short-path aliases."""
    if predicted_path in expected_paths:
        return predicted_path
    for expected_path in expected_paths:
        try:
            if predicted_path.samefile(expected_path):
                return expected_path
        except OSError:
            continue
    return None


def _create_prediction_loader(dataset: Any, device: str) -> Any:
    """Create a Windows-safe loader that balances GPU throughput and result latency."""
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=INFERENCE_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device == "gpu",
        collate_fn=dataset.collate_fn,
    )


class InferenceResultCollector:
    """Convert streamed Anomalib batches into application results and JSON Lines events."""

    def __init__(
        self,
        total_images: int,
        expected_paths: set[Path],
        visualizations_directory: Path,
        threshold: float,
        pixel_threshold: float | None,
        region_metadata: dict[str, object],
        inspection_processor: InspectionRegionProcessor | None = None,
        preprocessing_pipeline: PreprocessingPipeline | None = None,
    ) -> None:
        self._total_images = total_images
        self._expected_paths = expected_paths
        self._visualizations_directory = visualizations_directory
        self._threshold = threshold
        self._pixel_threshold = pixel_threshold
        self._region_metadata = region_metadata
        self._inspection_processor = inspection_processor
        self._preprocessing_pipeline = preprocessing_pipeline
        self._preprocessed_accumulator: PreprocessedPredictionAccumulator | None = None
        self._preview_path_by_source: dict[Path, Path] = {}
        self.predictions: list[PredictionResult] = []
        self.predicted_paths: set[Path] = set()

    def configure_preprocessed_inputs(
        self,
        source_path_by_staged_path: dict[Path, Path],
        preprocessing_tile_by_staged_path: dict[Path, PreprocessingTile],
        preview_path_by_source: dict[Path, Path],
    ) -> None:
        """Supply temporary file mappings after preprocessing has staged its model inputs."""
        if self._preprocessing_pipeline is None:
            raise ValueError("Preprocessed inputs cannot be configured for legacy inference.")
        self._preprocessed_accumulator = PreprocessedPredictionAccumulator(
            source_path_by_staged_path,
            preprocessing_tile_by_staged_path,
            self._preprocessing_pipeline,
        )
        self._preview_path_by_source = preview_path_by_source

    def add_batch(self, output: Any) -> None:
        """Emit every source result completed by one Anomalib batch."""
        if self._preprocessing_pipeline is None:
            for anomalib_prediction in iter_anomalib_predictions(output):
                source_path = _expected_source_path(anomalib_prediction.image_path, self._expected_paths)
                if source_path is None:
                    raise ValueError(
                        f"Anomalib returned a prediction outside the selected input: {anomalib_prediction.image_path}"
                    )
                self._add_prediction(
                    source_path,
                    anomalib_prediction.score,
                    anomalib_prediction.anomaly_map,
                    anomalib_prediction.score,
                    (anomalib_prediction.score,),
                    anomalib_prediction.score_semantic,
                    self._rectified_image(source_path),
                )
            return
        if self._preprocessed_accumulator is None:
            raise ValueError("Preprocessed inference results arrived before staged inputs were configured.")
        for anomalib_prediction in self._preprocessed_accumulator.add_batch(output):
            self._add_prediction(
                anomalib_prediction.source_path,
                anomalib_prediction.score,
                anomalib_prediction.anomaly_map,
                anomalib_prediction.native_image_score,
                anomalib_prediction.native_tile_scores,
                anomalib_prediction.score_semantic,
                self._rectified_image(anomalib_prediction.source_path),
            )

    def finalize(self) -> None:
        """Verify that all selected sources produced exactly one completed result."""
        if self._preprocessed_accumulator is not None:
            self._preprocessed_accumulator.finalize()
        if self.predicted_paths != self._expected_paths:
            missing_paths = sorted(self._expected_paths - self.predicted_paths)
            missing_summary = ", ".join(str(path) for path in missing_paths[:3])
            raise ValueError(
                f"Anomalib produced {len(self.predicted_paths)} predictions for {self._total_images} input images; "
                f"missing: {missing_summary}"
            )

    def _add_prediction(
        self,
        source_path: Path,
        score: float,
        anomaly_map: Any,
        native_image_score: float | None,
        native_tile_scores: tuple[float, ...],
        score_semantic: str,
        rectified_image: np.ndarray | None,
    ) -> None:
        if source_path in self.predicted_paths:
            raise ValueError(f"Anomalib returned more than one prediction for: {source_path}")
        self.predicted_paths.add(source_path)
        artifacts = save_prediction_artifacts(
            source_path,
            anomaly_map,
            self._visualizations_directory,
            len(self.predictions),
            rectified_image=rectified_image,
            pixel_threshold=self._pixel_threshold,
        )
        prediction = PredictionResult(
            source_path=str(source_path),
            predicted_label="NG" if score >= self._threshold else "OK",
            ground_truth_label="Unknown",
            anomaly_score=score,
            threshold=self._threshold,
            original_image=str(source_path),
            anomaly_map=artifacts.heatmap_image,
            overlay_image=artifacts.overlay_image,
            native_image_score=native_image_score,
            native_tile_scores=list(native_tile_scores),
            score_semantic=score_semantic,
            continuous_anomaly_map=artifacts.continuous_anomaly_map,
            binary_mask=artifacts.binary_mask,
            contour_overlay_image=artifacts.contour_overlay_image,
            pixel_threshold=artifacts.pixel_threshold,
            pixel_threshold_comparator=artifacts.pixel_threshold_comparator,
            pixel_threshold_semantic=artifacts.pixel_threshold_semantic,
            map_display_normalization=artifacts.display_normalization or {},
            region_metadata=self._region_metadata,
        )
        self.predictions.append(prediction)
        emit({"type": "prediction", **prediction.to_dict()})
        emit({"type": "progress", "current": len(self.predictions), "total": self._total_images})

    def _rectified_image(self, source_path: Path) -> np.ndarray | None:
        if self._inspection_processor is not None and self._inspection_processor.config.enabled:
            return self._inspection_processor.apply_path(source_path)
        preview_path = self._preview_path_by_source.get(source_path)
        if preview_path is None:
            return None
        from PIL import Image

        with Image.open(preview_path) as preview:
            return np.asarray(preview.convert("RGB"))


class InferencePredictionWriter(BasePredictionWriter):
    """Bridge each completed Lightning prediction batch to the worker JSONL stream."""

    def __init__(self, collector: InferenceResultCollector) -> None:
        super().__init__(write_interval="batch")
        self._collector = collector

    def write_on_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        prediction: Any,
        batch_indices: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self._collector.add_batch(prediction)


def _stage_preprocessed_inputs(
    source_paths: tuple[Path, ...],
    preprocessing_pipeline: PreprocessingPipeline,
    destination: Path,
) -> tuple[Path, dict[Path, Path], dict[Path, PreprocessingTile], dict[Path, Path]]:
    """Prepare inputs once and keep only temporary prepared/rectified files for prediction and overlays."""
    from PIL import Image

    prepared_directory = destination / "prepared"
    preview_directory = destination / "rectified"
    prepared_directory.mkdir(parents=True)
    preview_directory.mkdir()
    source_path_by_staged_path: dict[Path, Path] = {}
    preprocessing_tile_by_staged_path: dict[Path, PreprocessingTile] = {}
    preview_path_by_source: dict[Path, Path] = {}
    for source_index, source_path in enumerate(source_paths):
        prepared_images, rectified_image = preprocessing_pipeline.prepare_path_with_rectified(source_path)
        preview_path = (preview_directory / f"{source_index:06d}.png").resolve()
        Image.fromarray(rectified_image, "RGB").save(preview_path)
        preview_path_by_source[source_path] = preview_path
        for prepared in prepared_images:
            staged_path = (
                prepared_directory / f"{source_index:06d}_tile{prepared.tile.index:02d}_{source_path.stem}.png"
            ).resolve()
            Image.fromarray(prepared.image_rgb, "RGB").save(staged_path)
            source_path_by_staged_path[staged_path] = source_path
            preprocessing_tile_by_staged_path[staged_path] = prepared.tile
    return prepared_directory, source_path_by_staged_path, preprocessing_tile_by_staged_path, preview_path_by_source


def run(run_directory: Path, input_path: Path) -> int:
    configure_worker_stdio()
    run_directory = run_directory.expanduser().resolve()
    input_path = input_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Inference input does not exist: {input_path}")
    config_path = run_directory / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Training configuration was not found in {run_directory}.")
    source_paths = _discover_images(input_path)
    total_images = len(source_paths)
    if total_images == 0:
        raise ValueError("Select an image file or a folder containing supported image files.")
    checkpoint_path = read_canonical_checkpoint(run_directory).path
    threshold = read_persisted_threshold(run_directory)
    pixel_operating_point = read_persisted_pixel_operating_point(run_directory)
    pixel_threshold = pixel_operating_point.active_threshold
    inspection_region = read_verified_inspection_region(run_directory)
    preprocessing_plan = read_verified_preprocessing_plan(run_directory)
    preprocessing_pipeline = (
        PreprocessingPipeline(inspection_region, preprocessing_plan) if preprocessing_plan is not None else None
    )
    inspection_processor = InspectionRegionProcessor(inspection_region) if preprocessing_pipeline is None else None
    output_directory = run_directory / "inference" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_directory.mkdir(parents=True, exist_ok=False)
    (output_directory / "inference_manifest.json").write_text(
        json.dumps(
            {
                "run_directory": str(run_directory),
                "input_path": str(input_path),
                "decision_threshold": threshold,
                "decision_threshold_source": "run_manifest",
                "pixel_threshold": pixel_threshold,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    visualizations_directory = output_directory / "visualizations"
    visualizations_directory.mkdir()
    config = TrainingConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8")))
    collector = InferenceResultCollector(
        total_images,
        set(source_paths),
        visualizations_directory,
        threshold,
        pixel_threshold,
        inspection_region_metadata(inspection_region),
        inspection_processor=inspection_processor,
        preprocessing_pipeline=preprocessing_pipeline,
    )
    prediction_writer = InferencePredictionWriter(collector)
    service = AnomalibService()
    components = (
        service.create_inference_components(config, output_directory, preprocessing_plan, callbacks=[prediction_writer])
        if preprocessing_plan is not None
        else service.create_inference_components(config, output_directory, callbacks=[prediction_writer])
    )
    device_note = str(components["device_note"])
    if device_note:
        emit({"type": "log", "level": "warning", "message": device_note})
    emit(
        {
            "type": "log",
            "level": "info",
            "message": (
                f"Loaded {components['definition'].display_name} on {components['device']}; "
                f"run={run_directory}; checkpoint={checkpoint_path}; input={input_path}; images={total_images}; "
                f"roi={'enabled' if inspection_region.enabled else 'disabled'}; "
                f"preprocessing={'v2' if preprocessing_plan is not None else 'legacy'}"
            ),
        }
    )
    emit({"type": "progress", "current": 0, "total": total_images})
    if preprocessing_pipeline is None:
        from anomalib.data import PredictDataset

        dataset = PredictDataset(input_path, transform=inspection_processor)
        components["engine"].predict(
            model=components["model"],
            dataloaders=_create_prediction_loader(dataset, str(components["device"])),
            return_predictions=False,
            ckpt_path=checkpoint_path,
        )
    else:
        from anomalib.data import PredictDataset

        with TemporaryDirectory(prefix="aigaikan-preprocessing-v2-") as temporary_directory:
            (
                prepared_directory,
                source_path_by_staged_path,
                preprocessing_tile_by_staged_path,
                preview_path_by_source,
            ) = _stage_preprocessed_inputs(source_paths, preprocessing_pipeline, Path(temporary_directory))
            collector.configure_preprocessed_inputs(
                source_path_by_staged_path,
                preprocessing_tile_by_staged_path,
                preview_path_by_source,
            )
            dataset = PredictDataset(prepared_directory)
            components["engine"].predict(
                model=components["model"],
                dataloaders=_create_prediction_loader(dataset, str(components["device"])),
                return_predictions=False,
                ckpt_path=checkpoint_path,
            )
    collector.finalize()
    ResultParser().export_predictions_csv(output_directory / "predictions.csv", collector.predictions)
    emit({"type": "completed", "result_dir": str(output_directory)})
    return 0


def main() -> int:
    """Run trained-model inference for an image or image folder."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    try:
        return run(Path(args.run_dir), Path(args.input))
    except Exception:
        emit(
            {
                "type": "error",
                "message": "Inference failed",
                "details": traceback.format_exc(),
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
