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

from app.core.inspection_region import InspectionRegionProcessor
from app.core.prediction_adapter import iter_anomalib_predictions, iter_preprocessed_predictions
from app.core.preprocessing_pipeline import PreprocessingPipeline
from app.core.result_parser import ResultParser
from app.core.run_artifacts import (
    read_canonical_checkpoint,
    read_persisted_threshold,
    read_verified_inspection_region,
    read_verified_preprocessing_plan,
)
from app.models.prediction_result import PredictionResult
from app.models.preprocessing_config import PreprocessingTile
from app.models.training_config import TrainingConfig
from app.services.anomalib_service import AnomalibService

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


def _save_visualizations(
    source_path: Path,
    anomaly_map: Any,
    output_directory: Path,
    index: int,
    rectified_image: Any | None = None,
) -> tuple[str, str]:
    if anomaly_map is None:
        return "", ""
    import numpy as np
    from PIL import Image

    values = anomaly_map.detach().float().cpu().numpy() if hasattr(anomaly_map, "detach") else np.asarray(anomaly_map)
    while values.ndim > 2:
        values = values[0]
    if values.ndim != 2 or values.size == 0:
        return "", ""
    values = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    normalized = np.clip(values, 0.0, 1.0)
    heatmap = np.stack(
        (
            np.clip(1.8 * normalized, 0.0, 1.0),
            np.clip(1.8 * (1.0 - np.abs(normalized - 0.5) * 2.0), 0.0, 1.0),
            np.clip(1.8 * (1.0 - normalized), 0.0, 1.0),
        ),
        axis=-1,
    )
    original = Image.fromarray(rectified_image, "RGB") if rectified_image is not None else Image.open(source_path).convert("RGB")
    heatmap_image = Image.fromarray((heatmap * 255).astype(np.uint8), "RGB").resize(
        original.size,
        Image.Resampling.BILINEAR,
    )
    heatmap_path = output_directory / f"{index:04d}_heatmap.png"
    overlay_path = output_directory / f"{index:04d}_overlay.png"
    heatmap_image.save(heatmap_path)
    Image.blend(original, heatmap_image, 0.45).save(overlay_path)
    return str(heatmap_path), str(overlay_path)


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
    inspection_region = read_verified_inspection_region(run_directory)
    preprocessing_plan = read_verified_preprocessing_plan(run_directory)
    preprocessing_pipeline = (
        PreprocessingPipeline(inspection_region, preprocessing_plan) if preprocessing_plan is not None else None
    )
    inspection_processor = InspectionRegionProcessor(inspection_region) if preprocessing_pipeline is None else None
    rectified_images = {
        source_path: inspection_processor.apply_path(source_path)
        for source_path in source_paths
    } if inspection_processor is not None and inspection_region.enabled else {}
    output_directory = run_directory / "inference" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_directory.mkdir(parents=True, exist_ok=False)
    visualizations_directory = output_directory / "visualizations"
    visualizations_directory.mkdir()
    config = TrainingConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8")))
    service = AnomalibService()
    components = (
        service.create_inference_components(config, output_directory, preprocessing_plan)
        if preprocessing_plan is not None
        else service.create_inference_components(config, output_directory)
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
    predictions: list[PredictionResult] = []
    expected_paths = set(source_paths)
    predicted_paths: set[Path] = set()
    if preprocessing_pipeline is None:
        from anomalib.data import PredictDataset

        output = components["engine"].predict(
            model=components["model"],
            dataset=PredictDataset(input_path, transform=inspection_processor),
            return_predictions=True,
            ckpt_path=checkpoint_path,
        )
        for anomalib_prediction in iter_anomalib_predictions(output):
            source_path = _expected_source_path(anomalib_prediction.image_path, expected_paths)
            if source_path is None:
                raise ValueError(f"Anomalib returned a prediction outside the selected input: {anomalib_prediction.image_path}")
            if source_path in predicted_paths:
                raise ValueError(f"Anomalib returned more than one prediction for: {source_path}")
            predicted_paths.add(source_path)
            heatmap_path, overlay_path = _save_visualizations(
                source_path,
                anomalib_prediction.anomaly_map,
                visualizations_directory,
                len(predictions),
                rectified_images.get(source_path),
            )
            prediction = PredictionResult(
                source_path=str(source_path),
                predicted_label="NG" if anomalib_prediction.score >= threshold else "OK",
                ground_truth_label="Unknown",
                anomaly_score=anomalib_prediction.score,
                threshold=threshold,
                original_image=str(source_path),
                anomaly_map=heatmap_path,
                overlay_image=overlay_path,
            )
            predictions.append(prediction)
            emit({"type": "prediction", **prediction.to_dict()})
            emit({"type": "progress", "current": len(predictions), "total": total_images})
    else:
        from PIL import Image
        from anomalib.data import PredictDataset

        with TemporaryDirectory(prefix="aigaikan-preprocessing-v2-") as temporary_directory:
            (
                prepared_directory,
                source_path_by_staged_path,
                preprocessing_tile_by_staged_path,
                preview_path_by_source,
            ) = _stage_preprocessed_inputs(source_paths, preprocessing_pipeline, Path(temporary_directory))
            output = components["engine"].predict(
                model=components["model"],
                dataset=PredictDataset(prepared_directory),
                return_predictions=True,
                ckpt_path=checkpoint_path,
            )
            for anomalib_prediction in iter_preprocessed_predictions(
                output,
                source_path_by_staged_path,
                preprocessing_tile_by_staged_path,
                preprocessing_pipeline,
            ):
                source_path = anomalib_prediction.source_path
                if source_path in predicted_paths:
                    raise ValueError(f"Anomalib returned more than one prediction for: {source_path}")
                predicted_paths.add(source_path)
                with Image.open(preview_path_by_source[source_path]) as preview:
                    rectified_image = np.asarray(preview.convert("RGB"))
                heatmap_path, overlay_path = _save_visualizations(
                    source_path,
                    anomalib_prediction.anomaly_map,
                    visualizations_directory,
                    len(predictions),
                    rectified_image,
                )
                prediction = PredictionResult(
                    source_path=str(source_path),
                    predicted_label="NG" if anomalib_prediction.score >= threshold else "OK",
                    ground_truth_label="Unknown",
                    anomaly_score=anomalib_prediction.score,
                    threshold=threshold,
                    original_image=str(source_path),
                    anomaly_map=heatmap_path,
                    overlay_image=overlay_path,
                )
                predictions.append(prediction)
                emit({"type": "prediction", **prediction.to_dict()})
                emit({"type": "progress", "current": len(predictions), "total": total_images})
    if predicted_paths != expected_paths:
        missing_paths = sorted(expected_paths - predicted_paths)
        missing_summary = ", ".join(str(path) for path in missing_paths[:3])
        raise ValueError(
            f"Anomalib produced {len(predicted_paths)} predictions for {total_images} input images; "
            f"missing: {missing_summary}"
        )
    ResultParser().export_predictions_csv(output_directory / "predictions.csv", predictions)
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
