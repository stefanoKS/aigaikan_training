"""Inference worker entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.inspection_region import InspectionRegionProcessor
from app.core.prediction_adapter import iter_anomalib_predictions
from app.core.result_parser import ResultParser
from app.core.run_artifacts import read_canonical_checkpoint, read_persisted_threshold, read_verified_inspection_region
from app.models.prediction_result import PredictionResult
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
    inspection_processor = InspectionRegionProcessor(inspection_region)
    rectified_images = {
        source_path: inspection_processor.apply_path(source_path)
        for source_path in source_paths
    } if inspection_region.enabled else {}
    from anomalib.data import PredictDataset

    prediction_dataset = PredictDataset(input_path, transform=inspection_processor)
    output_directory = run_directory / "inference" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_directory.mkdir(parents=True, exist_ok=False)
    visualizations_directory = output_directory / "visualizations"
    visualizations_directory.mkdir()
    config = TrainingConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8")))
    service = AnomalibService()
    components = service.create_inference_components(config, output_directory)
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
                f"roi={'enabled' if inspection_region.enabled else 'disabled'}"
            ),
        }
    )
    emit({"type": "progress", "current": 0, "total": total_images})
    output = components["engine"].predict(
        model=components["model"],
        dataset=prediction_dataset,
        return_predictions=True,
        ckpt_path=checkpoint_path,
    )
    predictions: list[PredictionResult] = []
    expected_paths = set(source_paths)
    predicted_paths: set[Path] = set()
    for anomalib_prediction in iter_anomalib_predictions(output):
        source_path = _expected_source_path(anomalib_prediction.image_path, expected_paths)
        if source_path is None:
            raise ValueError(f"Anomalib returned a prediction outside the selected input: {anomalib_prediction.image_path}")
        if source_path in predicted_paths:
            raise ValueError(f"Anomalib returned more than one prediction for: {source_path}")
        predicted_paths.add(source_path)
        predicted_label = "NG" if anomalib_prediction.score >= threshold else "OK"
        heatmap_path, overlay_path = _save_visualizations(
            source_path,
            anomalib_prediction.anomaly_map,
            visualizations_directory,
            len(predictions),
            rectified_images.get(source_path),
        )
        prediction = PredictionResult(
            source_path=str(source_path),
            predicted_label=predicted_label,
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
