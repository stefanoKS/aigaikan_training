"""Inference worker entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from app.core.result_parser import ResultParser
from app.models.prediction_result import PredictionResult
from app.models.training_config import TrainingConfig
from app.services.anomalib_service import AnomalibService

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def emit(message: dict[str, object]) -> None:
    """Emit a JSON line."""
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _find_checkpoint(run_directory: Path) -> Path:
    checkpoints = sorted(
        run_directory.glob("**/weights/lightning/*.ckpt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not checkpoints:
        raise FileNotFoundError(f"No Lightning checkpoint was found in {run_directory}.")
    return checkpoints[0]


def _count_images(input_path: Path) -> int:
    if input_path.is_file():
        return int(input_path.suffix.lower() in IMAGE_SUFFIXES)
    return sum(1 for path in input_path.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def _batch_value(batch: Any, name: str) -> Any:
    if isinstance(batch, dict):
        return batch.get(name)
    return getattr(batch, name, None)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if getattr(value, "ndim", 0) == 0:
            return [value]
        return list(value)
    return [value]


def _at(values: list[Any], index: int) -> Any:
    return values[index] if index < len(values) else None


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except (TypeError, ValueError):
        return default


def _threshold_from_model(model: Any) -> float:
    post_processor = getattr(model, "post_processor", None)
    for name in ("image_threshold", "pixel_threshold"):
        threshold = getattr(post_processor, name, None)
        value = getattr(threshold, "value", threshold)
        threshold_value = _as_float(value, default=-1.0)
        if 0.0 <= threshold_value <= 1.0:
            return threshold_value
    return 0.5


def _save_visualizations(source_path: Path, anomaly_map: Any, output_directory: Path, index: int) -> tuple[str, str]:
    if anomaly_map is None:
        return "", ""
    import numpy as np
    from PIL import Image

    values = anomaly_map.detach().float().cpu().numpy() if hasattr(anomaly_map, "detach") else np.asarray(anomaly_map)
    while values.ndim > 2:
        values = values[0]
    if values.ndim != 2 or values.size == 0:
        return "", ""
    values = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lower, upper = float(values.min()), float(values.max())
    normalized = np.zeros_like(values) if upper <= lower else (values - lower) / (upper - lower)
    heatmap = np.stack(
        (
            np.clip(1.8 * normalized, 0.0, 1.0),
            np.clip(1.8 * (1.0 - np.abs(normalized - 0.5) * 2.0), 0.0, 1.0),
            np.clip(1.8 * (1.0 - normalized), 0.0, 1.0),
        ),
        axis=-1,
    )
    original = Image.open(source_path).convert("RGB")
    heatmap_image = Image.fromarray((heatmap * 255).astype(np.uint8), "RGB").resize(
        original.size,
        Image.Resampling.BILINEAR,
    )
    heatmap_path = output_directory / f"{index:04d}_heatmap.png"
    overlay_path = output_directory / f"{index:04d}_overlay.png"
    heatmap_image.save(heatmap_path)
    Image.blend(original, heatmap_image, 0.45).save(overlay_path)
    return str(heatmap_path), str(overlay_path)


def _prediction_batches(output: Any) -> Iterable[Any]:
    for item in output or []:
        if isinstance(item, list):
            yield from item
        else:
            yield item


def run(run_directory: Path, input_path: Path) -> int:
    run_directory = run_directory.expanduser().resolve()
    input_path = input_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Inference input does not exist: {input_path}")
    config_path = run_directory / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Training configuration was not found in {run_directory}.")
    total_images = _count_images(input_path)
    if total_images == 0:
        raise ValueError("Select an image file or a folder containing supported image files.")
    checkpoint_path = _find_checkpoint(run_directory)
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
    emit({"type": "log", "level": "info", "message": f"Loaded {components['definition'].display_name} on {components['device']}"})
    emit({"type": "progress", "current": 0, "total": total_images})
    output = components["engine"].predict(
        model=components["model"],
        data_path=input_path,
        return_predictions=True,
        ckpt_path=checkpoint_path,
    )
    predictions: list[PredictionResult] = []
    threshold = _threshold_from_model(components["model"])
    for batch in _prediction_batches(output):
        paths = _as_list(_batch_value(batch, "image_path"))
        scores = _as_list(_batch_value(batch, "pred_score"))
        labels = _as_list(_batch_value(batch, "pred_label"))
        anomaly_maps = _as_list(_batch_value(batch, "anomaly_map"))
        for index, raw_path in enumerate(paths):
            source_path = Path(str(raw_path)).resolve()
            score = _as_float(_at(scores, index))
            predicted_label = "NG" if _as_float(_at(labels, index)) >= 0.5 else "OK"
            heatmap_path, overlay_path = _save_visualizations(
                source_path,
                _at(anomaly_maps, index),
                visualizations_directory,
                len(predictions),
            )
            prediction = PredictionResult(
                source_path=str(source_path),
                predicted_label=predicted_label,
                ground_truth_label="Unknown",
                anomaly_score=score,
                threshold=threshold,
                original_image=str(source_path),
                anomaly_map=heatmap_path,
                overlay_image=overlay_path,
            )
            predictions.append(prediction)
            emit({"type": "prediction", **prediction.to_dict()})
            emit({"type": "progress", "current": len(predictions), "total": total_images})
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
