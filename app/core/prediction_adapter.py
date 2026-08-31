"""Strict conversion of Anomalib prediction batches into application values."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class AnomalibPrediction:
    """One score and anomaly map returned by Anomalib for a source image."""

    image_path: Path
    score: float
    anomaly_map: Any


def iter_anomalib_predictions(output: Any) -> Iterator[AnomalibPrediction]:
    """Yield finite, path-addressable predictions or fail instead of guessing values."""
    for batch in _prediction_batches(output):
        paths = _as_list(_batch_value(batch, "image_path"))
        scores = _as_list(_batch_value(batch, "pred_score"))
        anomaly_maps = _as_list(_batch_value(batch, "anomaly_map"))
        if not paths or len(paths) != len(scores):
            raise ValueError("Anomalib prediction output must contain one image path and score per image.")
        for index, raw_path in enumerate(paths):
            score = _as_float(scores[index])
            if not isfinite(score):
                raise ValueError(f"Anomalib returned a non-finite score for {raw_path}")
            yield AnomalibPrediction(
                image_path=Path(str(raw_path)).expanduser().resolve(),
                score=score,
                anomaly_map=anomaly_maps[index] if index < len(anomaly_maps) else None,
            )


def _prediction_batches(output: Any) -> Iterable[Any]:
    if output is None:
        return ()
    if isinstance(output, (list, tuple)):
        return tuple(item for batch in output for item in batch) if output and isinstance(output[0], list) else output
    return (output,)


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


def _as_float(value: Any) -> float:
    if hasattr(value, "item"):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Anomalib returned a non-numeric prediction score: {value!r}") from exc