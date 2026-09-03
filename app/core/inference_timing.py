"""Versioned timing records for reproducible inference latency reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from statistics import median
from time import perf_counter_ns
from typing import Callable, TypeVar

TIMING_RECORD_VERSION = 1
_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class InferenceTimingRecord:
    """Separated inference phases; values are milliseconds or ``None`` when not observed."""

    input_decode_ms: float | None = None
    roi_rectification_ms: float | None = None
    image_filter_ms: float | None = None
    padding_tiling_ms: float | None = None
    preprocess_compute_ms: float | None = None
    staging_io_ms: float | None = None
    host_to_device_ms: float | None = None
    model_forward_ms: float | None = None
    native_postprocess_ms: float | None = None
    application_postprocess_ms: float | None = None
    inference_total_ms: float | None = None
    artifact_io_ms: float | None = None
    end_to_end_ms: float | None = None
    model_load_ms: float | None = None
    device: str = ""
    dtype: str = ""
    batch_size: int = 1
    tile_count: int = 1
    raw_input_size: tuple[int, int] = (0, 0)
    rectified_size: tuple[int, int] = (0, 0)
    model_input_size: tuple[int, int] = (0, 0)
    warmup_status: str = "not_applicable"
    timing_record_version: int = TIMING_RECORD_VERSION

    def validate(self) -> None:
        if self.timing_record_version != TIMING_RECORD_VERSION:
            raise ValueError("Unsupported inference timing record version.")
        for name, value in asdict(self).items():
            if name.endswith("_ms") and value is not None and (not isfinite(float(value)) or float(value) < 0):
                raise ValueError(f"Inference timing {name} must be finite and non-negative when measured.")
        if self.batch_size <= 0 or self.tile_count <= 0:
            raise ValueError("Inference timing batch size and tile count must be positive.")
        for name, size in (
            ("raw input", self.raw_input_size),
            ("rectified", self.rectified_size),
            ("model input", self.model_input_size),
        ):
            if len(size) != 2 or any(value < 0 for value in size):
                raise ValueError(f"Inference timing {name} size is invalid.")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        payload = asdict(self)
        for key in ("raw_input_size", "rectified_size", "model_input_size"):
            payload[key] = list(payload[key])
        return payload

    @classmethod
    def from_dict(cls, payload: object) -> "InferenceTimingRecord":
        if payload in (None, ""):
            return cls()
        if not isinstance(payload, dict):
            raise ValueError("Inference timing metadata must be an object.")
        time_fields = {field_name for field_name in cls.__dataclass_fields__ if field_name.endswith("_ms")}
        values = {
            name: (None if payload.get(name) in (None, "") else float(payload[name]))
            for name in time_fields
        }
        result = cls(
            **values,
            device=str(payload.get("device", "")),
            dtype=str(payload.get("dtype", "")),
            batch_size=int(payload.get("batch_size", 1)),
            tile_count=int(payload.get("tile_count", 1)),
            raw_input_size=_size(payload.get("raw_input_size", (0, 0)), "raw_input_size"),
            rectified_size=_size(payload.get("rectified_size", (0, 0)), "rectified_size"),
            model_input_size=_size(payload.get("model_input_size", (0, 0)), "model_input_size"),
            warmup_status=str(payload.get("warmup_status", "not_applicable")),
            timing_record_version=int(payload.get("timing_record_version", TIMING_RECORD_VERSION)),
        )
        result.validate()
        return result


def timed_call(call: Callable[[], _T]) -> tuple[_T, float]:
    """Measure a synchronous CPU phase with a monotonic nanosecond clock."""
    started = perf_counter_ns()
    value = call()
    return value, (perf_counter_ns() - started) / 1_000_000


def timed_model_call(call: Callable[[], _T], device: str) -> tuple[_T, float]:
    """Measure model work with CUDA events when the selected device supports them."""
    if device.casefold() not in {"cuda", "gpu"}:
        return timed_call(call)
    try:
        import torch

        if not torch.cuda.is_available():
            return timed_call(call)
        torch.cuda.synchronize()
        started = torch.cuda.Event(enable_timing=True)
        finished = torch.cuda.Event(enable_timing=True)
        started.record()
        value = call()
        finished.record()
        finished.synchronize()
        return value, float(started.elapsed_time(finished))
    except Exception:
        return timed_call(call)


def timing_percentiles(values_ms: list[float]) -> dict[str, float]:
    """Return deterministic linear-interpolated latency percentiles and throughput inputs."""
    if not values_ms or any(not isfinite(value) or value < 0 for value in values_ms):
        raise ValueError("Timing percentiles require finite non-negative measurements.")
    ordered = sorted(values_ms)
    return {
        "p50_ms": _percentile(ordered, 0.50),
        "p95_ms": _percentile(ordered, 0.95),
        "p99_ms": _percentile(ordered, 0.99),
        "maximum_ms": ordered[-1],
        "mean_ms": sum(ordered) / len(ordered),
        "median_ms": median(ordered),
    }


def _percentile(ordered: list[float], fraction: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _size(value: object, name: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Inference timing {name} must contain two integers.")
    return int(value[0]), int(value[1])