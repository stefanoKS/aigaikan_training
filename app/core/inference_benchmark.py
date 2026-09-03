"""Persistent batch-one checkpoint benchmarking without deployment-export claims."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
import csv
import hashlib
import json
from math import isfinite
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import numpy as np
from PIL import Image

from app.core.decision_score import require_matching_score_semantic, resolve_decision_score
from app.core.inference_timing import InferenceTimingRecord, timed_call, timed_model_call, timing_percentiles
from app.core.preprocessing_pipeline import PreprocessingPipeline
from app.core.run_artifacts import (
    read_canonical_checkpoint,
    read_persisted_threshold,
    read_persisted_threshold_metadata,
    read_verified_inspection_region,
    read_verified_preprocessing_plan,
)
from app.models.dataset_config import SUPPORTED_IMAGE_EXTENSIONS
from app.models.training_config import DeviceMode, TrainingConfig
from app.services.anomalib_service import AnomalibService
from app.services.threshold_revision_service import ThresholdRevisionService

BENCHMARK_VERSION = 1


class BenchmarkMode(StrEnum):
    """Where a measured frame begins."""

    CAMERA_EQUIVALENT = "camera-equivalent"
    FILE_END_TO_END = "file-end-to-end"


class BenchmarkCancelled(RuntimeError):
    """Raised when the caller cancels a benchmark between frames."""


@dataclass(frozen=True, slots=True)
class BenchmarkRequest:
    """One deterministic, batch-one benchmark request."""

    mode: BenchmarkMode = BenchmarkMode.CAMERA_EQUIVALENT
    warmup_frames: int = 20
    measured_frames: int = 200
    target_fps: float = 10.0
    reserve_percent: float = 20.0

    def validate(self) -> None:
        if self.warmup_frames < 0 or self.measured_frames <= 0:
            raise ValueError("Warmup frames must be non-negative and measured frames must be positive.")
        if not isfinite(self.target_fps) or self.target_fps <= 0:
            raise ValueError("Target FPS must be finite and positive.")
        if not isfinite(self.reserve_percent) or not 0 <= self.reserve_percent < 100:
            raise ValueError("Safety reserve percent must be at least zero and less than 100.")


@dataclass(frozen=True, slots=True)
class IndustrialDeadlineAssessment:
    """P95 compute-latency decision against an operator-selected frame budget."""

    target_fps: float
    reserve_percent: float
    frame_period_ms: float
    allowed_compute_budget_ms: float
    passed: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "target_fps": self.target_fps,
            "reserve_percent": self.reserve_percent,
            "frame_period_ms": self.frame_period_ms,
            "allowed_compute_budget_ms": self.allowed_compute_budget_ms,
            "pass": self.passed,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Portable benchmark result with steady-state and cold-start data separated."""

    metadata: dict[str, object]
    timing: dict[str, dict[str, float | int | None]]
    measured_steady_state_fps: float
    conservative_p95_fps: float
    file_source_p95_fps: float | None
    deadline: IndustrialDeadlineAssessment

    def to_dict(self) -> dict[str, object]:
        return {
            "benchmark_version": BENCHMARK_VERSION,
            "notice": "Checkpoint benchmark only; this is not a validated deployment export.",
            "metadata": self.metadata,
            "timing": self.timing,
            "measured_steady_state_fps": self.measured_steady_state_fps,
            "conservative_p95_fps": self.conservative_p95_fps,
            "file_source_p95_fps": self.file_source_p95_fps,
            "deadline": self.deadline.to_dict(),
        }


class CheckpointBenchmarkRunner:
    """Keep one completed-run checkpoint resident for deterministic batch-one measurements."""

    def __init__(
        self,
        *,
        model: Any,
        config: TrainingConfig,
        pipeline: PreprocessingPipeline | None,
        threshold: float,
        threshold_semantic: str,
        device: str,
        model_load_ms: float,
        metadata: Mapping[str, object],
    ) -> None:
        if not config.is_super_add:
            raise ValueError("Industrial checkpoint benchmarking currently supports completed SuperADD runs only.")
        if pipeline is not None and pipeline.plan.tiled:
            raise ValueError("SuperADD checkpoint benchmarking does not permit external preprocessing tiling.")
        self.model = model
        self.config = config
        self.pipeline = pipeline
        self.threshold = threshold
        self.threshold_semantic = threshold_semantic
        self.device = device
        self.model_load_ms = model_load_ms
        self.metadata = dict(metadata)

    @classmethod
    def load_completed_run(
        cls,
        run_directory: Path,
        device: str,
        *,
        anomalib_service: AnomalibService | None = None,
    ) -> "CheckpointBenchmarkRunner":
        """Verify a completed run and load its exact checkpoint once for benchmark use."""
        run_directory = run_directory.expanduser().resolve()
        if device not in {"cpu", "cuda"}:
            raise ValueError("Benchmark device must be cpu or cuda.")
        config_path = run_directory / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Training configuration not found: {config_path}")
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Training configuration must be a JSON object.")
        config = TrainingConfig.from_dict(payload)
        config.device = DeviceMode.CUDA if device == "cuda" else DeviceMode.CPU
        config.validate()
        checkpoint = read_canonical_checkpoint(run_directory)
        inspection_region = read_verified_inspection_region(run_directory)
        plan = read_verified_preprocessing_plan(run_directory)
        pipeline = PreprocessingPipeline(inspection_region, plan) if plan is not None else None
        active_revision = ThresholdRevisionService.read_active_revision(run_directory)
        threshold_metadata = read_persisted_threshold_metadata(run_directory)
        threshold = active_revision.image_operating_point.threshold if active_revision is not None else read_persisted_threshold(run_directory)
        threshold_semantic = (
            active_revision.image_operating_point.score_semantic
            if active_revision is not None
            else str(threshold_metadata.get("score_semantic", ""))
        )
        service = anomalib_service or AnomalibService()
        definition = service.model_registry.get(config.model_name)
        resolved_device = service.resolve_device(config.device)
        service._validate_superadd_precision(config, resolved_device)
        started = perf_counter_ns()
        model = _load_checkpoint_model(service, definition, config, plan, checkpoint.path, device)
        model_load_ms = (perf_counter_ns() - started) / 1_000_000
        memory_bank_shape, memory_bank_dtype = memory_bank_metadata(model)
        metadata = {
            "run_directory": str(run_directory),
            "checkpoint_sha256": checkpoint.sha256,
            "active_threshold_revision": active_revision.revision_path.stem if active_revision is not None else "calibrated",
            "backbone": config.superadd_backbone_name,
            "model_precision": config.superadd_precision,
            "input_color_order": "RGB",
            "input_dtype": "uint8",
            "roi_hash": _run_manifest_value(run_directory, "inspection_region_hash"),
            "preprocessing_hash": _run_manifest_value(run_directory, "preprocessing_contract", "metadata_sha256"),
            "rectified_size": list(plan.rectified_size) if plan is not None else list(inspection_region.rectified_size()),
            "prepared_canvas_size": list(plan.model_input_size) if plan is not None else [],
            "patch_size": 448,
            "patch_overlap": 16,
            "memory_bank_shape": list(memory_bank_shape),
            "memory_bank_dtype": memory_bank_dtype,
            "device": _device_metadata(device),
        }
        return cls(
            model=model,
            config=config,
            pipeline=pipeline,
            threshold=threshold,
            threshold_semantic=threshold_semantic,
            device=device,
            model_load_ms=model_load_ms,
            metadata=metadata,
        )

    def run(
        self,
        input_path: Path,
        request: BenchmarkRequest,
        *,
        cancelled: Callable[[], bool] | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> BenchmarkResult:
        """Warm a persistent model then report only measured, batch-one steady-state frames."""
        request.validate()
        paths = benchmark_image_paths(input_path)
        if not paths:
            raise ValueError("Benchmark input must be one image or a folder containing supported images.")
        preloaded = [_read_rgb(path) for path in paths] if request.mode is BenchmarkMode.CAMERA_EQUIVALENT else []
        self._prepare_model()
        for index in range(request.warmup_frames):
            self._ensure_not_cancelled(cancelled)
            source = preloaded[index % len(preloaded)] if preloaded else _read_rgb(paths[index % len(paths)])
            self._measure_frame(source, input_decode_ms=0.0)
        self._synchronize()
        records: list[InferenceTimingRecord] = []
        started = perf_counter_ns()
        for index in range(request.measured_frames):
            self._ensure_not_cancelled(cancelled)
            path = paths[index % len(paths)]
            if request.mode is BenchmarkMode.FILE_END_TO_END:
                source, decode_ms = timed_call(lambda path=path: _read_rgb(path))
            else:
                source, decode_ms = preloaded[index % len(preloaded)], 0.0
            records.append(self._measure_frame(source, input_decode_ms=decode_ms))
            if progress is not None:
                progress(index + 1, request.measured_frames)
        self._synchronize()
        wall_ms = (perf_counter_ns() - started) / 1_000_000
        return self._result(paths, request, records, wall_ms)

    def measure_rgb(self, source_rgb: np.ndarray) -> InferenceTimingRecord:
        """Measure one already-available RGB camera frame without decode, artifact I/O, or batch amortization."""
        self._prepare_model()
        return self._measure_frame(source_rgb, input_decode_ms=0.0)

    def _measure_frame(self, source_rgb: np.ndarray, *, input_decode_ms: float) -> InferenceTimingRecord:
        source = _validate_rgb(source_rgb)
        started = perf_counter_ns()
        if self.pipeline is None:
            raise ValueError("Benchmarking a historical run without a frozen preprocessing plan is not supported.")
        prepared, rectified, preprocessing = self.pipeline.prepare_array_with_timing(source)
        if len(prepared) != 1:
            raise ValueError("SuperADD benchmark requires one prepared input per source frame.")
        with _torch_inference_mode():
            host_tensor, transform_ms = timed_call(lambda: _apply_anomalib_transform(self.model, prepared[0].image_rgb))
            device_tensor: Any
            if self.device == "cuda":
                device_tensor, host_to_device_ms = timed_model_call(lambda: host_tensor.to("cuda", non_blocking=False), "cuda")
            else:
                device_tensor, host_to_device_ms = host_tensor, None
            raw_output, model_forward_ms = timed_model_call(lambda: self.model(device_tensor), self.device)
            raw_score = _finite_scalar(_value(raw_output, "pred_score"), "raw native image score")
            output, native_postprocess_ms = timed_model_call(
                lambda: _native_postprocess(self.model, raw_output), self.device
            )
            decision, decision_postprocess_ms = timed_call(
                lambda: resolve_decision_score(
                    self.pipeline.plan,
                    postprocessed_image_score=_finite_scalar(_value(output, "pred_score"), "postprocessed image score"),
                    raw_image_score=raw_score,
                )
            )
        require_matching_score_semantic(decision, self.threshold_semantic)
        _is_ng = decision.value >= self.threshold
        del _is_ng
        preprocess_total_ms = float(preprocessing["preprocess_total_ms"]) + transform_ms
        model_pipeline_ms = sum(value for value in (host_to_device_ms, model_forward_ms, native_postprocess_ms, decision_postprocess_ms) if value is not None)
        end_to_end_compute_ms = preprocess_total_ms + model_pipeline_ms
        file_source_end_to_end_ms = input_decode_ms + end_to_end_compute_ms
        wall_end_to_end_ms = (perf_counter_ns() - started) / 1_000_000
        memory_bank_shape, memory_bank_dtype = memory_bank_metadata(self.model)
        return InferenceTimingRecord(
            input_decode_ms=input_decode_ms,
            roi_rectification_ms=float(preprocessing["roi_rectification_ms"]),
            image_filter_ms=float(preprocessing["image_filter_ms"]),
            padding_tiling_ms=float(preprocessing["padding_tiling_ms"]),
            padding_ms=float(preprocessing["padding_ms"]),
            anomalib_transform_ms=transform_ms,
            preprocess_compute_ms=preprocess_total_ms,
            preprocess_total_ms=preprocess_total_ms,
            host_to_device_ms=host_to_device_ms,
            model_forward_ms=model_forward_ms,
            native_postprocess_ms=native_postprocess_ms,
            application_postprocess_ms=decision_postprocess_ms,
            decision_postprocess_ms=decision_postprocess_ms,
            inference_total_ms=model_pipeline_ms,
            model_pipeline_ms=model_pipeline_ms,
            end_to_end_compute_ms=end_to_end_compute_ms,
            file_source_end_to_end_ms=file_source_end_to_end_ms,
            artifact_io_ms=0.0,
            end_to_end_ms=wall_end_to_end_ms,
            model_load_ms=self.model_load_ms,
            device=self.device,
            input_color_order="RGB",
            input_dtype="uint8",
            model_precision=self.config.superadd_precision,
            memory_bank_dtype=memory_bank_dtype,
            memory_bank_shape=memory_bank_shape,
            batch_size=1,
            batch_wall_ms=wall_end_to_end_ms,
            amortized_batch_ms_per_image=wall_end_to_end_ms,
            true_batch_one_latency_ms=wall_end_to_end_ms,
            tile_count=1,
            raw_input_size=(source.shape[1], source.shape[0]),
            rectified_size=(rectified.shape[1], rectified.shape[0]),
            model_input_size=(prepared[0].image_rgb.shape[1], prepared[0].image_rgb.shape[0]),
            warmup_status="steady_state",
        )

    def _result(
        self,
        paths: list[Path],
        request: BenchmarkRequest,
        records: list[InferenceTimingRecord],
        wall_ms: float,
    ) -> BenchmarkResult:
        names = (
            "input_decode_ms", "roi_rectification_ms", "image_filter_ms", "padding_ms", "preprocess_total_ms",
            "anomalib_transform_ms",
            "host_to_device_ms", "model_forward_ms", "native_postprocess_ms", "application_postprocess_ms", "decision_postprocess_ms",
            "model_pipeline_ms", "end_to_end_compute_ms", "artifact_io_ms", "file_source_end_to_end_ms", "model_load_ms",
        )
        timing = {name: _stats([getattr(record, name) for record in records]) for name in names}
        timing["model_load_ms"] = _stats([self.model_load_ms])
        timing["cpu_wall_end_to_end_ms"] = _stats([record.end_to_end_ms for record in records])
        compute_p95 = float(timing["end_to_end_compute_ms"]["p95_ms"] or 0.0)
        file_p95 = timing["file_source_end_to_end_ms"]["p95_ms"]
        memory_bank_shape, memory_bank_dtype = memory_bank_metadata(self.model)
        metadata = {
            **self.metadata,
            "mode": request.mode.value,
            "warmup_count": request.warmup_frames,
            "measured_count": request.measured_frames,
            "batch_size": 1,
            "input_manifest_sha256": input_manifest_sha256(paths),
            "input_decode_excluded_from_camera_equivalent": request.mode is BenchmarkMode.CAMERA_EQUIVALENT,
            "memory_bank_shape": list(memory_bank_shape),
            "memory_bank_dtype": memory_bank_dtype,
            "peak_cuda_memory_allocated": _cuda_memory("max_memory_allocated") if self.device == "cuda" else None,
            "peak_cuda_memory_reserved": _cuda_memory("max_memory_reserved") if self.device == "cuda" else None,
        }
        deadline = assess_industrial_deadline(request.target_fps, request.reserve_percent, compute_p95)
        return BenchmarkResult(
            metadata=metadata,
            timing=timing,
            measured_steady_state_fps=request.measured_frames * 1000 / wall_ms if wall_ms else 0.0,
            conservative_p95_fps=1000 / compute_p95 if compute_p95 else 0.0,
            file_source_p95_fps=(1000 / float(file_p95) if request.mode is BenchmarkMode.FILE_END_TO_END and file_p95 else None),
            deadline=deadline,
        )

    def _prepare_model(self) -> None:
        self.model.eval()
        if self.device == "cuda":
            try:
                import torch

                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

    def _synchronize(self) -> None:
        if self.device == "cuda":
            try:
                import torch

                torch.cuda.synchronize()
            except Exception:
                pass

    @staticmethod
    def _ensure_not_cancelled(cancelled: Callable[[], bool] | None) -> None:
        if cancelled is not None and cancelled():
            raise BenchmarkCancelled("Industrial inference benchmark cancelled.")


def assess_industrial_deadline(target_fps: float, reserve_percent: float, p95_compute_ms: float) -> IndustrialDeadlineAssessment:
    """Assess the compute P95 only; never derive an industrial decision from model-forward time."""
    request = BenchmarkRequest(target_fps=target_fps, reserve_percent=reserve_percent)
    request.validate()
    if not isfinite(p95_compute_ms) or p95_compute_ms < 0:
        raise ValueError("P95 end-to-end compute latency must be finite and non-negative.")
    frame_period_ms = 1000 / target_fps
    allowed_compute_budget_ms = frame_period_ms * (1 - reserve_percent / 100)
    passed = p95_compute_ms <= allowed_compute_budget_ms
    return IndustrialDeadlineAssessment(
        target_fps=target_fps,
        reserve_percent=reserve_percent,
        frame_period_ms=frame_period_ms,
        allowed_compute_budget_ms=allowed_compute_budget_ms,
        passed=passed,
        reason=(
            "P95 end-to-end compute latency is within the reserved frame budget."
            if passed
            else f"P95 end-to-end compute latency {p95_compute_ms:.3f} ms exceeds the allowed {allowed_compute_budget_ms:.3f} ms budget."
        ),
    )


def benchmark_image_paths(path: Path) -> list[Path]:
    """Return the deterministic non-recursive input manifest used by both modes."""
    source = path.expanduser().resolve()
    if source.is_file() and source.suffix.casefold() in SUPPORTED_IMAGE_EXTENSIONS:
        return [source]
    if not source.is_dir():
        return []
    return sorted(
        (item.resolve() for item in source.iterdir() if item.is_file() and item.suffix.casefold() in SUPPORTED_IMAGE_EXTENSIONS),
        key=lambda item: item.name.casefold(),
    )


def input_manifest_sha256(paths: Iterable[Path]) -> str:
    """Hash stable source names and contents so comparisons reject incompatible input sets."""
    digest = hashlib.sha256()
    for path in paths:
        resolved = Path(path).resolve()
        digest.update(resolved.name.encode("utf-8"))
        digest.update(_sha256_file(resolved).encode("ascii"))
    return digest.hexdigest()


def write_benchmark_json(path: Path, result: BenchmarkResult) -> Path:
    """Persist a portable benchmark document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return path


def write_benchmark_csv(path: Path, result: BenchmarkResult) -> Path:
    """Persist one wide CSV row for spreadsheet comparison."""
    payload = result.to_dict()
    row = _flatten(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(row))
        writer.writeheader()
        writer.writerow(row)
    return path


def read_benchmark_json(path: Path) -> dict[str, object]:
    """Read and minimally validate a benchmark document."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("benchmark_version") != BENCHMARK_VERSION:
        raise ValueError(f"Unsupported inference benchmark document: {path}")
    return payload


def memory_bank_metadata(model: Any) -> tuple[tuple[int, ...], str]:
    """Read actual model memory-bank dimensions without modifying the trained bank."""
    for name in ("memory_bank", "memory_bank_features", "memory_bank_embedding"):
        value = getattr(model, name, None)
        if value is not None and hasattr(value, "shape"):
            return tuple(int(size) for size in value.shape), str(getattr(value, "dtype", ""))
    return (), ""


def _load_checkpoint_model(
    service: AnomalibService,
    definition: Any,
    config: TrainingConfig,
    plan: Any,
    checkpoint_path: Path,
    device: str,
) -> Any:
    """Construct exactly one checkpoint model with the run's frozen SuperADD contract."""
    try:
        import torch

        target = torch.device(device)
        model_class = getattr(service._anomalib_models(), definition.anomalib_class_name, None)
        loader = getattr(model_class, "load_from_checkpoint", None)
        if not callable(loader):
            raise RuntimeError("The selected Anomalib model cannot load a checkpoint directly for benchmarking.")
        kwargs: dict[str, object] = {
            "backbone": config.superadd_backbone_name,
            "precision": config.superadd_precision,
            "patch_size": 448,
            "patch_overlap": 16,
        }
        if plan is not None:
            kwargs["pre_processor"] = service._create_v2_pre_processor(model_class, definition, plan)
        model = loader(str(checkpoint_path), map_location=target, **kwargs)
        model.to(target)
        model.eval()
        return model
    except Exception as exc:
        raise RuntimeError(f"Could not load the completed checkpoint for in-memory benchmarking: {exc}") from exc


def _apply_anomalib_transform(model: Any, image_rgb: np.ndarray) -> Any:
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(image_rgb)).permute(2, 0, 1).unsqueeze(0).float().div_(255)
    processor = getattr(model, "pre_processor", None)
    return processor(tensor) if callable(processor) else tensor


def _torch_inference_mode() -> Any:
    """Use eager Torch inference mode without introducing a compilation/precision optimization."""
    import torch

    return torch.inference_mode()


def _native_postprocess(model: Any, output: Any) -> Any:
    processor = getattr(model, "post_processor", None)
    if processor is None:
        return output
    method = getattr(processor, "post_process_batch", None)
    if not callable(method):
        raise ValueError("Benchmark model does not provide a usable Anomalib native postprocessor.")
    method(output)
    return output


def _value(output: Any, name: str) -> Any:
    return output.get(name) if isinstance(output, Mapping) else getattr(output, name, None)


def _finite_scalar(value: Any, name: str) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numel") and value.numel() != 1:
        raise ValueError(f"Benchmark {name} must contain exactly one value.")
    if hasattr(value, "item"):
        value = value.item()
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"Benchmark {name} must be finite.")
    return result


def _validate_rgb(value: np.ndarray) -> np.ndarray:
    image = np.asarray(value)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise ValueError("Industrial benchmark requires a non-empty uint8 RGB array.")
    return np.ascontiguousarray(image)


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _stats(values: Iterable[float | None]) -> dict[str, float | int | None]:
    actual = [float(value) for value in values if value is not None]
    if not actual:
        return {"count": 0, "mean_ms": None, "standard_deviation_ms": None, "minimum_ms": None, "p50_ms": None, "p95_ms": None, "p99_ms": None, "maximum_ms": None}
    return timing_percentiles(actual)


def _run_manifest_value(run_directory: Path, *keys: str) -> object:
    payload: object = json.loads((run_directory / "run_manifest.json").read_text(encoding="utf-8"))
    for key in keys:
        payload = payload.get(key) if isinstance(payload, Mapping) else None
    return payload


def _device_metadata(device: str) -> dict[str, object]:
    try:
        import torch

        if device != "cuda":
            return {"kind": "cpu", "torch_version": torch.__version__, "cuda_version": torch.version.cuda}
        capability = torch.cuda.get_device_capability(0)
        return {
            "kind": "cuda",
            "name": torch.cuda.get_device_name(0),
            "capability": list(capability),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        }
    except Exception:
        return {"kind": device}


def _cuda_memory(name: str) -> int | None:
    try:
        import torch

        return int(getattr(torch.cuda, name)())
    except Exception:
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flatten(value: object, prefix: str = "") -> dict[str, object]:
    if isinstance(value, Mapping):
        rows: dict[str, object] = {}
        for key, child in value.items():
            rows.update(_flatten(child, f"{prefix}{key}."))
        return rows
    if isinstance(value, list):
        return {prefix[:-1]: json.dumps(value)}
    return {prefix[:-1]: value}