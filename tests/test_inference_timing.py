"""Tests for versioned inference timing records and aggregation."""

from __future__ import annotations

import pytest

from app.core.inference_timing import InferenceTimingRecord, timed_model_call, timing_percentiles


def test_timing_record_round_trips_and_rejects_negative_measurements() -> None:
    record = InferenceTimingRecord(model_forward_ms=1.5, end_to_end_ms=2.0, device="cpu", raw_input_size=(640, 480))

    assert InferenceTimingRecord.from_dict(record.to_dict()) == record
    with pytest.raises(ValueError, match="non-negative"):
        InferenceTimingRecord(model_forward_ms=-0.1).validate()


def test_timing_percentiles_use_deterministic_linear_interpolation() -> None:
    summary = timing_percentiles([1.0, 2.0, 3.0, 4.0, 5.0])

    assert summary["p50_ms"] == 3.0
    assert summary["p95_ms"] == pytest.approx(4.8)
    assert summary["p99_ms"] == pytest.approx(4.96)


def test_cuda_model_timing_uses_a_synchronized_path_when_cuda_is_available() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA timing requires an available CUDA device.")

    _value, elapsed_ms = timed_model_call(lambda: torch.ones(1, device="cuda").sum(), "cuda")

    assert elapsed_ms >= 0