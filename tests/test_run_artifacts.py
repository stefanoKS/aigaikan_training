"""Tests for canonical checkpoint run artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.run_artifacts import (
    read_canonical_checkpoint,
    read_persisted_threshold_metadata,
    resolve_canonical_checkpoint,
    write_evaluation_revision,
    write_run_manifest,
)


class FakeEngine:
    """Expose Anomalib's chosen final checkpoint for the unit boundary."""

    def __init__(self, checkpoint_path: Path | None) -> None:
        self.best_model_path = str(checkpoint_path) if checkpoint_path else ""


def test_canonical_checkpoint_is_engine_selected_and_hash_verified(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"trusted model")
    canonical = resolve_canonical_checkpoint(FakeEngine(checkpoint))

    write_run_manifest(
        tmp_path / "run_manifest.json",
        canonical_checkpoint=canonical,
        dataset_manifest_sha256="a" * 64,
        split_counts={"final_test": {"ok": 3, "ng": 3}},
        threshold=0.5,
    )

    assert read_canonical_checkpoint(tmp_path) == canonical


def test_missing_engine_checkpoint_is_not_replaced_by_a_timestamp_search() -> None:
    with pytest.raises(RuntimeError, match="canonical final checkpoint"):
        resolve_canonical_checkpoint(FakeEngine(None))


def test_threshold_revision_preserves_the_canonical_model_hash(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"trusted model")
    canonical = resolve_canonical_checkpoint(FakeEngine(checkpoint))
    metadata = {
        "threshold_value": 0.5,
        "threshold_method": "normal_only_conformal",
        "calibration_manifest_sha256": "a" * 64,
    }
    revision_path = write_evaluation_revision(
        tmp_path,
        canonical_checkpoint=canonical,
        calibration_manifest_sha256="a" * 64,
        final_test_manifest_sha256="b" * 64,
        threshold_metadata=metadata,
    )
    write_run_manifest(
        tmp_path / "run_manifest.json",
        canonical_checkpoint=canonical,
        dataset_manifest_sha256="c" * 64,
        split_counts={"final_test": {"ok": 3, "ng": 0}},
        threshold=0.5,
        threshold_metadata={**metadata, "threshold_revision": "revision-001"},
    )

    revision = __import__("json").loads(revision_path.read_text(encoding="utf-8"))
    assert revision["canonical_checkpoint"]["sha256"] == canonical.sha256
    assert revision["threshold_metadata"]["threshold_revision"] == "revision-001"
    assert read_persisted_threshold_metadata(tmp_path)["threshold_revision"] == "revision-001"