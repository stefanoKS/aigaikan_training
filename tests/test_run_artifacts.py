"""Tests for canonical checkpoint run artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.run_artifacts import (
    extract_decision_threshold,
    read_canonical_checkpoint,
    read_persisted_pixel_operating_point,
    read_persisted_threshold_metadata,
    next_evaluation_revision_id,
    read_verified_inspection_region,
    read_verified_preprocessing_plan,
    resolve_canonical_checkpoint,
    write_evaluation_revision,
    write_run_manifest,
)
from app.core.threshold_contract import PixelThresholdOperatingPoint
from app.core.inspection_region import inspection_region_hash, write_inspection_region
from app.models.inspection_region import InspectionRegionConfig
from app.core.preprocessing_contract import resolved_preprocessing_hash, write_resolved_preprocessing_plan
from app.models.preprocessing_config import LEGACY_PREPROCESSING_CONTRACT_VERSION, PreprocessingConfig


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


def test_image_decision_threshold_does_not_fall_back_to_anomalib_pixel_threshold() -> None:
    model = type("Model", (), {"post_processor": type("PostProcessor", (), {"pixel_threshold": 0.9})()})()

    with pytest.raises(RuntimeError, match="decision threshold"):
        extract_decision_threshold(model)


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


def test_next_evaluation_revision_id_can_scope_artifacts_before_the_record_is_written(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"trusted model")
    canonical = resolve_canonical_checkpoint(FakeEngine(checkpoint))
    revision_id = next_evaluation_revision_id(tmp_path)

    revision_path = write_evaluation_revision(
        tmp_path,
        canonical_checkpoint=canonical,
        calibration_manifest_sha256="a" * 64,
        final_test_manifest_sha256="b" * 64,
        threshold_metadata={"threshold_value": 0.5},
        revision_id=revision_id,
    )

    assert revision_id == "revision-001"
    assert revision_path.name == "revision-001.json"
    assert next_evaluation_revision_id(tmp_path) == "revision-002"


def test_threshold_metadata_preserves_distinct_raw_and_deployed_thresholds(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"trusted model")
    canonical = resolve_canonical_checkpoint(FakeEngine(checkpoint))
    metadata = {
        "threshold_value": 0.5000000000000001,
        "threshold_raw": 0.5,
        "threshold_deployed": 0.5000000000000001,
        "threshold_method": "normal_only_conformal",
    }

    write_run_manifest(
        tmp_path / "run_manifest.json",
        canonical_checkpoint=canonical,
        dataset_manifest_sha256="a" * 64,
        split_counts={"final_test": {"ok": 1, "ng": 0}},
        threshold=0.5000000000000001,
        threshold_metadata=metadata,
    )

    restored = read_persisted_threshold_metadata(tmp_path)
    assert restored["threshold_raw"] == 0.5
    assert restored["threshold_deployed"] == restored["threshold_value"]


def test_persisted_pixel_operating_point_is_explicit_and_legacy_runs_remain_mask_free(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"trusted model")
    canonical = resolve_canonical_checkpoint(FakeEngine(checkpoint))
    operating_point = PixelThresholdOperatingPoint(enabled=True, threshold=2.5)
    write_run_manifest(
        tmp_path / "run_manifest.json",
        canonical_checkpoint=canonical,
        dataset_manifest_sha256="a" * 64,
        split_counts={"final_test": {"ok": 1, "ng": 0}},
        threshold=0.5,
        extra={"pixel_operating_point": operating_point.to_dict()},
    )

    restored = read_persisted_pixel_operating_point(tmp_path)
    assert restored.active_threshold == 2.5
    assert restored.to_dict()["semantic"] == "continuous_anomaly_map_gte_v1"

    legacy_directory = tmp_path / "legacy"
    legacy_directory.mkdir()
    write_run_manifest(
        legacy_directory / "run_manifest.json",
        canonical_checkpoint=canonical,
        dataset_manifest_sha256="b" * 64,
        split_counts={"final_test": {"ok": 1, "ng": 0}},
        threshold=0.5,
    )
    assert read_persisted_pixel_operating_point(legacy_directory).active_threshold is None


def test_run_roi_sidecar_must_match_its_manifest_hash(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"trusted model")
    canonical = resolve_canonical_checkpoint(FakeEngine(checkpoint))
    roi = InspectionRegionConfig()
    write_inspection_region(tmp_path / "inspection_region.json", roi)
    write_run_manifest(
        tmp_path / "run_manifest.json",
        canonical_checkpoint=canonical,
        dataset_manifest_sha256="a" * 64,
        split_counts={"final_test": {"ok": 1, "ng": 0}},
        threshold=0.5,
        extra={
            "inspection_region_hash": inspection_region_hash(roi),
            "inspection_preprocessing": {
                "roi_contract_version": roi.roi_contract_version,
                "metadata_file": "inspection_region.json",
                "metadata_sha256": inspection_region_hash(roi),
                "source_size": [0, 0],
                "rectified_size": [0, 0],
            }
        },
    )

    assert read_verified_inspection_region(tmp_path) == roi


def test_run_preprocessing_v2_sidecar_must_match_its_manifest_hash(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"trusted model")
    canonical = resolve_canonical_checkpoint(FakeEngine(checkpoint))
    plan = PreprocessingConfig(
        preprocessing_contract_version=LEGACY_PREPROCESSING_CONTRACT_VERSION
    ).resolve("dinomaly_dinov3", (639, 177))
    write_resolved_preprocessing_plan(tmp_path / "preprocessing_plan.json", plan)
    write_run_manifest(
        tmp_path / "run_manifest.json",
        canonical_checkpoint=canonical,
        dataset_manifest_sha256="a" * 64,
        split_counts={"final_test": {"ok": 1, "ng": 0}},
        threshold=0.5,
        extra={
            "preprocessing_contract": {
                "preprocessing_contract_version": 2,
                "metadata_file": "preprocessing_plan.json",
                "metadata_sha256": resolved_preprocessing_hash(plan),
                "model_id": "dinomaly_dinov3",
                "model_input_size": [640, 192],
                "score_aggregation": "max",
                "tiled": False,
            }
        },
    )

    assert read_verified_preprocessing_plan(tmp_path) == plan


def test_completed_legacy_run_has_no_v2_preprocessing_plan(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"trusted model")
    canonical = resolve_canonical_checkpoint(FakeEngine(checkpoint))
    write_run_manifest(
        tmp_path / "run_manifest.json",
        canonical_checkpoint=canonical,
        dataset_manifest_sha256="a" * 64,
        split_counts={"final_test": {"ok": 1, "ng": 0}},
        threshold=0.5,
    )

    assert read_verified_preprocessing_plan(tmp_path) is None