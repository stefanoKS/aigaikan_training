"""Tests for threshold-only artifact and label revisions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.core.inspection_region import inspection_region_hash, write_inspection_region
from app.core.prediction_artifacts import save_prediction_artifacts
from app.core.run_artifacts import CanonicalCheckpoint, write_run_manifest
from app.core.result_parser import ResultParser
from app.core.threshold_contract import ImageThresholdOperatingPoint, PixelThresholdOperatingPoint
from app.models.inspection_region import InspectionRegionConfig
from app.models.prediction_result import PredictionResult
from app.models.training_config import TrainingConfig
from app.models.training_run import TrainingRun
from app.services.threshold_revision_service import ThresholdRevisionService


def test_threshold_revision_regenerates_labels_and_masks_without_model_inference(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    source = tmp_path / "source.png"
    Image.new("RGB", (4, 4), (20, 30, 40)).save(source)
    inspection_region = InspectionRegionConfig()
    write_inspection_region(tmp_path / "inspection_region.json", inspection_region)
    (tmp_path / "config.json").write_text(json.dumps(TrainingConfig().to_dict()), encoding="utf-8")
    write_run_manifest(
        tmp_path / "run_manifest.json",
        canonical_checkpoint=CanonicalCheckpoint(checkpoint, hashlib.sha256(b"checkpoint").hexdigest()),
        dataset_manifest_sha256="a" * 64,
        split_counts={"final_test": {"ok": 1, "ng": 0}},
        threshold=0.5,
        threshold_metadata={"threshold_value": 0.5, "score_semantic": "anomalib_postprocessed_pred_score_v1"},
        extra={
            "inspection_region_hash": inspection_region_hash(inspection_region),
            "inspection_preprocessing": {
                "roi_contract_version": inspection_region.roi_contract_version,
                "metadata_file": "inspection_region.json",
                "metadata_sha256": inspection_region_hash(inspection_region),
                "source_size": [0, 0],
                "rectified_size": [0, 0],
            },
        },
    )
    artifacts = save_prediction_artifacts(
        source,
        np.array([[0.2, 0.8], [0.1, 0.9]], dtype=np.float32),
        tmp_path / "original_artifacts",
        0,
    )
    ResultParser().write_training_run(
        tmp_path / "results.json",
        TrainingRun(
            run_name="run",
            run_dir=str(tmp_path),
            model_name="PatchCore",
            device="cpu",
            predictions=[
                PredictionResult(
                    source_path=str(source),
                    original_image=str(source),
                    predicted_label="NG",
                    ground_truth_label="OK",
                    anomaly_score=0.7,
                    threshold=0.5,
                    score_semantic="anomalib_postprocessed_pred_score_v1",
                    postprocessed_image_score=0.7,
                    postprocessed_score_semantic="anomalib_postprocessed_pred_score_v1",
                    postprocessed_anomaly_map=artifacts.continuous_anomaly_map,
                    continuous_anomaly_map=artifacts.continuous_anomaly_map,
                )
            ],
        ),
    )

    canonical_results = (tmp_path / "results.json").read_bytes()
    service = ThresholdRevisionService()
    with pytest.raises(ValueError, match="score semantic"):
        service.create_revision(
            tmp_path,
            ImageThresholdOperatingPoint(0.8, "anomalib_model_raw_score_v1"),
        )

    result = service.create_revision(
        tmp_path,
        ImageThresholdOperatingPoint(0.8),
        PixelThresholdOperatingPoint(enabled=True, threshold=0.85),
    )
    revised = ResultParser().read_predictions_csv(result.predictions_path)[0]
    active = ThresholdRevisionService.read_active_revision(tmp_path)

    assert result.revision_path.name == "threshold-001.json"
    assert revised.predicted_label == "OK"
    assert revised.threshold == 0.8
    assert np.array_equal(np.asarray(Image.open(revised.binary_mask)), np.array([[0, 0], [0, 255]], dtype=np.uint8))
    assert active is not None and active.revision_path == result.revision_path
    assert (tmp_path / "results.json").read_bytes() == canonical_results

    reactivated = service.activate_revision(tmp_path, "threshold-001")

    assert reactivated.predictions_path == result.predictions_path

    (tmp_path / "active_threshold_revision.json").write_text(
        json.dumps(
            {
                "version": 1,
                "revision_file": "../results.json",
                "revision_sha256": "unused",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid revision filename"):
        ThresholdRevisionService.read_active_revision(tmp_path)


def test_revision_numbering_does_not_reuse_removed_revision_ids(tmp_path: Path) -> None:
    revisions = tmp_path / "threshold_revisions"
    revisions.mkdir()
    (revisions / "threshold-002.json").write_text("{}", encoding="utf-8")

    assert ThresholdRevisionService._next_revision_id(tmp_path) == "threshold-003"