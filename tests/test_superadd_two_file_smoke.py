"""Opt-in real SuperADD two-file deployment smoke test."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from app.core.deployment_package import DEPLOYMENT_METADATA_FILENAME, DEPLOYMENT_MODEL_FILENAME, DeploymentPackage
from app.core.model_registry import ModelRegistry, ModelSupportLevel
from app.core.prediction_contract import SUPERADD_NATIVE_IMAGE_SCORE_SEMANTIC
from app.core.result_parser import ResultParser
from app.core.run_artifacts import read_persisted_threshold_metadata
from app.models.training_config import TrainingConfig
from app.services.export_service import ExportService, ModelExportFormat
from app.services.threshold_revision_service import ThresholdRevisionService


class _IntegrationSuperADDRegistry(ModelRegistry):
    """Permit the real integration test to exercise the guarded SuperADD export path only."""

    def get(self, identifier: str):
        definition = super().get(identifier)
        if definition.key != "super_add":
            return definition
        return replace(
            definition,
            supports_export=True,
            support_level=ModelSupportLevel.TORCH_EXPORT_VALIDATED,
        )


@pytest.mark.anomalib_smoke
def test_real_superadd_checkpoint_exports_and_reloads_with_trainer_parity(tmp_path: Path, monkeypatch) -> None:
    """Export a real completed SuperADD run without fake models, engine, or inference output."""
    run_value = os.environ.get("SUPERADD_EXPORT_RUN")
    if not run_value:
        pytest.skip("Set SUPERADD_EXPORT_RUN to a completed SuperADD run to execute real export/reload parity.")
    run_directory = Path(run_value).expanduser().resolve()
    config = TrainingConfig.from_dict(json.loads((run_directory / "config.json").read_text(encoding="utf-8")))
    assert config.is_super_add
    active_revision = ThresholdRevisionService.read_active_revision(run_directory)
    persisted_threshold = read_persisted_threshold_metadata(run_directory)
    expected_predictions = ResultParser().read_predictions_csv(active_revision.predictions_path) if active_revision else (
        ResultParser().read_training_run(run_directory / "results.json").predictions
    )
    assert expected_predictions, "The real integration run must retain active final-test predictions."
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("TRUST_REMOTE_CODE", "1")
    report = ExportService(model_registry=_IntegrationSuperADDRegistry()).export_model(
        run_directory,
        tmp_path / "exports",
        [ModelExportFormat.TORCH],
    )

    assert report.package_directory is not None
    assert {path.name for path in report.package_directory.iterdir()} == {
        DEPLOYMENT_MODEL_FILENAME,
        DEPLOYMENT_METADATA_FILENAME,
    }
    metadata = json.loads((report.package_directory / DEPLOYMENT_METADATA_FILENAME).read_text(encoding="utf-8"))
    assert metadata["decision"]["threshold"] == (
        active_revision.image_operating_point.threshold if active_revision else persisted_threshold["threshold_value"]
    )
    assert metadata["decision"]["threshold_revision_id"] == (
        active_revision.revision_path.stem if active_revision else "calibrated"
    )
    assert metadata["decision"]["operator_note"] == (active_revision.operator_note if active_revision else "")
    assert metadata["decision"]["score_semantic"] == SUPERADD_NATIVE_IMAGE_SCORE_SEMANTIC
    assert metadata["model"]["export_adapter"] == "superadd_native_v1"
    assert metadata["validation"]["status"] == "PASS"

    device = "cuda" if config.superadd_precision == "float16" else "cpu"
    deployment = DeploymentPackage.load(report.package_directory, device=device)
    tolerance = float(metadata["validation"]["score_tolerance"])
    for expected in expected_predictions:
        assert expected.raw_anomaly_map, "SuperADD trainer parity requires the saved raw continuous map."
        with Image.open(expected.source_path) as image:
            raw = np.asarray(image if image.mode == "L" else image.convert("RGB"))
        with np.load(expected.raw_anomaly_map, allow_pickle=False) as stored:
            trainer_map = stored["anomaly_map"]
        result = deployment.predict(raw)
        assert result.score_semantic == SUPERADD_NATIVE_IMAGE_SCORE_SEMANTIC
        assert np.isfinite(result.decision_score)
        assert np.isfinite(result.anomaly_map).all()
        assert result.decision_score == pytest.approx(expected.anomaly_score, abs=tolerance)
        assert result.anomaly_map.shape == trainer_map.shape
        np.testing.assert_allclose(result.anomaly_map, trainer_map, rtol=0, atol=tolerance)
        assert result.is_ng is (expected.predicted_label.upper() == "NG")