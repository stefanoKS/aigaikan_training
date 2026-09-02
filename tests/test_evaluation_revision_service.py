"""Tests for post-training threshold revisions without model retraining."""

import json
from pathlib import Path

from PIL import Image

from app.core.dataset_manifest import sha256_file
from app.core.inspection_region import inspection_region_hash, write_inspection_region
from app.models.inspection_region import InspectionRegionConfig
from app.core.run_artifacts import CanonicalCheckpoint, write_run_manifest
from app.core.threshold_calibrator import ThresholdCalibrationConfig, ThresholdMethod
from app.models.dataset_config import DatasetConfig, DatasetRole
from app.models.training_config import TrainingConfig
from app.services.evaluation_revision_service import EvaluationDirectories, EvaluationRevisionService


def _image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(path)


class _FakeEngine:
    def predict(self, *, datamodule: DatasetConfig, **_kwargs: object) -> dict[str, object]:
        paths = [
            *sorted(Path(datamodule.folders[DatasetRole.OK_TEST].path).glob("*.png")),
            *(
                sorted(Path(datamodule.folders[DatasetRole.NG_TEST].path).glob("*.png"))
                if datamodule.folders[DatasetRole.NG_TEST].path
                else []
            ),
        ]
        return {
            "image_path": [str(path.resolve()) for path in paths],
            "pred_score": [0.1 if "ok" in path.name else 0.9 for path in paths],
            "anomaly_map": [None for path in paths],
        }


class _FakeAnomalibService:
    def __init__(self) -> None:
        self.engine = _FakeEngine()

    def create_inference_components(self, _config: TrainingConfig, _output_directory: Path) -> dict[str, object]:
        return {"model": object(), "engine": self.engine}

    @staticmethod
    def create_datamodule(
        dataset: DatasetConfig,
        _config: TrainingConfig,
        *,
        calibration_mode: bool,
        inspection_region: InspectionRegionConfig | None = None,
    ) -> DatasetConfig:
        assert not calibration_mode
        assert inspection_region is not None
        return dataset


def test_reevaluation_revises_threshold_without_retraining_the_canonical_model(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"canonical-model")
    canonical = CanonicalCheckpoint(checkpoint.resolve(), sha256_file(checkpoint))
    (tmp_path / "config.json").write_text(json.dumps(TrainingConfig().to_dict()), encoding="utf-8")
    inspection_region = InspectionRegionConfig()
    write_inspection_region(tmp_path / "inspection_region.json", inspection_region)
    write_run_manifest(
        tmp_path / "run_manifest.json",
        canonical_checkpoint=canonical,
        dataset_manifest_sha256="a" * 64,
        split_counts={"final_test": {"ok": 1, "ng": 1}},
        threshold=0.5,
        extra={
            "inspection_region_hash": inspection_region_hash(inspection_region),
            "inspection_preprocessing": {
                "roi_contract_version": inspection_region.roi_contract_version,
                "metadata_file": "inspection_region.json",
                "metadata_sha256": inspection_region_hash(inspection_region),
                "source_size": [0, 0],
                "rectified_size": [0, 0],
            }
        },
    )
    _image(tmp_path / "calibration_ok" / "ok.png", (10, 10, 10))
    _image(tmp_path / "final_ok" / "ok.png", (20, 20, 20))
    _image(tmp_path / "final_ng" / "ng.png", (240, 240, 240))

    result = EvaluationRevisionService(_FakeAnomalibService()).reevaluate(
        tmp_path,
        EvaluationDirectories(
            calibration_ok=tmp_path / "calibration_ok",
            final_test_ok=tmp_path / "final_ok",
            final_test_ng=tmp_path / "final_ng",
        ),
        ThresholdCalibrationConfig(ThresholdMethod.NORMAL_ONLY_MAX),
    )

    revision = json.loads(result.revision_path.read_text(encoding="utf-8"))
    assert result.canonical_checkpoint.sha256 == canonical.sha256
    assert result.threshold_metadata["threshold_revision"] == "revision-001"
    assert revision["canonical_checkpoint"]["sha256"] == canonical.sha256
    assert result.quality_report.status == "WARNING"
    assert result.predictions_path.is_file()