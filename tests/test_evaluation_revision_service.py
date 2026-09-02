"""Tests for post-training threshold revisions without model retraining."""

import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest
from PIL import Image

from app.core.dataset_manifest import sha256_file
from app.core.inspection_region import inspection_region_hash, write_inspection_region
from app.core.preprocessing_contract import resolved_preprocessing_hash, write_resolved_preprocessing_plan
from app.core.result_parser import ResultParser
from app.models.inspection_region import InspectionRegionConfig
from app.models.preprocessing_config import PreprocessingConfig
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


def test_reevaluation_replays_preprocessing_v2_and_excludes_padding_scores(tmp_path: Path, monkeypatch) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"canonical-model")
    canonical = CanonicalCheckpoint(checkpoint.resolve(), sha256_file(checkpoint))
    (tmp_path / "config.json").write_text(json.dumps(TrainingConfig().to_dict()), encoding="utf-8")
    inspection_region = InspectionRegionConfig()
    write_inspection_region(tmp_path / "inspection_region.json", inspection_region)
    plan = PreprocessingConfig().resolve("patchcore", (639, 177))
    write_resolved_preprocessing_plan(tmp_path / "preprocessing_plan.json", plan)
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
            },
            "preprocessing_contract": {
                "preprocessing_contract_version": 2,
                "metadata_file": "preprocessing_plan.json",
                "metadata_sha256": resolved_preprocessing_hash(plan),
                "project_policy_sha256": "a" * 64,
                "model_id": "patchcore",
                "model_input_size": [640, 192],
                "score_aggregation": "max",
                "tiled": False,
            },
        },
    )
    _image(tmp_path / "calibration_ok" / "ok.png", (10, 10, 10))
    _image(tmp_path / "final_ok" / "ok.png", (20, 20, 20))
    _image(tmp_path / "final_ng" / "ng.png", (240, 240, 240))
    for path in tmp_path.rglob("*.png"):
        Image.open(path).resize((639, 177)).save(path)
    source_bytes = (tmp_path / "final_ng" / "ng.png").read_bytes()
    prepared_sizes: list[tuple[int, int]] = []

    class FakePredictDataset:
        def __init__(self, path: Path) -> None:
            self.path = Path(path)

    class FakeEngine:
        def predict(self, *, dataset: FakePredictDataset, **_kwargs: object) -> dict[str, object]:
            paths = sorted(dataset.path.glob("*.png"))
            prepared_sizes.extend(Image.open(path).size for path in paths)
            anomaly_maps = []
            for path in paths:
                anomaly_map = np.zeros((192, 640), dtype=np.float32)
                anomaly_map[0, 0] = 0.9 if path.name.endswith("_ng.png") else 0.1
                anomaly_map[191, 639] = 1.0
                anomaly_maps.append(anomaly_map)
            return {
                "image_path": [str(path.resolve()) for path in paths],
                "pred_score": [1.0 for _path in paths],
                "anomaly_map": anomaly_maps,
            }

    class FakeService:
        def create_inference_components(
            self,
            _config: TrainingConfig,
            _output_directory: Path,
            received_plan: object,
        ) -> dict[str, object]:
            assert received_plan == plan
            return {"model": object(), "engine": FakeEngine()}

        def create_datamodule(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("Legacy datamodule must not be used for a preprocessing-v2 run.")

    anomalib_data = ModuleType("anomalib.data")
    anomalib_data.PredictDataset = FakePredictDataset
    monkeypatch.setitem(sys.modules, "anomalib.data", anomalib_data)

    result = EvaluationRevisionService(FakeService()).reevaluate(
        tmp_path,
        EvaluationDirectories(
            calibration_ok=tmp_path / "calibration_ok",
            final_test_ok=tmp_path / "final_ok",
            final_test_ng=tmp_path / "final_ng",
        ),
        ThresholdCalibrationConfig(ThresholdMethod.NORMAL_ONLY_MAX),
    )

    assert prepared_sizes == [(640, 192), (640, 192), (640, 192)]
    assert result.threshold_metadata["threshold_value"] == pytest.approx(0.1)
    assert sorted(prediction.anomaly_score for prediction in ResultParser().read_predictions_csv(result.predictions_path)) == pytest.approx(
        [0.1, 0.9]
    )
    assert (tmp_path / "final_ng" / "ng.png").read_bytes() == source_bytes