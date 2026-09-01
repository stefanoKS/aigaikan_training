"""Real Anomalib 2.6.0 smoke coverage for the supported stock profiles."""

from __future__ import annotations

import os
from math import isfinite
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from app.core.prediction_adapter import iter_anomalib_predictions
from app.core.run_artifacts import resolve_canonical_checkpoint
from app.models.dataset_config import DatasetConfig, DatasetRole
from app.models.training_config import DeviceMode, TrainingConfig
from app.services.anomalib_service import AnomalibService, REQUIRED_ANOMALIB_VERSION
from app.services.export_service import DEFAULT_SCORE_TOLERANCE, ExportService


if os.environ.get("RUN_ANOMALIB_SMOKE") != "1":
    pytest.skip("Set RUN_ANOMALIB_SMOKE=1 to run real Anomalib integration smoke tests.", allow_module_level=True)


pytestmark = pytest.mark.anomalib_smoke


def _write_image(path: Path, background: tuple[int, int, int], defect: bool = False) -> None:
    image = Image.new("RGB", (128, 128), background)
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 12, 116, 116), outline=(30, 30, 30), width=3)
    if defect:
        draw.ellipse((44, 44, 84, 84), fill=(220, 30, 30))
    image.save(path)


def _dataset_config(root: Path) -> DatasetConfig:
    for folder_name in ("ok_train", "ok_test", "ng_test"):
        (root / folder_name).mkdir(parents=True, exist_ok=True)
    for index, background in enumerate(((110, 140, 170), (115, 145, 175), (120, 150, 180), (125, 155, 185))):
        _write_image(root / "ok_train" / f"normal_{index}.png", background)
    _write_image(root / "ok_test" / "normal_holdout.png", (118, 148, 178))
    _write_image(root / "ng_test" / "defect_holdout.png", (118, 148, 178), defect=True)

    dataset = DatasetConfig()
    dataset.folders[DatasetRole.OK_TRAIN].path = str(root / "ok_train")
    dataset.folders[DatasetRole.OK_TEST].path = str(root / "ok_test")
    dataset.folders[DatasetRole.NG_TEST].path = str(root / "ng_test")
    return dataset


@pytest.mark.parametrize(
    "model_id",
    ("patchcore", "padim", "dinomaly_dinov2", "dinomaly_dinov3"),
)
def test_stock_model_train_export_reload_score_and_decision_parity(model_id: str, tmp_path: Path) -> None:
    """Exercise stock construction, fit, predict, Torch export, and deployment parity."""
    import anomalib
    from anomalib.deploy import TorchInferencer
    from anomalib.engine import Engine

    assert anomalib.__version__ == REQUIRED_ANOMALIB_VERSION
    dataset = _dataset_config(tmp_path / "dataset")
    config = TrainingConfig(model_name=model_id, device=DeviceMode.CPU, batch_size=1)
    service = AnomalibService()
    components = service.create_components(
        dataset=dataset,
        config=config,
        run_directory=tmp_path / "run",
        calibration_mode=True,
    )
    assert type(components["model"]).__name__ in {"Patchcore", "Padim", "Dinomaly"}
    if config.is_dinomaly:
        assert config.max_epochs > 1
        engine = Engine(
            accelerator="cpu",
            devices=1,
            max_epochs=2,
            max_steps=1,
            default_root_dir=tmp_path / "smoke",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )
    else:
        engine = components["engine"]

    engine.fit(model=components["model"], datamodule=components["datamodule"])
    checkpoint = resolve_canonical_checkpoint(engine)
    predictions = list(
        iter_anomalib_predictions(
            engine.predict(
                model=components["model"],
                datamodule=components["datamodule"],
                return_predictions=True,
                ckpt_path=checkpoint.path,
            )
        )
    )
    assert predictions
    assert all(isfinite(prediction.score) for prediction in predictions)

    exported_path = engine.export(
        model=components["model"],
        export_type="torch",
        export_root=tmp_path / "export",
        model_file_name=f"{model_id}_smoke",
        input_size=None,
        ckpt_path=checkpoint.path,
    )
    assert exported_path is not None and exported_path.is_file() and exported_path.stat().st_size > 0

    inferencer = TorchInferencer(path=exported_path, device="cpu")
    threshold = max(prediction.score for prediction in predictions) + (2 * DEFAULT_SCORE_TOLERANCE)
    for expected in predictions:
        deployed_score = ExportService._deployment_score(inferencer.predict(expected.image_path))
        assert isfinite(deployed_score)
        assert deployed_score == pytest.approx(expected.score, abs=DEFAULT_SCORE_TOLERANCE)
        assert (deployed_score >= threshold) is (expected.score >= threshold)