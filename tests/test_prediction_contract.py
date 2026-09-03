"""Tests for explicit raw and postprocessed prediction domains."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch
from anomalib.data.dataclasses.torch import ImageBatch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader

from app.core.prediction_adapter import (
    ExplicitPredictionPostProcessor,
    explicitly_postprocessed_predict,
    iter_anomalib_predictions,
)
from app.core.prediction_contract import (
    POSTPROCESSED_SCORE_SEMANTIC,
    RAW_SCORE_SEMANTIC,
    ImageThreshold,
    PredictionContract,
)


def test_prediction_contract_preserves_unbounded_raw_scores_without_comparing_them() -> None:
    contract = PredictionContract(
        raw_image_score=7.5,
        raw_anomaly_map=np.array([[7.5]], dtype=np.float32),
        postprocessed_image_score=0.75,
        postprocessed_anomaly_map=np.array([[0.75]], dtype=np.float32),
        image_threshold=ImageThreshold(0.5, POSTPROCESSED_SCORE_SEMANTIC),
        pixel_threshold=0.5,
        predicted_label="NG",
    )

    assert contract.raw_image_score == 7.5
    assert contract.predicted_label == "NG"


def test_prediction_contract_rejects_a_raw_domain_threshold() -> None:
    with pytest.raises(ValueError, match="score semantic"):
        PredictionContract(
            raw_image_score=4.0,
            raw_anomaly_map=np.array([[4.0]], dtype=np.float32),
            postprocessed_image_score=0.4,
            postprocessed_anomaly_map=np.array([[0.4]], dtype=np.float32),
            image_threshold=ImageThreshold(4.0, RAW_SCORE_SEMANTIC),
            pixel_threshold=None,
            predicted_label="OK",
        )


def test_explicit_postprocessor_snapshots_raw_values_and_normalizes_once() -> None:
    class FakePostProcessor:
        enable_normalization = True

        def __init__(self) -> None:
            self.calls = 0

        def post_process_batch(self, batch: dict[str, object]) -> None:
            self.calls += 1
            batch["pred_score"] = [float(score) / 10 for score in batch["pred_score"]]
            batch["anomaly_map"] = [np.asarray(values, dtype=np.float32) / 10 for values in batch["anomaly_map"]]

    post_processor = FakePostProcessor()
    processor = ExplicitPredictionPostProcessor(post_processor)
    batch: dict[str, object] = {
        "image_path": ["part.png"],
        "pred_score": [7.5],
        "anomaly_map": [np.array([[7.5]], dtype=np.float32)],
    }

    output = processor.postprocess(batch)
    post_processor.post_process_batch(batch)
    prediction = next(iter_anomalib_predictions(output))

    assert post_processor.calls == 1
    assert prediction.raw_image_score == 7.5
    assert prediction.postprocessed_image_score == pytest.approx(0.75)
    assert np.array_equal(prediction.raw_anomaly_map, np.array([[7.5]], dtype=np.float32))
    assert np.array_equal(prediction.postprocessed_anomaly_map, np.array([[0.75]], dtype=np.float32))


def test_explicit_postprocessor_accepts_a_copied_return_prediction_batch() -> None:
    class FakePostProcessor:
        enable_normalization = True

        def __init__(self) -> None:
            self.calls = 0

        def post_process_batch(self, batch: ImageBatch) -> None:
            self.calls += 1
            batch.pred_score = batch.pred_score / 10
            batch.anomaly_map = batch.anomaly_map / 10

    class FakeModel:
        def __init__(self) -> None:
            self.post_processor = FakePostProcessor()

    class FakeEngine:
        def predict(self, *, model, **kwargs):
            del kwargs
            batch = ImageBatch(
                image=torch.zeros((1, 3, 2, 2)),
                image_path=["part.png"],
                pred_score=torch.tensor([7.5]),
                anomaly_map=torch.tensor([[[7.5]]]),
            )
            model.post_processor.post_process_batch(batch)
            return [[deepcopy(batch)]]

    model = FakeModel()
    output = explicitly_postprocessed_predict(FakeEngine(), model=model)
    prediction = next(iter_anomalib_predictions(output))

    assert model.post_processor.calls == 1
    assert prediction.raw_image_score == pytest.approx(7.5)
    assert prediction.postprocessed_image_score == pytest.approx(0.75)


def test_lightning_writer_captures_explicitly_postprocessed_predictions_once() -> None:
    captured_scores: list[float] = []

    class FakePostProcessor(Callback):
        enable_normalization = True

        def __init__(self) -> None:
            self.calls = 0

        def post_process_batch(self, batch: dict[str, object]) -> None:
            self.calls += 1
            batch["pred_score"] = batch["pred_score"] / 10
            batch["anomaly_map"] = batch["anomaly_map"] / 10

        def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
            self.post_process_batch(outputs)

    class CapturingCollector:
        def add_batch(self, batch: object) -> None:
            captured_scores.extend(prediction.score for prediction in iter_anomalib_predictions(batch))

    class FakeModel(LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.post_processor = FakePostProcessor()

        def configure_callbacks(self):
            return [self.post_processor]

        def predict_step(self, batch, batch_idx):
            return {
                "image_path": ["part.png"],
                "pred_score": torch.tensor([7.5]),
                "anomaly_map": torch.tensor([[[7.5]]]),
            }

    from app.workers.inference_worker import InferencePredictionWriter

    model = FakeModel()
    writer = InferencePredictionWriter(CapturingCollector())
    writer.configure_postprocessor(model)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        callbacks=[writer],
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        logger=False,
    )

    trainer.predict(
        model,
        dataloaders=DataLoader([torch.tensor(index) for index in range(3)], batch_size=1),
        return_predictions=False,
    )

    assert model.post_processor.calls == 3
    assert captured_scores == [pytest.approx(0.75)] * 3