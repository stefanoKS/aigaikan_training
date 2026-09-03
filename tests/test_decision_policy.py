"""Tests for immutable checksummed deployment decision policies."""

from __future__ import annotations

import pytest

from app.core.decision_policy import DecisionPolicy, decision_policy_hash, read_decision_policy, write_decision_policy
from app.core.threshold_contract import PixelThresholdOperatingPoint
from app.core.prediction_contract import SUPERADD_NATIVE_IMAGE_SCORE_SEMANTIC


def _policy() -> DecisionPolicy:
    return DecisionPolicy(
        threshold=1.7,
        score_semantic=SUPERADD_NATIVE_IMAGE_SCORE_SEMANTIC,
        source="operator_override",
        base_calibrated_threshold=0.7,
        revision_id="threshold-003",
        operator_note="line trial",
        model_sha256="a" * 64,
        preprocessing_plan_sha256="b" * 64,
        pixel_operating_point=PixelThresholdOperatingPoint(enabled=True, threshold=0.42),
    )


def test_policy_preserves_unbounded_superadd_threshold_and_round_trips(tmp_path) -> None:
    policy = _policy()
    path = write_decision_policy(tmp_path / "decision_policy.json", policy)

    assert read_decision_policy(path) == policy
    assert len(decision_policy_hash(policy)) == 64
    assert policy.to_dict()["comparator"] == ">="


@pytest.mark.parametrize("payload_update", [{"threshold": "nan"}, {"comparator": ">"}, {"score_semantic": ""}])
def test_policy_rejects_invalid_decision_contract(payload_update: dict[str, object]) -> None:
    payload = _policy().to_dict()
    payload.update(payload_update)

    with pytest.raises(ValueError):
        DecisionPolicy.from_dict(payload)


def test_missing_policy_fails_closed(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="decision policy"):
        read_decision_policy(tmp_path / "decision_policy.json")