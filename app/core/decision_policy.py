"""Checksummed deployment image-decision policy independent of Anomalib defaults."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from math import isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Mapping

from app.core.threshold_contract import PixelThresholdOperatingPoint

DECISION_POLICY_VERSION = 1
DECISION_COMPARATOR = ">="
DECISION_ABOVE_OR_EQUAL_LABEL = "NG"
DECISION_BELOW_LABEL = "OK"
DECISION_SOURCES = frozenset({"calibrated", "operator_override"})


@dataclass(frozen=True, slots=True)
class DecisionPolicy:
    """An immutable threshold policy bound to one deployed model and preprocessing plan."""

    threshold: float
    score_semantic: str
    source: str
    base_calibrated_threshold: float
    revision_id: str
    model_sha256: str
    preprocessing_plan_sha256: str
    pixel_operating_point: PixelThresholdOperatingPoint = field(default_factory=PixelThresholdOperatingPoint)
    operator_note: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    decision_policy_version: int = DECISION_POLICY_VERSION
    comparator: str = DECISION_COMPARATOR
    above_or_equal_label: str = DECISION_ABOVE_OR_EQUAL_LABEL
    below_label: str = DECISION_BELOW_LABEL

    def validate(self) -> None:
        """Reject unsafe defaults, incompatible semantics, and unbound policies."""
        if self.decision_policy_version != DECISION_POLICY_VERSION:
            raise ValueError("Unsupported decision policy version.")
        if not isfinite(self.threshold) or not isfinite(self.base_calibrated_threshold):
            raise ValueError("Deployment NG score thresholds must be finite.")
        if self.comparator != DECISION_COMPARATOR:
            raise ValueError("Deployment decision comparator must be >=.")
        if self.above_or_equal_label != DECISION_ABOVE_OR_EQUAL_LABEL or self.below_label != DECISION_BELOW_LABEL:
            raise ValueError("Deployment decision labels must be NG for >= threshold and OK below threshold.")
        if not self.score_semantic:
            raise ValueError("Deployment decision policy must declare a score semantic.")
        if self.source not in DECISION_SOURCES:
            raise ValueError("Deployment decision policy source is unsupported.")
        if not self.revision_id:
            raise ValueError("Deployment decision policy must declare a revision ID.")
        for field_name, value in (
            ("model SHA-256", self.model_sha256),
            ("preprocessing plan SHA-256", self.preprocessing_plan_sha256),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.casefold()):
                raise ValueError(f"Deployment {field_name} must be a SHA-256 hex digest.")
        try:
            datetime.fromisoformat(self.created_at)
        except ValueError as exc:
            raise ValueError("Deployment decision policy created_at must be ISO-8601.") from exc
        self.pixel_operating_point.validate()

    def to_dict(self) -> dict[str, object]:
        self.validate()
        payload = asdict(self)
        payload["pixel_operating_point"] = self.pixel_operating_point.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "DecisionPolicy":
        if payload.get("decision_policy_version") != DECISION_POLICY_VERSION:
            raise ValueError("Unsupported decision policy version.")
        pixel_payload = payload.get("pixel_operating_point")
        if not isinstance(pixel_payload, Mapping):
            raise ValueError("Deployment decision policy must contain a pixel operating point.")
        try:
            result = cls(
                threshold=float(payload.get("threshold")),
                score_semantic=str(payload.get("score_semantic", "")),
                source=str(payload.get("source", "")),
                base_calibrated_threshold=float(payload.get("base_calibrated_threshold")),
                revision_id=str(payload.get("revision_id", "")),
                model_sha256=str(payload.get("model_sha256", "")),
                preprocessing_plan_sha256=str(payload.get("preprocessing_plan_sha256", "")),
                pixel_operating_point=PixelThresholdOperatingPoint.from_dict(pixel_payload),
                operator_note=str(payload.get("operator_note", "")),
                created_at=str(payload.get("created_at", "")),
                decision_policy_version=int(payload.get("decision_policy_version", -1)),
                comparator=str(payload.get("comparator", "")),
                above_or_equal_label=str(payload.get("above_or_equal_label", "")),
                below_label=str(payload.get("below_label", "")),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("Deployment decision policy has invalid field values.") from exc
        result.validate()
        return result


def canonical_decision_policy_json(policy: DecisionPolicy) -> str:
    """Return canonical decision-policy bytes used for integrity checking."""
    return json.dumps(policy.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def decision_policy_hash(policy: DecisionPolicy) -> str:
    """Return SHA-256 of the exact canonical policy payload."""
    return hashlib.sha256(canonical_decision_policy_json(policy).encode("utf-8")).hexdigest()


def write_decision_policy(path: Path, policy: DecisionPolicy) -> Path:
    """Atomically write an immutable deployment policy artifact."""
    content = canonical_decision_policy_json(policy)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as handle:
        handle.write(content)
        temporary_path = Path(handle.name)
    temporary_path.replace(path)
    return path


def read_decision_policy(path: Path) -> DecisionPolicy:
    """Read a strict policy; missing, corrupt, or incompatible metadata fails closed."""
    if not path.is_file():
        raise FileNotFoundError(f"Deployment decision policy is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Deployment decision policy is not valid JSON.") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Deployment decision policy must be a JSON object.")
    return DecisionPolicy.from_dict(payload)