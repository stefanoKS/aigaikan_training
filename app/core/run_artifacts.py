"""Immutable training-run artifact helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

from app.core.dataset_manifest import sha256_file


@dataclass(frozen=True, slots=True)
class CanonicalCheckpoint:
    """The one Anomalib-selected checkpoint used by all subsequent run operations."""

    path: Path
    sha256: str


def resolve_canonical_checkpoint(engine: Any) -> CanonicalCheckpoint:
    """Resolve Anomalib's chosen best checkpoint without timestamp heuristics."""
    path_value = getattr(engine, "best_model_path", None)
    if not path_value:
        raise RuntimeError("Anomalib did not report a canonical final checkpoint.")
    path = Path(path_value).expanduser().resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Anomalib reported an invalid canonical checkpoint: {path}")
    return CanonicalCheckpoint(path=path, sha256=sha256_file(path))


def write_run_manifest(
    path: Path,
    *,
    canonical_checkpoint: CanonicalCheckpoint,
    dataset_manifest_sha256: str,
    split_counts: Mapping[str, Mapping[str, int]],
    threshold: float,
    extra: Mapping[str, object] | None = None,
) -> Path:
    """Write the run contract used by testing, inference, export, and deployment."""
    if not isfinite(threshold):
        raise ValueError("The persisted decision threshold must be finite.")
    payload: dict[str, object] = {
        "canonical_checkpoint": {
            "path": str(canonical_checkpoint.path),
            "sha256": canonical_checkpoint.sha256,
        },
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "effective_split_counts": {name: dict(counts) for name, counts in split_counts.items()},
        "threshold": threshold,
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def extract_decision_threshold(model: Any) -> float:
    """Read Anomalib's calibrated image decision boundary without inventing a fallback."""
    post_processor = getattr(model, "post_processor", None)
    for name in ("image_threshold", "pixel_threshold"):
        threshold = getattr(post_processor, name, None)
        value = getattr(threshold, "value", threshold)
        if hasattr(value, "item"):
            value = value.item()
        try:
            result = float(value)
        except (TypeError, ValueError):
            continue
        if isfinite(result):
            return result
    raise RuntimeError("Anomalib did not provide a finite calibrated decision threshold.")


def read_run_manifest(run_directory: Path) -> dict[str, Any]:
    """Load the persisted run contract and reject malformed manifests."""
    manifest_path = run_directory / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Run manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Run manifest must be a JSON object: {manifest_path}")
    return payload


def read_persisted_threshold(run_directory: Path) -> float:
    """Read the calibrated threshold used to approve this specific training run."""
    value = read_run_manifest(run_directory).get("threshold")
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Run manifest does not contain a decision threshold.") from exc
    if not isfinite(threshold):
        raise ValueError("Run manifest decision threshold must be finite.")
    return threshold


def read_canonical_checkpoint(run_directory: Path) -> CanonicalCheckpoint:
    """Read and verify the persisted canonical checkpoint before downstream work."""
    payload = read_run_manifest(run_directory)
    checkpoint = payload.get("canonical_checkpoint", {})
    checkpoint_path = Path(str(checkpoint.get("path", ""))).expanduser().resolve()
    expected_hash = str(checkpoint.get("sha256", ""))
    if not checkpoint_path.is_file() or not expected_hash:
        raise ValueError("Run manifest does not contain a valid canonical checkpoint.")
    current_hash = sha256_file(checkpoint_path)
    if current_hash != expected_hash:
        raise ValueError("Canonical checkpoint hash does not match run_manifest.json.")
    return CanonicalCheckpoint(path=checkpoint_path, sha256=current_hash)