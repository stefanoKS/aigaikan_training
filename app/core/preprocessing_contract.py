"""Canonical preprocessing-v2 serialization, hashing, and sidecar persistence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from app.models.image_preprocessing import ImagePreprocessingConfig
from app.models.preprocessing_config import PreprocessingConfig, ResolvedPreprocessingPlan


def canonical_preprocessing_json(config: PreprocessingConfig) -> str:
    """Return canonical project policy bytes used for SHA-256 identities."""
    return json.dumps(config.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def preprocessing_hash(config: PreprocessingConfig) -> str:
    """Return the deterministic SHA-256 identity for a project policy."""
    return _sha256(canonical_preprocessing_json(config))


def canonical_image_preprocessing_json(config: ImagePreprocessingConfig) -> str:
    """Return canonical profile bytes reproducible by the reference preprocessing runner."""
    return json.dumps(config.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def image_preprocessing_hash(config: ImagePreprocessingConfig) -> str:
    """Return the semantic SHA-256 identity of the image-operation profile."""
    return _sha256(canonical_image_preprocessing_json(config))


def canonical_resolved_preprocessing_json(plan: ResolvedPreprocessingPlan) -> str:
    """Return canonical model-ready plan bytes used for run compatibility."""
    return json.dumps(plan.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def resolved_preprocessing_hash(plan: ResolvedPreprocessingPlan) -> str:
    """Return the deterministic SHA-256 identity for one resolved run plan."""
    return _sha256(canonical_resolved_preprocessing_json(plan))


def write_preprocessing_config(path: Path, config: PreprocessingConfig) -> Path:
    """Atomically persist a canonical project preprocessing policy."""
    return _atomic_write(path, canonical_preprocessing_json(config))


def write_image_preprocessing_config(path: Path, config: ImagePreprocessingConfig) -> Path:
    """Atomically persist a standalone full profile for a run or deployment bundle."""
    import cv2
    import numpy as np

    payload = config.to_dict()
    payload["implementation"] = {
        "component": "app.core.image_preprocessor.ImagePreprocessor",
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
    }
    return _atomic_write(path, json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")))


def read_preprocessing_config(path: Path) -> PreprocessingConfig:
    """Load an internally consistent project preprocessing sidecar."""
    return PreprocessingConfig.from_dict(_read_payload(path, "Preprocessing configuration"))


def read_image_preprocessing_config(path: Path) -> ImagePreprocessingConfig:
    """Load one strict versioned image-operation profile."""
    return ImagePreprocessingConfig.from_dict(_read_payload(path, "Image preprocessing configuration"))


def write_resolved_preprocessing_plan(path: Path, plan: ResolvedPreprocessingPlan) -> Path:
    """Atomically persist the exact model-ready v2 run plan."""
    return _atomic_write(path, canonical_resolved_preprocessing_json(plan))


def read_resolved_preprocessing_plan(path: Path) -> ResolvedPreprocessingPlan:
    """Load an internally consistent run preprocessing sidecar."""
    return ResolvedPreprocessingPlan.from_dict(_read_payload(path, "Resolved preprocessing plan"))


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _read_payload(path: Path, description: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object.")
    return payload


def _atomic_write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as handle:
        handle.write(content)
        temporary_path = Path(handle.name)
    temporary_path.replace(path)
    return path