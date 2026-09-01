"""Reproducible dataset manifest and deterministic split helpers."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from app.models.dataset_config import DatasetConfig, DatasetRole, SUPPORTED_IMAGE_EXTENSIONS


@dataclass(frozen=True, slots=True)
class DatasetManifestRecord:
    """One source image used by a project or a persisted run."""

    path: str
    dataset_role: str
    file_size: int
    sha256: str
    width: int
    height: int
    image_mode: str


@dataclass(frozen=True, slots=True)
class EffectiveSplit:
    """The exact source files used for the three training phases."""

    training_ok: tuple[Path, ...]
    validation_ok: tuple[Path, ...]
    validation_ng: tuple[Path, ...]
    final_test_ok: tuple[Path, ...]
    final_test_ng: tuple[Path, ...]
    seed: int

    def counts(self) -> dict[str, dict[str, int]]:
        """Return UI-friendly class counts for the effective split."""
        return {
            "training": {"ok": len(self.training_ok), "ng": 0},
            "validation": {"ok": len(self.validation_ok), "ng": len(self.validation_ng)},
            "final_test": {"ok": len(self.final_test_ok), "ng": len(self.final_test_ng)},
        }

    def roles(self) -> dict[str, tuple[Path, ...]]:
        """Return the canonical manifest role mapping."""
        return {
            "training_ok": self.training_ok,
            "validation_ok": self.validation_ok,
            "validation_ng": self.validation_ng,
            "final_test_ok": self.final_test_ok,
            "final_test_ng": self.final_test_ng,
        }


@dataclass(frozen=True, slots=True)
class StagedDataset:
    """Run-local image directories passed to Anomalib without mutating source data."""

    training_config: DatasetConfig
    final_test_config: DatasetConfig
    source_path_by_staged_path: dict[Path, Path]


def collect_configured_images(config: DatasetConfig) -> dict[DatasetRole, list[Path]]:
    """Return configured image files in stable order without altering source data."""
    images: dict[DatasetRole, list[Path]] = {}
    for role, folder in config.folders.items():
        directory = folder.resolved_path()
        if directory is None or not directory.is_dir():
            continue
        images[role] = sorted(
            path.resolve()
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )
    return images


def build_effective_split(config: DatasetConfig, seed: int) -> EffectiveSplit:
    """Create reproducible disjoint partitions for training, calibration, and final testing.

    A project with only ``ok_train`` keeps disjoint normal images for training,
    calibration, and final evaluation. Genuine NG images are optional: they are
    never borrowed from final testing to calibrate a threshold.
    """
    images = collect_configured_images(config)
    ok_train = images.get(DatasetRole.OK_TRAIN, [])
    ok_test = images.get(DatasetRole.OK_TEST, [])
    ng_test = images.get(DatasetRole.NG_TEST, [])
    validation_ok = images.get(DatasetRole.OK_VALIDATION, [])
    validation_ng = images.get(DatasetRole.NG_VALIDATION, [])
    if len(ok_train) < 2:
        raise ValueError("At least two OK training images are required to create a split.")
    shuffled_ok = _shuffled(ok_train, seed)
    if ok_test:
        training_ok = tuple(shuffled_ok)
        held_out_ok = list(ok_test)
    else:
        held_out_count = _holdout_count(len(shuffled_ok))
        training_ok = tuple(shuffled_ok[:-held_out_count])
        held_out_ok = shuffled_ok[-held_out_count:]

    if validation_ok:
        final_test_ok = tuple(held_out_ok)
    else:
        final_test_ok, validation_ok = _split_for_validation(held_out_ok, seed + 1)
    if validation_ng:
        final_test_ng = tuple(ng_test)
    else:
        final_test_ng, validation_ng = _split_for_optional_validation(ng_test, seed + 2)

    split = EffectiveSplit(
        training_ok=training_ok,
        validation_ok=tuple(validation_ok),
        validation_ng=tuple(validation_ng),
        final_test_ok=tuple(final_test_ok),
        final_test_ng=tuple(final_test_ng),
        seed=seed,
    )
    validate_effective_split(split)
    return split


def validate_effective_split(split: EffectiveSplit) -> None:
    """Reject exact source-file reuse between model fitting, calibration, and test."""
    file_locations: dict[Path, list[str]] = {}
    content_locations: dict[str, list[str]] = {}
    for role, paths in split.roles().items():
        for path in paths:
            resolved_path = path.resolve()
            file_locations.setdefault(resolved_path, []).append(role)
            content_locations.setdefault(sha256_file(resolved_path), []).append(role)
    duplicate_paths = [f"{path}: {', '.join(roles)}" for path, roles in file_locations.items() if len(roles) > 1]
    duplicate_content = [
        f"SHA-256 {digest}: {', '.join(roles)}"
        for digest, roles in content_locations.items()
        if len(set(roles)) > 1
    ]
    if duplicate_paths or duplicate_content:
        details = "; ".join([*duplicate_paths, *duplicate_content])
        raise ValueError(f"Dataset leakage detected across training, validation, and final test: {details}")
    if not split.final_test_ok:
        raise ValueError("The final test split must contain at least one OK image.")
    if not split.validation_ok:
        raise ValueError("Validation must contain at least one held-out OK image for threshold calibration.")


def build_dataset_manifest(
    roles: Mapping[str, Iterable[Path]],
    project_root: Path | None = None,
) -> dict[str, object]:
    """Build an auditable, content-hashed dataset manifest for a run."""
    records: list[DatasetManifestRecord] = []
    for role, paths in roles.items():
        for path in sorted((Path(item).resolve() for item in paths), key=lambda item: str(item).lower()):
            with Image.open(path) as image:
                width, height = image.size
                image_mode = image.mode
            try:
                path_text = str(path.relative_to(project_root.resolve())) if project_root else str(path)
            except ValueError:
                path_text = str(path)
            records.append(
                DatasetManifestRecord(
                    path=path_text,
                    dataset_role=role,
                    file_size=path.stat().st_size,
                    sha256=sha256_file(path),
                    width=width,
                    height=height,
                    image_mode=image_mode,
                )
            )
    payload: dict[str, object] = {
        "records": [asdict(record) for record in records],
    }
    payload["manifest_sha256"] = hashlib.sha256(
        json.dumps(payload["records"], ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def write_dataset_manifest(path: Path, manifest: Mapping[str, object]) -> Path:
    """Write a stable JSON dataset manifest."""
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def stage_effective_split(split: EffectiveSplit, config: DatasetConfig, destination: Path) -> StagedDataset:
    """Copy a disjoint split into a run-local snapshot for Anomalib's folder API."""
    destination.mkdir(parents=True, exist_ok=False)
    mappings: dict[str, dict[Path, Path]] = {}
    for role, paths in split.roles().items():
        mappings[role] = _stage_images(paths, destination / role)

    mask_directory = config.folders[DatasetRole.MASKS].resolved_path()
    calibration_masks = _stage_matching_masks(
        mappings["validation_ng"],
        mask_directory,
        destination / "validation_masks",
    )
    final_masks = _stage_matching_masks(
        mappings["final_test_ng"],
        mask_directory,
        destination / "final_test_masks",
    )
    training_config = _staged_config(
        training_ok=mappings["training_ok"],
        evaluation_ok=mappings["validation_ok"],
        evaluation_ng=mappings["validation_ng"],
        masks=calibration_masks,
    )
    final_test_config = _staged_config(
        training_ok=mappings["training_ok"],
        evaluation_ok=mappings["final_test_ok"],
        evaluation_ng=mappings["final_test_ng"],
        masks=final_masks,
    )
    source_path_by_staged_path = {
        staged_path: source_path
        for role_mapping in mappings.values()
        for source_path, staged_path in role_mapping.items()
    }
    return StagedDataset(training_config, final_test_config, source_path_by_staged_path)


def sha256_file(path: Path) -> str:
    """Hash a file incrementally so large inspection images remain inexpensive."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shuffled(paths: Iterable[Path], seed: int) -> list[Path]:
    shuffled = sorted((Path(path).resolve() for path in paths), key=lambda path: str(path).lower())
    random.Random(seed).shuffle(shuffled)
    return shuffled


def _holdout_count(count: int) -> int:
    return min(max(1, round(count * 0.2)), count - 1)


def _split_for_validation(paths: Iterable[Path], seed: int) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    values = _shuffled(paths, seed)
    if len(values) < 2:
        raise ValueError("At least two held-out images per class are required for validation and final testing.")
    validation_count = min(max(1, len(values) // 2), len(values) - 1)
    return tuple(values[:-validation_count]), tuple(values[-validation_count:])


def _split_for_optional_validation(paths: Iterable[Path], seed: int) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Reserve NG calibration data only when doing so leaves genuine final-test data."""
    values = _shuffled(paths, seed)
    if len(values) < 2:
        return tuple(values), ()
    return _split_for_validation(values, seed)


def _stage_images(paths: Iterable[Path], destination: Path) -> dict[Path, Path]:
    destination.mkdir(parents=True, exist_ok=True)
    mapping: dict[Path, Path] = {}
    for index, source_path in enumerate(paths):
        source_path = source_path.resolve()
        staged_path = destination / f"{index:06d}_{source_path.name}"
        shutil.copy2(source_path, staged_path)
        mapping[source_path] = staged_path
    return mapping


def _stage_matching_masks(
    staged_ng_paths: Mapping[Path, Path],
    mask_directory: Path | None,
    destination: Path,
) -> dict[Path, Path]:
    if mask_directory is None or not mask_directory.is_dir():
        return {}
    source_masks = [
        path.resolve()
        for path in mask_directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    mapping: dict[Path, Path] = {}
    for ng_source, staged_ng_path in staged_ng_paths.items():
        matches = [mask_path for mask_path in source_masks if ng_source.stem in mask_path.stem]
        if len(matches) != 1:
            continue
        destination.mkdir(parents=True, exist_ok=True)
        staged_mask_path = destination / staged_ng_path.name
        shutil.copy2(matches[0], staged_mask_path)
        mapping[matches[0]] = staged_mask_path
    return mapping


def _staged_config(
    *,
    training_ok: Mapping[Path, Path],
    evaluation_ok: Mapping[Path, Path],
    evaluation_ng: Mapping[Path, Path],
    masks: Mapping[Path, Path],
) -> DatasetConfig:
    config = DatasetConfig()
    config.folders[DatasetRole.OK_TRAIN].path = str(next(iter(training_ok.values())).parent)
    config.folders[DatasetRole.OK_TEST].path = str(next(iter(evaluation_ok.values())).parent)
    if evaluation_ng:
        config.folders[DatasetRole.NG_TEST].path = str(next(iter(evaluation_ng.values())).parent)
    if masks:
        config.folders[DatasetRole.MASKS].path = str(next(iter(masks.values())).parent)
    return config