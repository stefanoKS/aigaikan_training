"""Reproducible dataset manifest and deterministic split helpers."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image

from app.core.preprocessing_pipeline import PreprocessingPipeline
from app.core.prepared_data_cache import PreparedDataCache
from app.models.dataset_config import DatasetConfig, DatasetRole, SUPPORTED_IMAGE_EXTENSIONS
from app.models.preprocessing_config import PreprocessingTile


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
    evaluation_method: str = "deterministic_partition"

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
    preprocessing_tile_by_staged_path: dict[Path, PreprocessingTile] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _StagedImage:
    """One staged model input and the original image it represents."""

    source_path: Path
    staged_path: Path
    preprocessing_tile: PreprocessingTile | None = None


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
    if validation_ok:
        calibration_ok = tuple(validation_ok)
        if ok_test:
            training_ok = tuple(shuffled_ok)
            final_test_ok = tuple(ok_test)
        else:
            training_ok, final_test_ok = _reserve_ok_holdout(shuffled_ok, 1)
    elif ok_test:
        if len(ok_test) >= 2:
            training_ok = tuple(shuffled_ok)
            final_test_ok, calibration_ok = _split_for_validation(ok_test, seed + 1)
        else:
            training_ok, calibration_ok = _reserve_ok_holdout(shuffled_ok, 1)
            final_test_ok = tuple(ok_test)
    else:
        held_out_count = max(_holdout_count(len(shuffled_ok)), 2)
        training_ok, held_out_ok = _reserve_ok_holdout(shuffled_ok, held_out_count)
        final_test_ok, calibration_ok = _split_for_validation(held_out_ok, seed + 1)
    if validation_ng:
        final_test_ng = tuple(ng_test)
    else:
        final_test_ng, validation_ng = _split_for_optional_validation(ng_test, seed + 2)

    split = EffectiveSplit(
        training_ok=training_ok,
        validation_ok=calibration_ok,
        validation_ng=tuple(validation_ng),
        final_test_ok=tuple(final_test_ok),
        final_test_ng=tuple(final_test_ng),
        seed=seed,
        evaluation_method=_evaluation_method(images),
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


def stage_effective_split(
    split: EffectiveSplit,
    config: DatasetConfig,
    destination: Path,
    preprocessing_pipeline: PreprocessingPipeline | None = None,
    prepared_data_cache: PreparedDataCache | None = None,
) -> StagedDataset:
    """Stage a disjoint split into model-ready folders without altering source data."""
    destination.mkdir(parents=True, exist_ok=False)
    if preprocessing_pipeline is not None:
        return _stage_preprocessed_split(split, config, destination, preprocessing_pipeline, prepared_data_cache)
    if prepared_data_cache is not None:
        raise ValueError("Prepared-data caching requires a resolved preprocessing pipeline.")
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


def _stage_preprocessed_split(
    split: EffectiveSplit,
    config: DatasetConfig,
    destination: Path,
    preprocessing_pipeline: PreprocessingPipeline,
    prepared_data_cache: PreparedDataCache | None,
) -> StagedDataset:
    staged_roles = {
        role: _stage_preprocessed_images(paths, destination / role, preprocessing_pipeline, prepared_data_cache)
        for role, paths in split.roles().items()
    }
    mask_directory = config.folders[DatasetRole.MASKS].resolved_path()
    calibration_masks = _stage_preprocessed_masks(
        staged_roles["validation_ng"],
        mask_directory,
        destination / "validation_masks",
        preprocessing_pipeline,
    )
    final_masks = _stage_preprocessed_masks(
        staged_roles["final_test_ng"],
        mask_directory,
        destination / "final_test_masks",
        preprocessing_pipeline,
    )
    folder_mappings = {
        role: {image.source_path: image.staged_path for image in images}
        for role, images in staged_roles.items()
    }
    training_config = _staged_config(
        training_ok=folder_mappings["training_ok"],
        evaluation_ok=folder_mappings["validation_ok"],
        evaluation_ng=folder_mappings["validation_ng"],
        masks=calibration_masks,
    )
    final_test_config = _staged_config(
        training_ok=folder_mappings["training_ok"],
        evaluation_ok=folder_mappings["final_test_ok"],
        evaluation_ng=folder_mappings["final_test_ng"],
        masks=final_masks,
    )
    source_path_by_staged_path = {
        image.staged_path: image.source_path
        for images in staged_roles.values()
        for image in images
    }
    preprocessing_tile_by_staged_path = {
        image.staged_path: image.preprocessing_tile
        for images in staged_roles.values()
        for image in images
        if image.preprocessing_tile is not None
    }
    return StagedDataset(
        training_config,
        final_test_config,
        source_path_by_staged_path,
        preprocessing_tile_by_staged_path,
    )


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


def _reserve_ok_holdout(paths: list[Path], count: int) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Reserve disjoint normal images while retaining the minimum fit set."""
    if len(paths) - count < 2:
        raise ValueError(
            "At least four OK training images are required when normal calibration and final testing "
            "must both be held out from OK training data."
        )
    return tuple(paths[:-count]), tuple(paths[-count:])


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


def _evaluation_method(images: Mapping[DatasetRole, list[Path]]) -> str:
    """Classify whether final-test evidence comes from independent configured folders."""
    has_explicit_ok_evidence = bool(images.get(DatasetRole.OK_VALIDATION)) and bool(images.get(DatasetRole.OK_TEST))
    has_ng_test = bool(images.get(DatasetRole.NG_TEST))
    has_explicit_ng_evidence = not has_ng_test or bool(images.get(DatasetRole.NG_VALIDATION))
    return "independent_explicit" if has_explicit_ok_evidence and has_explicit_ng_evidence else "deterministic_partition"


def _stage_images(paths: Iterable[Path], destination: Path) -> dict[Path, Path]:
    destination.mkdir(parents=True, exist_ok=True)
    mapping: dict[Path, Path] = {}
    for index, source_path in enumerate(paths):
        source_path = source_path.resolve()
        staged_path = destination / f"{index:06d}_{source_path.name}"
        shutil.copy2(source_path, staged_path)
        mapping[source_path] = staged_path
    return mapping


def _stage_preprocessed_images(
    paths: Iterable[Path],
    destination: Path,
    preprocessing_pipeline: PreprocessingPipeline,
    prepared_data_cache: PreparedDataCache | None,
) -> tuple[_StagedImage, ...]:
    destination.mkdir(parents=True, exist_ok=True)
    staged: list[_StagedImage] = []
    for source_index, source_path in enumerate(paths):
        source_path = source_path.resolve()
        if prepared_data_cache is None:
            prepared_tiles = preprocessing_pipeline.prepare_path(source_path)
            for prepared in prepared_tiles:
                staged_path = (destination / f"{source_index:06d}_tile{prepared.tile.index:02d}_{source_path.stem}.png").resolve()
                Image.fromarray(prepared.image_rgb, "RGB").save(staged_path)
                staged.append(_StagedImage(source_path, staged_path, prepared.tile))
            continue
        cached_tiles = prepared_data_cache.materialize(source_path)
        for tile, cached_path in zip(preprocessing_pipeline.plan.tiles, cached_tiles, strict=True):
            staged_path = (destination / f"{source_index:06d}_tile{tile.index:02d}_{source_path.stem}.png").resolve()
            shutil.copy2(cached_path, staged_path)
            staged.append(_StagedImage(source_path, staged_path, tile))
    return tuple(staged)


def _stage_preprocessed_masks(
    staged_ng_images: Iterable[_StagedImage],
    mask_directory: Path | None,
    destination: Path,
    preprocessing_pipeline: PreprocessingPipeline,
) -> dict[Path, Path]:
    if mask_directory is None or not mask_directory.is_dir():
        return {}
    source_masks = [
        path.resolve()
        for path in mask_directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    grouped: dict[Path, list[_StagedImage]] = {}
    for staged_image in staged_ng_images:
        grouped.setdefault(staged_image.source_path, []).append(staged_image)
    mapping: dict[Path, Path] = {}
    for source_path, images in grouped.items():
        matches = _matching_masks(source_path, source_masks)
        if not matches:
            continue
        mask_path = _require_one_matching_mask(source_path, matches)
        _validate_mask_dimensions(source_path, mask_path)
        prepared_masks = preprocessing_pipeline.prepare_mask_path(mask_path)
        destination.mkdir(parents=True, exist_ok=True)
        for staged_image in images:
            tile = staged_image.preprocessing_tile
            if tile is None:
                raise ValueError("Preprocessed mask staging requires tile provenance.")
            staged_mask_path = (destination / staged_image.staged_path.name).resolve()
            Image.fromarray(prepared_masks[tile.index]).save(staged_mask_path)
            mapping[mask_path] = staged_mask_path
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
        matches = _matching_masks(ng_source, source_masks)
        if not matches:
            continue
        mask_path = _require_one_matching_mask(ng_source, matches)
        _validate_mask_dimensions(ng_source, mask_path)
        destination.mkdir(parents=True, exist_ok=True)
        staged_mask_path = destination / staged_ng_path.name
        shutil.copy2(mask_path, staged_mask_path)
        mapping[mask_path] = staged_mask_path
    return mapping


def _matching_masks(source_path: Path, masks: Iterable[Path]) -> list[Path]:
    """Return exact documented mask names: ``image.ext`` or ``image_mask.ext`` only."""
    allowed_stems = {source_path.stem.casefold(), f"{source_path.stem}_mask".casefold()}
    return [mask for mask in masks if mask.stem.casefold() in allowed_stems]


def _require_one_matching_mask(source_path: Path, matches: list[Path]) -> Path:
    if len(matches) == 1:
        return matches[0]
    names = ", ".join(path.name for path in matches)
    raise ValueError(
        f"Ambiguous masks for {source_path.name}; provide exactly one of "
        f"{source_path.name} or {source_path.stem}_mask.<extension>. Found: {names}"
    )


def _validate_mask_dimensions(source_path: Path, mask_path: Path) -> None:
    with Image.open(source_path) as source, Image.open(mask_path) as mask:
        if source.size != mask.size:
            raise ValueError(
                f"Mask dimensions for {mask_path.name} must match {source_path.name}: "
                f"expected {source.size[0]}x{source.size[1]}, received {mask.size[0]}x{mask.size[1]}."
            )


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