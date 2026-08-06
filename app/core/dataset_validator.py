"""Dataset validation utilities."""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.models.dataset_config import SUPPORTED_IMAGE_EXTENSIONS, DatasetConfig, DatasetRole

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ValidationIssue:
    """A single dataset validation issue."""

    level: str
    role: str
    message: str
    path: str = ""


@dataclass(slots=True)
class DatasetValidationReport:
    """Validation report with errors and warnings."""

    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)
    stats: dict[str, dict[str, str | int]] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not self.errors


class DatasetValidator:
    """Validate imported dataset folders."""

    MIN_OK_TRAIN_IMAGES = 2
    MIN_OK_TEST_IMAGES = 1
    MIN_NG_TEST_IMAGES = 1

    def validate(self, config: DatasetConfig) -> DatasetValidationReport:
        """Validate all configured dataset folders."""
        report = DatasetValidationReport()
        role_files: dict[DatasetRole, list[Path]] = {}

        for role, folder in config.folders.items():
            path = folder.resolved_path()
            if path is None:
                if role is DatasetRole.MASKS:
                    continue
                report.errors.append(ValidationIssue("error", role.value, "Folder is not configured"))
                continue
            self._validate_folder(role, path, report, role_files)

        self._validate_counts(role_files, report)
        self._validate_masks(role_files, report)
        return report

    def _validate_folder(
        self,
        role: DatasetRole,
        path: Path,
        report: DatasetValidationReport,
        role_files: dict[DatasetRole, list[Path]],
    ) -> None:
        if not path.exists():
            report.errors.append(ValidationIssue("error", role.value, "Folder does not exist", str(path)))
            return
        if not path.is_dir():
            report.errors.append(ValidationIssue("error", role.value, "Path is not a folder", str(path)))
            return

        all_files = sorted(item for item in path.iterdir() if item.is_file())
        if not all_files:
            report.errors.append(ValidationIssue("error", role.value, "Folder is empty", str(path)))
            return

        valid_images: list[Path] = []
        dimensions: Counter[tuple[int, int]] = Counter()
        color_modes: Counter[str] = Counter()
        file_names: Counter[str] = Counter()
        content_hashes: dict[str, Path] = {}

        for file_path in all_files:
            file_names[file_path.name] += 1
            if file_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                report.errors.append(
                    ValidationIssue("error", role.value, "Unsupported file type", str(file_path))
                )
                continue
            try:
                with Image.open(file_path) as image:
                    image.verify()
                with Image.open(file_path) as image:
                    dimensions[image.size] += 1
                    color_modes[image.mode] += 1
            except (UnidentifiedImageError, OSError):
                LOGGER.exception("Corrupt image detected: %s", file_path)
                report.errors.append(ValidationIssue("error", role.value, "Corrupt image", str(file_path)))
                continue

            digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
            if digest in content_hashes:
                report.warnings.append(
                    ValidationIssue(
                        "warning",
                        role.value,
                        f"Duplicate file content matches {content_hashes[digest].name}",
                        str(file_path),
                    )
                )
            else:
                content_hashes[digest] = file_path
            valid_images.append(file_path)

        for name, count in file_names.items():
            if count > 1:
                report.errors.append(ValidationIssue("error", role.value, "Duplicate filename", name))

        if len(dimensions) > 1:
            report.warnings.append(
                ValidationIssue("warning", role.value, "Mixed image dimensions detected", str(path))
            )
        if len(color_modes) > 1:
            report.warnings.append(ValidationIssue("warning", role.value, "Mixed color modes detected", str(path)))

        if valid_images:
            common_resolution, _ = dimensions.most_common(1)[0]
            common_mode, _ = color_modes.most_common(1)[0]
            report.stats[role.value] = {
                "image_count": len(valid_images),
                "typical_resolution": f"{common_resolution[0]}x{common_resolution[1]}",
                "color_mode": common_mode,
            }
        role_files[role] = valid_images

    def _validate_counts(
        self,
        role_files: dict[DatasetRole, list[Path]],
        report: DatasetValidationReport,
    ) -> None:
        if len(role_files.get(DatasetRole.OK_TRAIN, [])) < self.MIN_OK_TRAIN_IMAGES:
            report.errors.append(
                ValidationIssue("error", DatasetRole.OK_TRAIN.value, "Insufficient OK training images")
            )
        if len(role_files.get(DatasetRole.OK_TEST, [])) < self.MIN_OK_TEST_IMAGES:
            report.errors.append(
                ValidationIssue("error", DatasetRole.OK_TEST.value, "Insufficient OK test images")
            )
        if len(role_files.get(DatasetRole.NG_TEST, [])) < self.MIN_NG_TEST_IMAGES:
            report.errors.append(
                ValidationIssue("error", DatasetRole.NG_TEST.value, "Insufficient NG test images")
            )

    def _validate_masks(
        self,
        role_files: dict[DatasetRole, list[Path]],
        report: DatasetValidationReport,
    ) -> None:
        ng_files = role_files.get(DatasetRole.NG_TEST, [])
        mask_files = role_files.get(DatasetRole.MASKS, [])
        if not mask_files:
            if ng_files:
                report.warnings.append(
                    ValidationIssue("warning", DatasetRole.MASKS.value, "Masks not supplied; pixel metrics unavailable")
                )
            return

        ng_stems = {path.stem for path in ng_files}
        mask_stems = {path.stem for path in mask_files}
        missing_masks = sorted(ng_stems - mask_stems)
        if missing_masks:
            for stem in missing_masks:
                report.warnings.append(
                    ValidationIssue(
                        "warning",
                        DatasetRole.MASKS.value,
                        "Missing optional mask for NG image",
                        stem,
                    )
                )

