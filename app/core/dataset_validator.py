"""Dataset validation utilities."""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.models.inspection_region import InspectionRegionConfig
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
    stats: dict[str, dict[str, object]] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not self.errors


class DatasetValidator:
    """Validate imported dataset folders."""

    MIN_OK_TRAIN_IMAGES = 2
    MIN_NG_TEST_IMAGES = 1
    REQUIRED_ROLES = (DatasetRole.OK_TRAIN,)
    OPTIONAL_ROLES = (
        DatasetRole.OK_VALIDATION,
        DatasetRole.NG_VALIDATION,
        DatasetRole.OK_TEST,
        DatasetRole.NG_TEST,
        DatasetRole.MASKS,
    )

    def validate(
        self,
        config: DatasetConfig,
        inspection_region: InspectionRegionConfig | None = None,
    ) -> DatasetValidationReport:
        """Validate all configured dataset folders."""
        report = DatasetValidationReport()
        role_files: dict[DatasetRole, list[Path]] = {}

        for role, folder in config.folders.items():
            path = folder.resolved_path()
            if path is None:
                if role not in self.REQUIRED_ROLES:
                    continue
                report.errors.append(ValidationIssue("error", role.value, "Folder is not configured"))
                continue
            self._validate_folder(role, path, report, role_files)

        self._validate_counts(role_files, report)
        self._describe_evaluation_method(role_files, report)
        self._validate_cross_role_duplicates(role_files, report)
        self._validate_source_resolution(role_files, report)
        self._validate_inspection_region(inspection_region, role_files, report)
        self._validate_masks(config, role_files, report)
        return report

    def _validate_folder(
        self,
        role: DatasetRole,
        path: Path,
        report: DatasetValidationReport,
        role_files: dict[DatasetRole, list[Path]],
    ) -> None:
        if not path.exists():
            if role in self.OPTIONAL_ROLES:
                return
            report.errors.append(ValidationIssue("error", role.value, "Folder does not exist", str(path)))
            return
        if not path.is_dir():
            report.errors.append(ValidationIssue("error", role.value, "Path is not a folder", str(path)))
            return

        all_files = sorted((item for item in path.rglob("*") if item.is_file()), key=lambda item: str(item).casefold())
        if not all_files:
            if role is DatasetRole.MASKS:
                role_files[role] = []
                return
            if role in self.OPTIONAL_ROLES:
                report.warnings.append(ValidationIssue("warning", role.value, "Optional folder is empty", str(path)))
                role_files[role] = []
                return
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
                "thumbnail_paths": [str(path) for path in valid_images[:1]],
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
        ng_test_count = len(role_files.get(DatasetRole.NG_TEST, []))
        if len(role_files.get(DatasetRole.OK_TRAIN, [])) < 20:
            report.warnings.append(
                ValidationIssue(
                    "warning",
                    DatasetRole.OK_TRAIN.value,
                    "Very small OK training set; production confidence will be limited",
                )
            )
        if not ng_test_count:
            report.warnings.append(
                ValidationIssue(
                    "warning",
                    DatasetRole.NG_TEST.value,
                    "No genuine NG test data; defect-detection performance will not be verified",
                )
            )
        elif ng_test_count < 10:
            report.warnings.append(
                ValidationIssue(
                    "warning",
                    DatasetRole.NG_TEST.value,
                    "Very small NG test set; production confidence will be limited",
                )
            )

    @staticmethod
    def _describe_evaluation_method(
        role_files: dict[DatasetRole, list[Path]],
        report: DatasetValidationReport,
    ) -> None:
        """Mark independent evidence and warn when a development partition will be created."""
        has_explicit_ok_evidence = bool(role_files.get(DatasetRole.OK_VALIDATION)) and bool(
            role_files.get(DatasetRole.OK_TEST)
        )
        has_ng_test = bool(role_files.get(DatasetRole.NG_TEST))
        has_explicit_ng_evidence = not has_ng_test or bool(role_files.get(DatasetRole.NG_VALIDATION))
        if has_explicit_ok_evidence and has_explicit_ng_evidence:
            report.stats["evaluation"] = {
                "method": "independent_explicit",
                "description": "Production-quality independent validation and final-test folders are configured.",
            }
            return
        report.stats["evaluation"] = {
            "method": "deterministic_partition",
            "description": "Development-grade deterministic random partition of configured source folders.",
        }
        report.warnings.append(
            ValidationIssue(
                "warning",
                "evaluation",
                "Development-grade evaluation: deterministic random partitioning will create calibration and/or final-test evidence. "
                "Configure independent validation and final-test folders for production-quality evaluation.",
            )
        )

    @staticmethod
    def _validate_cross_role_duplicates(
        role_files: dict[DatasetRole, list[Path]],
        report: DatasetValidationReport,
    ) -> None:
        """Reject source files whose bytes appear in more than one data split role."""
        hashes: dict[str, tuple[DatasetRole, Path]] = {}
        names: dict[str, tuple[DatasetRole, Path]] = {}
        for role, paths in role_files.items():
            if role is DatasetRole.MASKS:
                continue
            for path in paths:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                previous = hashes.get(digest)
                if previous is not None and previous[0] is not role:
                    report.errors.append(
                        ValidationIssue(
                            "error",
                            role.value,
                            f"Cross-split duplicate content matches {previous[0].value}/{previous[1].name}",
                            str(path),
                        )
                    )
                else:
                    hashes[digest] = (role, path)
                name_key = path.name.casefold()
                previous_name = names.get(name_key)
                if previous_name is not None and previous_name[0] is not role:
                    report.warnings.append(
                        ValidationIssue(
                            "warning",
                            role.value,
                            f"Cross-split duplicate filename matches {previous_name[0].value}/{previous_name[1].name}",
                            str(path),
                        )
                    )
                else:
                    names[name_key] = (role, path)

    @staticmethod
    def _validate_source_resolution(
        role_files: dict[DatasetRole, list[Path]],
        report: DatasetValidationReport,
    ) -> None:
        """Ensure source image resolution is consistent across train/validation/test roles."""
        resolutions: dict[tuple[int, int], Path] = {}
        for role, paths in role_files.items():
            if role is DatasetRole.MASKS:
                continue
            for path in paths:
                with Image.open(path) as image:
                    resolutions.setdefault(image.size, path)
        if len(resolutions) > 1:
            details = ", ".join(f"{width}x{height}" for width, height in sorted(resolutions))
            report.errors.append(
                ValidationIssue("error", "dataset", f"Inconsistent source resolutions: {details}")
            )

    @staticmethod
    def _validate_inspection_region(
        inspection_region: InspectionRegionConfig | None,
        role_files: dict[DatasetRole, list[Path]],
        report: DatasetValidationReport,
    ) -> None:
        """Require enabled ROIs to apply exactly to every original training/evaluation image."""
        if inspection_region is None or not inspection_region.enabled:
            return
        try:
            inspection_region.validate()
        except ValueError as exc:
            report.errors.append(ValidationIssue("error", "inspection_region", str(exc)))
            return
        source_images = [
            path
            for role, paths in role_files.items()
            if role is not DatasetRole.MASKS
            for path in paths
        ]
        for path in source_images:
            with Image.open(path) as image:
                if image.size != (inspection_region.source_width, inspection_region.source_height):
                    report.errors.append(
                        ValidationIssue(
                            "error",
                            "inspection_region",
                            (
                                "Source image resolution does not match the inspection ROI contract: "
                                f"expected {inspection_region.source_width}x{inspection_region.source_height}, "
                                f"received {image.width}x{image.height}."
                            ),
                            str(path),
                        )
                    )
        for message in inspection_region.warnings():
            report.warnings.append(ValidationIssue("warning", "inspection_region", message))

    def _validate_masks(
        self,
        config: DatasetConfig,
        role_files: dict[DatasetRole, list[Path]],
        report: DatasetValidationReport,
    ) -> None:
        ng_files = [
            *role_files.get(DatasetRole.NG_VALIDATION, []),
            *role_files.get(DatasetRole.NG_TEST, []),
        ]
        mask_files = role_files.get(DatasetRole.MASKS, [])
        if not mask_files:
            if ng_files:
                mask_path = config.folders[DatasetRole.MASKS].resolved_path()
                message = (
                    "Mask folder is empty; pixel metrics unavailable"
                    if mask_path is not None and mask_path.is_dir()
                    else "Masks not supplied; pixel metrics unavailable"
                )
                report.warnings.append(
                    ValidationIssue("warning", DatasetRole.MASKS.value, message)
                )
            return

        matched_masks: set[Path] = set()
        for ng_file in ng_files:
            matches = [mask_file for mask_file in mask_files if ng_file.stem in mask_file.stem]
            if not matches:
                report.warnings.append(
                    ValidationIssue(
                        "warning",
                        DatasetRole.MASKS.value,
                        "Missing optional mask for NG image",
                        ng_file.stem,
                    )
                )
                continue
            if len(matches) > 1:
                report.warnings.append(
                    ValidationIssue(
                        "warning",
                        DatasetRole.MASKS.value,
                        "Multiple masks match an NG image",
                        ng_file.stem,
                    )
                )
                continue
            mask_file = matches[0]
            matched_masks.add(mask_file)
            self._validate_mask_image(ng_file, mask_file, report)

        for mask_file in mask_files:
            if mask_file not in matched_masks:
                report.warnings.append(
                    ValidationIssue(
                        "warning",
                        DatasetRole.MASKS.value,
                        "Mask does not match an NG image",
                        str(mask_file),
                    )
                )

    @staticmethod
    def _validate_mask_image(ng_file: Path, mask_file: Path, report: DatasetValidationReport) -> None:
        """Check that an optional mask can serve as a pixel-level annotation."""
        with Image.open(ng_file) as ng_image, Image.open(mask_file) as mask_image:
            if mask_image.size != ng_image.size:
                report.warnings.append(
                    ValidationIssue(
                        "warning",
                        DatasetRole.MASKS.value,
                        "Mask dimensions do not match the NG image",
                        str(mask_file),
                    )
                )
            if mask_image.mode not in {"1", "L", "I", "I;16"}:
                report.warnings.append(
                    ValidationIssue(
                        "warning",
                        DatasetRole.MASKS.value,
                        "Mask should be a grayscale binary image",
                        str(mask_file),
                    )
                )

