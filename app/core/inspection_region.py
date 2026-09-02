"""Shared deterministic fixed inspection-region preprocessing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import cv2
import numpy as np
from PIL import Image
from torchvision.transforms.v2 import Transform
from torchvision.tv_tensors import Mask

from app.models.inspection_region import InspectionRegionConfig


class InspectionRegionProcessor(Transform):
    """Rectify one source-resolution-bound quadrilateral before Anomalib preprocessing."""

    def __init__(self, config: InspectionRegionConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self._matrix = self._perspective_matrix() if config.enabled else None

    def apply(self, image: Any) -> Any:
        """Apply the configured transform without modifying the source image or file."""
        if not self.config.enabled:
            return image
        if isinstance(image, Image.Image):
            return Image.fromarray(self._warp_array(np.asarray(image), cv2.INTER_LINEAR))
        if isinstance(image, np.ndarray):
            return self._warp_array(image, cv2.INTER_LINEAR)
        if hasattr(image, "detach"):
            return self._apply_tensor(image, cv2.INTER_NEAREST if isinstance(image, Mask) else cv2.INTER_LINEAR)
        raise TypeError(f"Inspection ROI cannot process image type {type(image)!r}.")

    def apply_mask(self, mask: np.ndarray) -> np.ndarray:
        """Apply the configured transform to a NumPy mask without interpolating its labels."""
        if mask.ndim != 2:
            raise ValueError("Inspection ROI mask input must have HW dimensions.")
        if not self.config.enabled:
            return mask
        return self._warp_array(mask, cv2.INTER_NEAREST)

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        """Apply the ROI through Anomalib's Torchvision transform pipeline."""
        return self.apply(inpt)

    def apply_path(self, source_path: Path) -> np.ndarray:
        """Read and rectify an RGB source image for an output visualization without modifying it."""
        with Image.open(source_path) as image:
            return self.apply(np.asarray(image.convert("RGB")))

    def validate_source_path(self, source_path: Path) -> None:
        """Ensure a selected source file is compatible with this resolution-bound contract."""
        if not self.config.enabled:
            return
        with Image.open(source_path) as image:
            self.validate_source_size(*image.size)

    def validate_source_size(self, width: int, height: int) -> None:
        """Reject input images that would make normalized ROI coordinates ambiguous."""
        if self.config.enabled and (width, height) != (self.config.source_width, self.config.source_height):
            raise ValueError(
                "Source image resolution does not match the inspection ROI contract: "
                f"expected {self.config.source_width}x{self.config.source_height}, received {width}x{height}."
            )

    @property
    def rectified_size(self) -> tuple[int, int]:
        """Return the ROI output size before model-specific preprocessing."""
        return self.config.rectified_size()

    def _perspective_matrix(self) -> np.ndarray:
        width, height = self.config.rectified_size()
        source = np.float32(self.config.points_px)
        destination = np.float32(((0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)))
        return cv2.getPerspectiveTransform(source, destination)

    def _warp_array(self, image: np.ndarray, interpolation: int) -> np.ndarray:
        if image.ndim not in {2, 3}:
            raise ValueError("Inspection ROI input must be a 2D mask or 3D image array.")
        height, width = image.shape[:2]
        self.validate_source_size(width, height)
        output_width, output_height = self.config.rectified_size()
        return cv2.warpPerspective(
            np.ascontiguousarray(image),
            self._matrix,
            (output_width, output_height),
            flags=interpolation,
        )

    def _apply_tensor(self, image: Any, interpolation: int) -> Any:
        import torch

        source = torch.as_tensor(image).detach()
        if source.ndim not in {2, 3}:
            raise ValueError("Inspection ROI tensor input must have CHW or HW dimensions.")
        device = source.device
        dtype = source.dtype
        values = source.cpu().numpy()
        if source.ndim == 3:
            warped = self._warp_array(np.moveaxis(values, 0, -1), interpolation)
            if warped.ndim == 2:
                warped = warped[..., None]
            output = torch.from_numpy(np.ascontiguousarray(np.moveaxis(warped, -1, 0)))
        else:
            output = torch.from_numpy(np.ascontiguousarray(self._warp_array(values, interpolation)))
        output = output.to(device=device, dtype=dtype)
        return Mask(output) if isinstance(image, Mask) else output


def canonical_inspection_region_json(config: InspectionRegionConfig) -> str:
    """Return canonical metadata bytes used for every ROI contract SHA-256."""
    config.validate()
    return json.dumps(config.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def inspection_region_hash(config: InspectionRegionConfig) -> str:
    """Hash the canonical ROI metadata rather than presentation-specific JSON formatting."""
    return hashlib.sha256(canonical_inspection_region_json(config).encode("utf-8")).hexdigest()


def write_inspection_region(path: Path, config: InspectionRegionConfig) -> Path:
    """Atomically persist canonical inspection metadata for project/run/deployment reuse."""
    content = canonical_inspection_region_json(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as handle:
        handle.write(content)
        temporary_path = Path(handle.name)
    temporary_path.replace(path)
    return path


def read_inspection_region(path: Path) -> InspectionRegionConfig:
    """Load only a supported, internally consistent inspection-region metadata file."""
    if not path.is_file():
        raise FileNotFoundError(f"Inspection ROI metadata not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Inspection ROI metadata must be a JSON object.")
    return InspectionRegionConfig.from_dict(payload)


def validate_inspection_region_sources(config: InspectionRegionConfig, source_paths: Any) -> None:
    """Validate every original source image before a run can use the ROI contract."""
    processor = InspectionRegionProcessor(config)
    for source_path in source_paths:
        processor.validate_source_path(Path(source_path))