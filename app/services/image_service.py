"""Image utility helpers."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


class ImageService:
    """Load lightweight image previews and overlays."""

    def typical_resolution(self, path: Path) -> tuple[int, int]:
        """Read image dimensions."""
        with Image.open(path) as image:
            return image.size

