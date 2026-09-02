"""Tests for the fixed inspection-region editor."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtWidgets import QApplication

from app.models.inspection_region import InspectionRegionConfig
from app.ui.pages.inspection_region_page import InspectionRegionPage


def test_inspection_region_page_renders_and_serializes_a_rectified_roi(tmp_path: Path) -> None:
    application = QApplication.instance() or QApplication([])
    source_path = tmp_path / "source.png"
    Image.new("RGB", (64, 64), (20, 30, 40)).save(source_path)
    page = InspectionRegionPage()
    page.resize(1000, 600)
    page.show()
    page.set_dataset_images((source_path,))
    page.set_inspection_region(
        InspectionRegionConfig(
            enabled=True,
            source_width=64,
            source_height=64,
            points_px=((4, 4), (59, 4), (59, 59), (4, 59)),
        )
    )
    application.processEvents()

    assert page.inspection_region().rectified_size() == (55, 55)
    assert not page.rectified_preview.pixmap().isNull()
    assert page.overlay_canvas.points == ((4, 4), (59, 4), (59, 59), (4, 59))

    page.close()