"""Tests for the project-only Preprocess Images tab."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from app.models.image_preprocessing import ColorMode, ImagePreprocessingConfig, MorphologyOperation
from app.models.inspection_region import InspectionRegionConfig
from app.models.preprocessing_config import PreprocessingConfig
from app.models.preprocessing_preview import PreprocessingPreviewState, PreviewSource
from app.ui.main_window import MainWindow
from app.ui.pages.preprocess_images_page import PreprocessImagesPage


def _image(path: Path, color: tuple[int, int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 12), color).save(path)
    return path


def _page(project_good_paths: tuple[Path, ...]) -> PreprocessImagesPage:
    QApplication.instance() or QApplication([])
    page = PreprocessImagesPage()
    page.set_context(
        project_root=Path.cwd(),
        preprocessing=PreprocessingConfig(),
        inspection_region=InspectionRegionConfig(),
        model_id="patchcore",
        project_good_paths=project_good_paths,
        preview_state=PreprocessingPreviewState(),
    )
    return page


def test_default_preview_source_is_project_good_images(tmp_path: Path) -> None:
    first = _image(tmp_path / "good-a.png", (10, 20, 30))
    second = _image(tmp_path / "good-b.png", (40, 50, 60))
    page = _page((second, first))

    assert page.active_preview_source is PreviewSource.PROJECT_GOOD_IMAGES
    assert page.active_source_paths == (first.resolve(), second.resolve())
    assert "Project Good Images" in page.active_source_label.text()

    page.close()


def test_custom_image_completely_replaces_project_preview_source(tmp_path: Path) -> None:
    project_image = _image(tmp_path / "good.png", (10, 20, 30))
    custom_image = _image(tmp_path / "custom.png", (90, 80, 70))
    page = _page((project_image,))

    page.set_custom_image(custom_image)

    assert page.active_preview_source is PreviewSource.CUSTOM_IMAGE
    assert page.active_source_paths == (custom_image.resolve(),)
    assert project_image.resolve() not in page.active_source_paths

    page.close()


def test_custom_folder_completely_replaces_preview_and_is_not_recursive(tmp_path: Path) -> None:
    project_image = _image(tmp_path / "good.png", (10, 20, 30))
    custom_folder = tmp_path / "custom"
    first = _image(custom_folder / "b.png", (40, 50, 60))
    second = _image(custom_folder / "a.png", (70, 80, 90))
    _image(custom_folder / "nested" / "not-in-preview.png", (0, 0, 0))
    (custom_folder / "bad.png").write_bytes(b"not an image")
    page = _page((project_image,))

    page.set_custom_folder(custom_folder)

    assert page.active_preview_source is PreviewSource.CUSTOM_FOLDER
    assert page.active_source_paths == (second.resolve(), first.resolve())
    assert project_image.resolve() not in page.active_source_paths
    assert "Unreadable preview image" in page.status_label.text()

    page.close()


def test_preview_pixels_use_the_same_pipeline_operations_as_training(tmp_path: Path) -> None:
    source = _image(tmp_path / "good.png", (20, 40, 80))
    page = _page((source,))
    page.set_context(
        project_root=tmp_path,
        preprocessing=PreprocessingConfig(
            image_preprocessing=ImagePreprocessingConfig(
                profile_id="gray-v1",
                color_mode=ColorMode.GRAYSCALE_REPLICATED_RGB,
            )
        ),
        inspection_region=InspectionRegionConfig(),
        model_id="patchcore",
        project_good_paths=(source,),
        preview_state=PreprocessingPreviewState(),
    )

    values = page.preprocessed_preview_array

    assert values is not None
    assert np.array_equal(values[:, :, 0], values[:, :, 1])
    assert np.array_equal(values[:, :, 1], values[:, :, 2])
    page.close()


def test_custom_preview_dimension_mismatch_never_stretches_the_saved_raw_roi(tmp_path: Path) -> None:
    project_image = _image(tmp_path / "project.png", (20, 30, 40))
    custom_image = _image(tmp_path / "custom.png", (50, 60, 70))
    page = _page((project_image,))
    inspection_region = InspectionRegionConfig(
        enabled=True,
        source_width=20,
        source_height=12,
        points_px=((2, 1), (17, 1), (17, 10), (2, 10)),
    )
    Image.new("RGB", (20, 12), (20, 30, 40)).save(project_image)
    page.set_context(
        project_root=tmp_path,
        preprocessing=PreprocessingConfig(),
        inspection_region=inspection_region,
        model_id="patchcore",
        project_good_paths=(project_image,),
        preview_state=PreprocessingPreviewState(),
    )

    page.set_custom_image(custom_image)

    assert page.preprocessed_preview_array is None
    assert not page.original_canvas._pixmap.isNull()
    assert "dimensions do not match" in page.status_label.text()
    page.close()


def test_custom_image_preprocessing_profile_is_not_mislabelled_as_grayscale_only(tmp_path: Path) -> None:
    source = _image(tmp_path / "good.png", (20, 30, 40))
    page = _page((source,))
    custom_profile = ImagePreprocessingConfig(
        profile_id="custom-v1",
        color_mode=ColorMode.GRAYSCALE_REPLICATED_RGB,
        gaussian_sigma=2.0,
    )
    page.set_context(
        project_root=tmp_path,
        preprocessing=PreprocessingConfig(image_preprocessing=custom_profile),
        inspection_region=InspectionRegionConfig(),
        model_id="patchcore",
        project_good_paths=(source,),
        preview_state=PreprocessingPreviewState(),
    )

    assert page.preset_combo.currentData() == "custom"
    page.close()


def test_narrow_layout_does_not_need_horizontal_scrolling(tmp_path: Path) -> None:
    page = _page((_image(tmp_path / "good.png", (20, 30, 40)),))
    scroll_area = MainWindow._create_page_scroll_area(page)
    scroll_area.resize(560, 700)
    scroll_area.show()
    QApplication.processEvents()

    assert page.minimumSizeHint().width() <= scroll_area.viewport().width()
    assert scroll_area.horizontalScrollBarPolicy() is Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert scroll_area.horizontalScrollBar().maximum() == 0

    scroll_area.close()


def test_inactive_numeric_controls_are_disabled_until_their_operation_is_selected(tmp_path: Path) -> None:
    page = _page((_image(tmp_path / "good.png", (20, 30, 40)),))

    assert not page.box_width_spin.isEnabled()
    assert not page.gaussian_kernel_spin.isEnabled()
    assert not page.disk_radius_spin.isEnabled()

    page.smoothing_combo.setCurrentIndex(page.smoothing_combo.findData("gaussian_blur"))
    page.gaussian_auto_kernel_check.setChecked(False)
    page.morphology_combo.setCurrentIndex(page.morphology_combo.findData(MorphologyOperation.DISK_OPENING.value))

    assert page.gaussian_kernel_spin.isEnabled()
    assert page.disk_radius_spin.isEnabled()
    assert page.disk_iterations_spin.isEnabled()
    page.close()