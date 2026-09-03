"""Project-draft image preprocessing configuration and exact runtime previews."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import random
from typing import Callable

import numpy as np
from PIL import Image, UnidentifiedImageError
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QMouseEvent, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QStyle,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.core.image_preprocessor import ImagePreprocessor
from app.core.inspection_region import InspectionRegionProcessor
from app.core.preprocessing_pipeline import PreprocessingPipeline
from app.models.dataset_config import SUPPORTED_IMAGE_EXTENSIONS
from app.models.image_preprocessing import (
    BorderMode,
    ColorMode,
    CUSTOM_PROFILE_ID,
    ImagePreprocessingConfig,
    MorphologyOperation,
    PreprocessingPreset,
    SmoothingFilter,
)
from app.models.inspection_region import InspectionRegionConfig
from app.models.preprocessing_config import PreprocessingConfig
from app.models.preprocessing_preview import PreprocessingPreviewState, PreviewSource
from app.ui.pages.inspection_region_page import InspectionRegionCanvas


class _ArrayPreviewLabel(QLabel):
    """Fixed-range RGB preview with optional source-pixel inspection."""

    pixel_hovered = Signal(str)

    def __init__(self, title: str) -> None:
        super().__init__(title)
        self._title = title
        self._array: np.ndarray | None = None
        self._pixmap = QPixmap()
        self._zoom_percent = 100
        self.setObjectName("DatasetThumbnail")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(220, 180)
        self.setMouseTracking(True)

    def set_array(self, values: np.ndarray | None) -> None:
        self._array = None if values is None else np.ascontiguousarray(values)
        if self._array is None:
            self._pixmap = QPixmap()
        else:
            image = QImage(
                self._array.data,
                self._array.shape[1],
                self._array.shape[0],
                self._array.strides[0],
                QImage.Format.Format_RGB888,
            ).copy()
            self._pixmap = QPixmap.fromImage(image)
        self._render()

    def set_zoom_percent(self, value: int) -> None:
        self._zoom_percent = value
        self._render()

    def resizeEvent(self, event: object) -> None:
        super().resizeEvent(event)  # type: ignore[arg-type]
        self._render()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._array is not None and not self._pixmap.isNull():
            target = self._target_rect()
            point = event.position().toPoint()
            if target.contains(point):
                x = min(self._array.shape[1] - 1, max(0, int((point.x() - target.x()) * self._array.shape[1] / target.width())))
                y = min(self._array.shape[0] - 1, max(0, int((point.y() - target.y()) * self._array.shape[0] / target.height())))
                red, green, blue = (int(value) for value in self._array[y, x])
                self.pixel_hovered.emit(f"{self._title}: ({x}, {y}) RGB=({red}, {green}, {blue})")
        super().mouseMoveEvent(event)

    def _target_rect(self):
        size = self._pixmap.size().scaled(
            self.size() * self._zoom_percent / 100,
            Qt.AspectRatioMode.KeepAspectRatio,
        )
        return self.contentsRect().adjusted(
            (self.width() - size.width()) // 2,
            (self.height() - size.height()) // 2,
            -(self.width() - size.width()) // 2,
            -(self.height() - size.height()) // 2,
        )

    def _render(self) -> None:
        if self._pixmap.isNull():
            self.clear()
            self.setText(self._title)
            return
        size = self._pixmap.size().scaled(
            self.size() * self._zoom_percent / 100,
            Qt.AspectRatioMode.KeepAspectRatio,
        )
        self.setPixmap(self._pixmap.scaled(size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.setText("")


class PreprocessImagesPage(QWidget):
    """Configure a frozen image profile and preview the exact shared pipeline output."""

    profile_save_requested = Signal(object)
    preview_state_changed = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self._project_root = Path()
        self._base_preprocessing = PreprocessingConfig()
        self._inspection_region = InspectionRegionConfig()
        self._model_id = "patchcore"
        self._project_good_paths: tuple[Path, ...] = ()
        self._active_paths: tuple[Path, ...] = ()
        self._active_index = -1
        self._state = PreprocessingPreviewState()
        self._preview_arrays: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 28)
        root.setSpacing(16)

        source_group = QGroupBox("Preview Source")
        source_layout = QVBoxLayout(source_group)
        source_layout.setSpacing(8)
        self.project_good_radio = QRadioButton("Project Good Images")
        self.custom_image_radio = QRadioButton("Custom Image")
        self.custom_folder_radio = QRadioButton("Custom Folder")
        self._source_buttons = QButtonGroup(self)
        source_modes = QGridLayout()
        source_modes.setHorizontalSpacing(8)
        source_modes.setVerticalSpacing(4)
        for button in (self.project_good_radio, self.custom_image_radio, self.custom_folder_radio):
            self._source_buttons.addButton(button)
        source_modes.addWidget(self.project_good_radio, 0, 0)
        source_modes.addWidget(self.custom_image_radio, 0, 1)
        source_modes.addWidget(self.custom_folder_radio, 1, 0)
        self.choose_custom_image_button = QPushButton("Select Custom Image")
        self.choose_custom_folder_button = QPushButton("Select Custom Folder")
        self.previous_button = QPushButton()
        self.previous_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowBack))
        self.previous_button.setToolTip("Previous preview image")
        self.next_button = QPushButton()
        self.next_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowForward))
        self.next_button.setToolTip("Next preview image")
        self.random_button = QPushButton("Random")
        self.reset_source_button = QPushButton("Reset to Project Good Images")
        self.already_rectified_check = QCheckBox("Custom source is already rectified")
        source_picker_actions = QVBoxLayout()
        source_picker_actions.addWidget(self.choose_custom_image_button)
        source_picker_actions.addWidget(self.choose_custom_folder_button)
        source_navigation = QHBoxLayout()
        source_navigation.addWidget(self.previous_button)
        source_navigation.addWidget(self.next_button)
        source_navigation.addWidget(self.random_button)
        source_navigation.addStretch(1)
        source_layout.addLayout(source_modes)
        source_layout.addLayout(source_picker_actions)
        source_layout.addLayout(source_navigation)
        source_layout.addWidget(self.reset_source_button)
        source_layout.addWidget(self.already_rectified_check)
        root.addWidget(source_group)

        self.active_source_label = QLabel("Active preview source: Project Good Images")
        self.active_source_label.setObjectName("ModelSupport")
        self.active_source_label.setWordWrap(True)
        root.addWidget(self.active_source_label)

        profile_group = QGroupBox("Image Preprocessing Profile")
        profile_form = QFormLayout(profile_group)
        profile_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        profile_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.preset_combo = QComboBox()
        for label, preset in (
            ("No Additional Preprocessing", PreprocessingPreset.NONE),
            ("Grayscale Only", PreprocessingPreset.GRAYSCALE_ONLY),
            ("Grayscale + Gaussian", PreprocessingPreset.GRAYSCALE_GAUSSIAN),
            ("Grayscale + Median", PreprocessingPreset.GRAYSCALE_MEDIAN),
            ("Grayscale + Disk Opening", PreprocessingPreset.GRAYSCALE_DISK_OPENING),
            ("Grayscale + Gaussian + Disk Opening", PreprocessingPreset.GRAYSCALE_GAUSSIAN_DISK_OPENING),
        ):
            self.preset_combo.addItem(label, preset.value)
        self.preset_combo.addItem("Custom", "custom")
        self.color_mode_combo = QComboBox()
        self.color_mode_combo.addItem("Preserve RGB", ColorMode.PRESERVE_RGB.value)
        self.color_mode_combo.addItem("Grayscale, replicated to 3 channels", ColorMode.GRAYSCALE_REPLICATED_RGB.value)
        self.smoothing_combo = QComboBox()
        self.smoothing_combo.addItem("None", SmoothingFilter.NONE.value)
        self.smoothing_combo.addItem("Box Blur", SmoothingFilter.BOX_BLUR.value)
        self.smoothing_combo.addItem("Gaussian Blur (Recommended)", SmoothingFilter.GAUSSIAN_BLUR.value)
        self.smoothing_combo.addItem("Median Blur", SmoothingFilter.MEDIAN_BLUR.value)
        self.smoothing_combo.setCurrentIndex(self.smoothing_combo.findData(SmoothingFilter.GAUSSIAN_BLUR.value))
        self.smoothing_border_combo = self._border_combo()
        self.box_width_spin = self._odd_spinbox(3)
        self.box_height_spin = self._odd_spinbox(3)
        self.gaussian_sigma_spin = QDoubleSpinBox()
        self.gaussian_sigma_spin.setRange(0.01, 1000.0)
        self.gaussian_sigma_spin.setDecimals(3)
        self.gaussian_sigma_spin.setSingleStep(0.1)
        self.gaussian_sigma_spin.setValue(1.0)
        self.gaussian_auto_kernel_check = QCheckBox("Automatic odd kernel")
        self.gaussian_auto_kernel_check.setChecked(True)
        self.gaussian_kernel_spin = self._odd_spinbox(7)
        self.gaussian_auto_kernel_check.setToolTip("Choose the kernel size automatically from the Gaussian sigma.")
        self.gaussian_kernel_spin.setToolTip("Available when Automatic odd kernel is turned off.")
        self.median_kernel_spin = self._odd_spinbox(3)
        self.morphology_combo = QComboBox()
        self.morphology_combo.addItem("None", MorphologyOperation.NONE.value)
        self.morphology_combo.addItem("Disk Morphological Opening", MorphologyOperation.DISK_OPENING.value)
        self.disk_radius_spin = QSpinBox()
        self.disk_radius_spin.setRange(1, 100000)
        self.disk_radius_spin.setValue(2)
        self.disk_iterations_spin = QSpinBox()
        self.disk_iterations_spin.setRange(1, 1000)
        self.disk_iterations_spin.setValue(1)
        self.morphology_border_combo = self._border_combo()
        for combo in (
            self.preset_combo,
            self.color_mode_combo,
            self.smoothing_combo,
            self.smoothing_border_combo,
            self.morphology_combo,
            self.morphology_border_combo,
        ):
            combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.fiber_thickness_spin = self._optional_positive_spinbox()
        self.minimum_defect_spin = self._optional_positive_spinbox()
        self.pixels_per_mm_spin = self._optional_positive_spinbox()
        self.profile_warning_label = QLabel("Gaussian blur suppresses fine texture and may weaken defects smaller than its blur scale.")
        self.profile_warning_label.setObjectName("ModelSupport")
        self.profile_warning_label.setWordWrap(True)
        self.disk_opening_help_label = QLabel(
            "Disk opening removes bright structures that are too thin to contain the selected disk. "
            "Select a disk larger than the expected fiber thickness but smaller than the smallest important defect."
        )
        self.disk_opening_help_label.setObjectName("ModelSupport")
        self.disk_opening_help_label.setWordWrap(True)
        profile_form.addRow("Preset", self.preset_combo)
        profile_form.addRow("Color Mode", self.color_mode_combo)
        profile_form.addRow("Smoothing Filter", self.smoothing_combo)
        profile_form.addRow("Smoothing Border", self.smoothing_border_combo)
        profile_form.addRow("Box Kernel Width", self.box_width_spin)
        profile_form.addRow("Box Kernel Height", self.box_height_spin)
        profile_form.addRow("Gaussian Sigma", self.gaussian_sigma_spin)
        profile_form.addRow("Gaussian Kernel", self.gaussian_auto_kernel_check)
        profile_form.addRow("Gaussian Kernel Size", self.gaussian_kernel_spin)
        profile_form.addRow("Median Kernel Size", self.median_kernel_spin)
        profile_form.addRow("Morphology", self.morphology_combo)
        profile_form.addRow(self.disk_opening_help_label)
        profile_form.addRow("Disk Radius", self.disk_radius_spin)
        profile_form.addRow("Disk Iterations", self.disk_iterations_spin)
        profile_form.addRow("Morphology Border", self.morphology_border_combo)
        profile_form.addRow("Expected Maximum Fiber Thickness", self.fiber_thickness_spin)
        profile_form.addRow("Expected Minimum Defect Diameter", self.minimum_defect_spin)
        profile_form.addRow("Pixels per Millimetre", self.pixels_per_mm_spin)
        profile_form.addRow(self.profile_warning_label)
        root.addWidget(profile_group)

        profile_buttons = QHBoxLayout()
        self.save_profile_button = QPushButton("Save Preprocessing Profile")
        self.save_profile_button.setObjectName("PrimaryButton")
        profile_buttons.addWidget(self.save_profile_button)
        profile_buttons.addStretch(1)
        root.addLayout(profile_buttons)

        inspection_controls = QGridLayout()
        inspection_controls.setHorizontalSpacing(12)
        inspection_controls.setVerticalSpacing(8)
        self.zoom_check = QCheckBox("Enable preview zoom")
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(50, 200)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setEnabled(False)
        self.pixel_inspection_check = QCheckBox("Enable pixel-value inspection")
        self.pixel_value_label = QLabel("Pixel inspection disabled")
        self.pixel_value_label.setWordWrap(True)
        self.pixel_value_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        inspection_controls.addWidget(self.zoom_check, 0, 0)
        inspection_controls.addWidget(self.zoom_slider, 0, 1)
        inspection_controls.addWidget(self.pixel_inspection_check, 1, 0)
        inspection_controls.addWidget(self.pixel_value_label, 1, 1)
        inspection_controls.setColumnStretch(1, 1)
        root.addLayout(inspection_controls)

        previews = QGridLayout()
        previews.setHorizontalSpacing(16)
        previews.setVerticalSpacing(16)
        self.original_canvas = InspectionRegionCanvas()
        self.original_canvas.setObjectName("DatasetThumbnail")
        self.original_canvas.setMinimumSize(220, 200)
        self.original_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.rectified_preview = _ArrayPreviewLabel("Rectified ROI")
        self.preprocessed_preview = _ArrayPreviewLabel("Preprocessed ROI")
        self.difference_preview = _ArrayPreviewLabel("Absolute Difference")
        for index, (title, widget) in enumerate((
            ("Original with ROI", self.original_canvas),
            ("Rectified ROI", self.rectified_preview),
            ("Final Preprocessed ROI", self.preprocessed_preview),
            ("Absolute Difference", self.difference_preview),
        )):
            column = QVBoxLayout()
            caption = QLabel(title)
            caption.setWordWrap(True)
            caption.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            column.addWidget(caption)
            column.addWidget(widget, stretch=1)
            previews.addLayout(column, index // 2, index % 2)
        previews.setColumnStretch(0, 1)
        previews.setColumnStretch(1, 1)
        root.addLayout(previews, stretch=1)

        self.status_label = QLabel("Project Good Images are used only for preview.")
        self.status_label.setObjectName("ModelSupport")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)
        root.addStretch(1)

        self.project_good_radio.toggled.connect(lambda checked: checked and self.reset_to_project_good_images())
        self.custom_image_radio.toggled.connect(lambda checked: checked and self._select_custom_radio(PreviewSource.CUSTOM_IMAGE))
        self.custom_folder_radio.toggled.connect(lambda checked: checked and self._select_custom_radio(PreviewSource.CUSTOM_FOLDER))
        self.choose_custom_image_button.clicked.connect(self._choose_custom_image)
        self.choose_custom_folder_button.clicked.connect(self._choose_custom_folder)
        self.previous_button.clicked.connect(lambda: self._move_source(-1))
        self.next_button.clicked.connect(lambda: self._move_source(1))
        self.random_button.clicked.connect(self._choose_random_source)
        self.reset_source_button.clicked.connect(self.reset_to_project_good_images)
        self.already_rectified_check.toggled.connect(self._set_already_rectified)
        self.preset_combo.currentIndexChanged.connect(self._apply_preset)
        for widget in (
            self.color_mode_combo,
            self.smoothing_combo,
            self.smoothing_border_combo,
            self.box_width_spin,
            self.box_height_spin,
            self.gaussian_sigma_spin,
            self.gaussian_auto_kernel_check,
            self.gaussian_kernel_spin,
            self.median_kernel_spin,
            self.morphology_combo,
            self.disk_radius_spin,
            self.disk_iterations_spin,
            self.morphology_border_combo,
            self.fiber_thickness_spin,
            self.minimum_defect_spin,
            self.pixels_per_mm_spin,
        ):
            signal = getattr(widget, "valueChanged", None) or getattr(widget, "currentIndexChanged", None) or getattr(widget, "toggled")
            signal.connect(self._refresh_preview)
        self.zoom_check.toggled.connect(self._set_zoom_enabled)
        self.zoom_slider.valueChanged.connect(self._set_zoom)
        self.pixel_inspection_check.toggled.connect(self._set_pixel_inspection)
        for preview in (self.rectified_preview, self.preprocessed_preview, self.difference_preview):
            preview.pixel_hovered.connect(self._show_pixel_value)
        self.save_profile_button.clicked.connect(self._save_profile)
        self._update_control_state()

    @property
    def active_preview_source(self) -> PreviewSource:
        """Return the source currently driving the visual preview only."""
        return self._state.source

    @property
    def active_source_paths(self) -> tuple[Path, ...]:
        """Return deterministic preview paths without exposing project dataset mutation."""
        return self._active_paths

    @property
    def preprocessed_preview_array(self) -> np.ndarray | None:
        """Return the latest exact pre-padding ROI preview for tests and diagnostics."""
        return None if self._preview_arrays is None else self._preview_arrays[1].copy()

    def set_context(
        self,
        *,
        project_root: Path,
        preprocessing: PreprocessingConfig,
        inspection_region: InspectionRegionConfig,
        model_id: str,
        project_good_paths: tuple[Path, ...],
        preview_state: PreprocessingPreviewState,
    ) -> None:
        """Load project draft/configuration state without scanning custom paths recursively."""
        self._project_root = project_root
        self._base_preprocessing = preprocessing
        self._inspection_region = inspection_region
        self._model_id = model_id
        self._project_good_paths, unreadable = self._readable_images(project_good_paths)
        self._state = preview_state
        self._set_controls_from_profile(preprocessing.image_preprocessing)
        self._set_preview_source(preview_state, emit_state=False, inherited_warnings=unreadable)

    def reset_to_project_good_images(self) -> None:
        """Restore the non-mutating project Good-image preview source."""
        self._set_preview_source(PreprocessingPreviewState(), emit_state=True)

    def set_custom_image(self, path: Path) -> None:
        """Select exactly one custom preview image and discard project preview paths."""
        candidate = path.expanduser().resolve()
        valid, warnings = self._readable_images((candidate,))
        if not valid:
            self._show_source_warning(warnings or [f"Unreadable custom image: {candidate}"])
            return
        self._set_preview_source(
            PreprocessingPreviewState(
                source=PreviewSource.CUSTOM_IMAGE,
                custom_image_path=str(candidate),
                already_rectified=self.already_rectified_check.isChecked(),
            ),
            emit_state=True,
        )

    def set_custom_folder(self, folder: Path) -> None:
        """Select a deterministic non-recursive custom folder preview source."""
        directory = folder.expanduser().resolve()
        if not directory.is_dir():
            self._show_source_warning([f"Custom preview folder does not exist: {directory}"])
            return
        paths, warnings = self._readable_images(
            tuple(
                path for path in sorted(directory.iterdir(), key=lambda item: item.name.casefold())
                if path.is_file() and path.suffix.casefold() in SUPPORTED_IMAGE_EXTENSIONS
            )
        )
        if not paths:
            self._show_source_warning([*warnings, "Custom folder contains no readable supported images."])
            return
        self._set_preview_source(
            PreprocessingPreviewState(
                source=PreviewSource.CUSTOM_FOLDER,
                custom_folder_path=str(directory),
                already_rectified=self.already_rectified_check.isChecked(),
            ),
            emit_state=True,
            inherited_warnings=warnings,
        )

    def profile(self) -> ImagePreprocessingConfig:
        """Return explicit, validated image operations represented by the current controls."""
        color_mode = ColorMode(str(self.color_mode_combo.currentData()))
        smoothing = SmoothingFilter(str(self.smoothing_combo.currentData()))
        morphology = MorphologyOperation(str(self.morphology_combo.currentData()))
        is_legacy_none = color_mode is ColorMode.PRESERVE_RGB and smoothing is SmoothingFilter.NONE and morphology is MorphologyOperation.NONE
        return ImagePreprocessingConfig(
            profile_id="legacy_none_v1" if is_legacy_none else CUSTOM_PROFILE_ID,
            color_mode=color_mode,
            smoothing_filter=smoothing,
            box_kernel_width=self.box_width_spin.value(),
            box_kernel_height=self.box_height_spin.value(),
            gaussian_sigma=self.gaussian_sigma_spin.value(),
            gaussian_kernel_size=None if self.gaussian_auto_kernel_check.isChecked() else self.gaussian_kernel_spin.value(),
            median_kernel_size=self.median_kernel_spin.value(),
            smoothing_border_mode=BorderMode(str(self.smoothing_border_combo.currentData())),
            morphology_operation=morphology,
            disk_radius=self.disk_radius_spin.value(),
            disk_iterations=self.disk_iterations_spin.value(),
            morphology_border_mode=BorderMode(str(self.morphology_border_combo.currentData())),
            expected_maximum_fiber_thickness_px=self._optional_spin_value(self.fiber_thickness_spin),
            expected_minimum_defect_diameter_px=self._optional_spin_value(self.minimum_defect_spin),
            pixels_per_millimetre=self._optional_spin_value(self.pixels_per_mm_spin),
        )

    def _choose_custom_image(self) -> None:
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "Select Custom Preview Image",
            str(self._project_root),
            "Images (*.bmp *.jpeg *.jpg *.png *.tif *.tiff *.webp)",
        )
        if selected:
            self.set_custom_image(Path(selected))

    def _choose_custom_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Select Custom Preview Folder", str(self._project_root))
        if selected:
            self.set_custom_folder(Path(selected))

    def _select_custom_radio(self, source: PreviewSource) -> None:
        if source is PreviewSource.CUSTOM_IMAGE and self._state.source is not PreviewSource.CUSTOM_IMAGE:
            self._choose_custom_image()
        elif source is PreviewSource.CUSTOM_FOLDER and self._state.source is not PreviewSource.CUSTOM_FOLDER:
            self._choose_custom_folder()
        if self._state.source is not source:
            self._set_source_radio(self._state.source)

    def _set_preview_source(
        self,
        state: PreprocessingPreviewState,
        *,
        emit_state: bool,
        inherited_warnings: tuple[str, ...] | list[str] = (),
    ) -> None:
        self._state = state
        if state.source is PreviewSource.PROJECT_GOOD_IMAGES:
            paths = self._project_good_paths
            warnings = list(inherited_warnings)
        elif state.source is PreviewSource.CUSTOM_IMAGE:
            paths, warnings = self._readable_images((Path(state.custom_image_path),))
            warnings = [*inherited_warnings, *warnings]
        else:
            folder = Path(state.custom_folder_path)
            if not folder.is_dir():
                paths, warnings = (), [*inherited_warnings, f"Custom preview folder does not exist: {folder}"]
            else:
                paths, warnings = self._readable_images(
                    tuple(
                        path for path in sorted(folder.iterdir(), key=lambda item: item.name.casefold())
                        if path.is_file() and path.suffix.casefold() in SUPPORTED_IMAGE_EXTENSIONS
                    )
                )
                warnings = [*inherited_warnings, *warnings]
        self._active_paths = paths
        self._active_index = min(state.selected_index, len(paths) - 1) if paths else -1
        self._set_source_radio(state.source)
        self.already_rectified_check.blockSignals(True)
        self.already_rectified_check.setChecked(state.already_rectified)
        self.already_rectified_check.blockSignals(False)
        self.already_rectified_check.setEnabled(state.source is not PreviewSource.PROJECT_GOOD_IMAGES)
        self._update_source_controls()
        self._refresh_preview(inherited_warnings=warnings)
        if emit_state:
            self.preview_state_changed.emit(self._state)

    def _move_source(self, offset: int) -> None:
        if not self._active_paths:
            return
        self._active_index = (self._active_index + offset) % len(self._active_paths)
        self._state = replace(self._state, selected_index=self._active_index)
        self._refresh_preview()
        self.preview_state_changed.emit(self._state)

    def _choose_random_source(self) -> None:
        if not self._active_paths:
            return
        self._active_index = random.randrange(len(self._active_paths))
        self._state = replace(self._state, selected_index=self._active_index)
        self._refresh_preview()
        self.preview_state_changed.emit(self._state)

    def _set_already_rectified(self, checked: bool) -> None:
        if self._state.source is PreviewSource.PROJECT_GOOD_IMAGES:
            return
        self._state = replace(self._state, already_rectified=checked)
        self._refresh_preview()
        self.preview_state_changed.emit(self._state)

    def _apply_preset(self) -> None:
        preset = PreprocessingPreset(str(self.preset_combo.currentData()))
        self._set_controls_from_profile(ImagePreprocessingConfig.from_preset(preset), keep_preset=True)
        self._refresh_preview()

    def _set_controls_from_profile(self, profile: ImagePreprocessingConfig, *, keep_preset: bool = False) -> None:
        widgets = (
            self.color_mode_combo,
            self.smoothing_combo,
            self.smoothing_border_combo,
            self.box_width_spin,
            self.box_height_spin,
            self.gaussian_sigma_spin,
            self.gaussian_auto_kernel_check,
            self.gaussian_kernel_spin,
            self.median_kernel_spin,
            self.morphology_combo,
            self.disk_radius_spin,
            self.disk_iterations_spin,
            self.morphology_border_combo,
            self.fiber_thickness_spin,
            self.minimum_defect_spin,
            self.pixels_per_mm_spin,
        )
        for widget in widgets:
            widget.blockSignals(True)
        self.color_mode_combo.setCurrentIndex(self.color_mode_combo.findData(profile.color_mode.value))
        self.smoothing_combo.setCurrentIndex(self.smoothing_combo.findData(profile.smoothing_filter.value))
        self.smoothing_border_combo.setCurrentIndex(self.smoothing_border_combo.findData(profile.smoothing_border_mode.value))
        self.box_width_spin.setValue(profile.box_kernel_width)
        self.box_height_spin.setValue(profile.box_kernel_height)
        self.gaussian_sigma_spin.setValue(profile.gaussian_sigma)
        self.gaussian_auto_kernel_check.setChecked(profile.gaussian_kernel_size is None)
        self.gaussian_kernel_spin.setValue(profile.resolved_gaussian_kernel_size)
        self.median_kernel_spin.setValue(profile.median_kernel_size)
        self.morphology_combo.setCurrentIndex(self.morphology_combo.findData(profile.morphology_operation.value))
        self.disk_radius_spin.setValue(profile.disk_radius)
        self.disk_iterations_spin.setValue(profile.disk_iterations)
        self.morphology_border_combo.setCurrentIndex(self.morphology_border_combo.findData(profile.morphology_border_mode.value))
        self.fiber_thickness_spin.setValue(profile.expected_maximum_fiber_thickness_px or 0.0)
        self.minimum_defect_spin.setValue(profile.expected_minimum_defect_diameter_px or 0.0)
        self.pixels_per_mm_spin.setValue(profile.pixels_per_millimetre or 0.0)
        for widget in widgets:
            widget.blockSignals(False)
        if not keep_preset:
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentIndex(self.preset_combo.findData(self._preset_value(profile)))
            self.preset_combo.blockSignals(False)
        self._update_control_state()

    def _refresh_preview(self, *_args: object, inherited_warnings: tuple[str, ...] | list[str] = ()) -> None:
        self._update_control_state()
        if self._active_index < 0 or not self._active_paths:
            self._clear_preview("No readable preview image is available.", inherited_warnings)
            return
        source_path = self._active_paths[self._active_index]
        try:
            with Image.open(source_path) as image:
                image_rgb = np.asarray(image.convert("RGB"))
            pixmap = QPixmap(str(source_path))
            self.original_canvas.set_pixmap(pixmap)
            use_overlay = (
                not self._state.already_rectified
                and self._inspection_region.enabled
                and image_rgb.shape[1::-1] == (self._inspection_region.source_width, self._inspection_region.source_height)
            )
            self.original_canvas.set_points(self._inspection_region.points_px if use_overlay else ())
            profile = self.profile()
            profile.validate()
            rectified, preprocessed, difference = self._preview_arrays_for_source(image_rgb, profile)
        except (OSError, UnidentifiedImageError, ValueError, RuntimeError) as exc:
            self._clear_preview(str(exc), inherited_warnings, retain_original=True)
            return
        self._preview_arrays = (rectified, preprocessed, difference)
        self.rectified_preview.set_array(rectified)
        self.preprocessed_preview.set_array(preprocessed)
        self.difference_preview.set_array(difference)
        warnings = [*inherited_warnings, *profile.warnings(), *self._contrast_warnings(rectified, preprocessed)]
        source_name = f"{source_path.name} ({self._active_index + 1}/{len(self._active_paths)})"
        source_kind = {
            PreviewSource.PROJECT_GOOD_IMAGES: "Project Good Images",
            PreviewSource.CUSTOM_IMAGE: "Custom Image",
            PreviewSource.CUSTOM_FOLDER: "Custom Folder",
        }[self._state.source]
        self.active_source_label.setText(f"Active preview source: {source_kind} | {source_name}")
        self.active_source_label.setToolTip(str(source_path))
        self.status_label.setText(" | ".join(warnings) if warnings else "Preview uses the same ROI and preprocessing implementation as training and inference.")

    def _preview_arrays_for_source(
        self,
        image_rgb: np.ndarray,
        profile: ImagePreprocessingConfig,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._state.already_rectified:
            expected = self._inspection_region.rectified_size() if self._inspection_region.enabled else image_rgb.shape[1::-1]
            if image_rgb.shape[1::-1] != expected:
                raise ValueError(
                    "Custom image is declared already rectified but its dimensions do not match the configured rectified ROI: "
                    f"expected {expected[0]}x{expected[1]}, received {image_rgb.shape[1]}x{image_rgb.shape[0]}."
                )
            processor = ImagePreprocessor(profile)
            preprocessed = processor.apply(image_rgb)
            return image_rgb, preprocessed, processor.absolute_difference(image_rgb, preprocessed)
        if self._inspection_region.enabled and image_rgb.shape[1::-1] != (
            self._inspection_region.source_width,
            self._inspection_region.source_height,
        ):
            raise ValueError(
                "Preview image dimensions do not match the configured raw-camera ROI source resolution; "
                "select a matching image or explicitly declare the custom source already rectified."
            )
        rectified_size = self._inspection_region.rectified_size() if self._inspection_region.enabled else image_rgb.shape[1::-1]
        config = replace(self._base_preprocessing, image_preprocessing=profile)
        pipeline = PreprocessingPipeline(self._inspection_region, config.resolve(self._model_id, rectified_size))
        return pipeline.preview_arrays(image_rgb)

    def _save_profile(self) -> None:
        try:
            profile = self.profile()
            profile.validate()
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return
        self.profile_save_requested.emit(profile)

    def _clear_preview(
        self,
        message: str,
        warnings: tuple[str, ...] | list[str] = (),
        *,
        retain_original: bool = False,
    ) -> None:
        self._preview_arrays = None
        if not retain_original:
            self.original_canvas.set_pixmap(QPixmap())
            self.original_canvas.set_points(())
        for preview in (self.rectified_preview, self.preprocessed_preview, self.difference_preview):
            preview.set_array(None)
        source_kind = {
            PreviewSource.PROJECT_GOOD_IMAGES: "Project Good Images",
            PreviewSource.CUSTOM_IMAGE: "Custom Image",
            PreviewSource.CUSTOM_FOLDER: "Custom Folder",
        }[self._state.source]
        self.active_source_label.setText(f"Active preview source: {source_kind}")
        self.status_label.setText(" | ".join([*warnings, message]))

    def _show_source_warning(self, warnings: tuple[str, ...] | list[str]) -> None:
        self.status_label.setText(" | ".join(warnings))

    def _update_control_state(self) -> None:
        smoothing = SmoothingFilter(str(self.smoothing_combo.currentData()))
        self.box_width_spin.setEnabled(smoothing is SmoothingFilter.BOX_BLUR)
        self.box_height_spin.setEnabled(smoothing is SmoothingFilter.BOX_BLUR)
        self.gaussian_sigma_spin.setEnabled(smoothing is SmoothingFilter.GAUSSIAN_BLUR)
        self.gaussian_auto_kernel_check.setEnabled(smoothing is SmoothingFilter.GAUSSIAN_BLUR)
        self.gaussian_kernel_spin.setEnabled(
            smoothing is SmoothingFilter.GAUSSIAN_BLUR and not self.gaussian_auto_kernel_check.isChecked()
        )
        self.median_kernel_spin.setEnabled(smoothing is SmoothingFilter.MEDIAN_BLUR)
        morphology = MorphologyOperation(str(self.morphology_combo.currentData()))
        self.disk_radius_spin.setEnabled(morphology is MorphologyOperation.DISK_OPENING)
        self.disk_iterations_spin.setEnabled(morphology is MorphologyOperation.DISK_OPENING)
        self.morphology_border_combo.setEnabled(morphology is MorphologyOperation.DISK_OPENING)
        self.previous_button.setEnabled(bool(self._active_paths))
        self.next_button.setEnabled(bool(self._active_paths))
        self.random_button.setEnabled(bool(self._active_paths))

    def _update_source_controls(self) -> None:
        self._update_control_state()

    def _set_source_radio(self, source: PreviewSource) -> None:
        buttons = {
            PreviewSource.PROJECT_GOOD_IMAGES: self.project_good_radio,
            PreviewSource.CUSTOM_IMAGE: self.custom_image_radio,
            PreviewSource.CUSTOM_FOLDER: self.custom_folder_radio,
        }
        for button in buttons.values():
            button.blockSignals(True)
        buttons[source].setChecked(True)
        for button in buttons.values():
            button.blockSignals(False)

    def _set_zoom_enabled(self, enabled: bool) -> None:
        self.zoom_slider.setEnabled(enabled)
        self._set_zoom(self.zoom_slider.value() if enabled else 100)

    def _set_zoom(self, percent: int) -> None:
        for preview in (self.rectified_preview, self.preprocessed_preview, self.difference_preview):
            preview.set_zoom_percent(percent)

    def _set_pixel_inspection(self, enabled: bool) -> None:
        self.pixel_value_label.setText("Move over a preview to inspect RGB values." if enabled else "Pixel inspection disabled")

    def _show_pixel_value(self, message: str) -> None:
        if self.pixel_inspection_check.isChecked():
            self.pixel_value_label.setText(message)

    @staticmethod
    def _border_combo() -> QComboBox:
        combo = QComboBox()
        combo.addItem("Reflect", BorderMode.REFLECT.value)
        combo.addItem("Replicate", BorderMode.REPLICATE.value)
        return combo

    @staticmethod
    def _odd_spinbox(value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(1, 100001)
        spin.setSingleStep(2)
        spin.setValue(value)
        return spin

    @staticmethod
    def _optional_positive_spinbox() -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 1_000_000.0)
        spin.setDecimals(3)
        spin.setSingleStep(0.1)
        spin.setSpecialValueText("Not specified")
        return spin

    @staticmethod
    def _optional_spin_value(spin: QDoubleSpinBox) -> float | None:
        return spin.value() or None

    @staticmethod
    def _preset_value(profile: ImagePreprocessingConfig) -> str:
        if profile.is_legacy_none:
            return PreprocessingPreset.NONE.value
        for preset in (
            PreprocessingPreset.GRAYSCALE_ONLY,
            PreprocessingPreset.GRAYSCALE_GAUSSIAN,
            PreprocessingPreset.GRAYSCALE_MEDIAN,
            PreprocessingPreset.GRAYSCALE_DISK_OPENING,
            PreprocessingPreset.GRAYSCALE_GAUSSIAN_DISK_OPENING,
        ):
            if profile == ImagePreprocessingConfig.from_preset(preset):
                return preset.value
        return "custom"

    @staticmethod
    def _readable_images(paths: tuple[Path, ...]) -> tuple[tuple[Path, ...], tuple[str, ...]]:
        readable: list[Path] = []
        warnings: list[str] = []
        for path in sorted((Path(path).expanduser().resolve() for path in paths), key=lambda item: str(item).casefold()):
            if not path.is_file() or path.suffix.casefold() not in SUPPORTED_IMAGE_EXTENSIONS:
                warnings.append(f"Unsupported preview image: {path}")
                continue
            try:
                with Image.open(path) as image:
                    image.convert("RGB").load()
            except (OSError, UnidentifiedImageError) as exc:
                warnings.append(f"Unreadable preview image {path.name}: {exc}")
                continue
            readable.append(path)
        return tuple(readable), tuple(warnings)

    @staticmethod
    def _contrast_warnings(rectified: np.ndarray, preprocessed: np.ndarray) -> tuple[str, ...]:
        original_luminance = rectified.astype(np.float32).mean(axis=2)
        processed_luminance = preprocessed.astype(np.float32).mean(axis=2)
        original_contrast = float(original_luminance.std())
        if original_contrast <= 0:
            return ()
        removed = 1 - float(processed_luminance.std()) / original_contrast
        return (f"Preprocessing removed {removed:.0%} of local contrast in this preview.",) if removed >= 0.5 else ()