"""Fixed inspection-region editor for resolution-bound perspective rectification."""

from __future__ import annotations

from pathlib import Path
import random

import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from app.core.inspection_region import InspectionRegionProcessor
from app.models.inspection_region import InspectionRegionConfig, order_quad_points


class InspectionRegionCanvas(QWidget):
    """Display one source image and allow four inspection corners to be clicked or dragged."""

    points_changed = Signal(object)
    POINT_NAMES = ("TL", "TR", "BR", "BL")

    def __init__(self) -> None:
        super().__init__()
        self._pixmap = QPixmap()
        self._points: tuple[tuple[int, int], ...] = ()
        self._dragged_index: int | None = None
        self.setMinimumSize(320, 260)
        self.setMouseTracking(True)

    @property
    def points(self) -> tuple[tuple[int, int], ...]:
        """Return the authoritative source-pixel points."""
        return self._points

    def set_pixmap(self, pixmap: QPixmap) -> None:
        """Display a new source image without mutating the selected polygon."""
        self._pixmap = pixmap
        self.update()

    def set_points(self, points: tuple[tuple[int, int], ...]) -> None:
        """Set externally validated/canonical points without emitting a user-change signal."""
        self._points = points
        self.update()

    def clear_points(self) -> None:
        """Clear all corners so the operator can define a new fixed region."""
        self._points = ()
        self.points_changed.emit(self._points)
        self.update()

    def paintEvent(self, event: object) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#050708"))
        if self._pixmap.isNull():
            painter.setPen(QColor("#829396"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No source image")
            return
        target = self._image_rect()
        painter.drawPixmap(target, self._pixmap)
        if not self._points:
            return
        display_points = [self._source_to_display(point) for point in self._points]
        if len(display_points) >= 2:
            painter.setPen(QPen(QColor("#35ddcf"), 2))
            for first, second in zip(display_points, display_points[1:] + display_points[:1]):
                painter.drawLine(first, second)
        painter.setPen(QPen(QColor("#061011"), 2))
        painter.setBrush(QColor("#35ddcf"))
        for index, point in enumerate(display_points):
            painter.drawEllipse(point, 7, 7)
            painter.setPen(QColor("#dcfffb"))
            name = self.POINT_NAMES[index] if len(self._points) == 4 else str(index + 1)
            painter.drawText(point + QPoint(10, -8), name)
            painter.setPen(QPen(QColor("#061011"), 2))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if not self.isEnabled() or event.button() != Qt.MouseButton.LeftButton:
            return
        point = event.position().toPoint()
        self._dragged_index = self._point_at(point)
        if self._dragged_index is None:
            source_point = self._display_to_source(point)
            if len(self._points) < 4:
                self._points = (*self._points, source_point)
                self._dragged_index = len(self._points) - 1
            elif self._points:
                self._dragged_index = min(
                    range(len(self._points)),
                    key=lambda index: (self._source_to_display(self._points[index]) - point).manhattanLength(),
                )
                points = list(self._points)
                points[self._dragged_index] = source_point
                self._points = tuple(points)
        self.points_changed.emit(self._points)
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragged_index is None or not self.isEnabled():
            return
        points = list(self._points)
        points[self._dragged_index] = self._display_to_source(event.position().toPoint())
        self._points = tuple(points)
        self.points_changed.emit(self._points)
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._dragged_index is None:
            return
        self._dragged_index = None
        if len(self._points) == 4:
            try:
                self._points = order_quad_points(self._points)
            except ValueError:
                pass
        self.points_changed.emit(self._points)
        self.update()

    def _image_rect(self) -> QRect:
        size = self._pixmap.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        return QRect((self.width() - size.width()) // 2, (self.height() - size.height()) // 2, size.width(), size.height())

    def _display_to_source(self, point: QPoint) -> tuple[int, int]:
        rect = self._image_rect()
        if rect.width() <= 1 or rect.height() <= 1:
            return 0, 0
        x = round((point.x() - rect.x()) * (self._pixmap.width() - 1) / (rect.width() - 1))
        y = round((point.y() - rect.y()) * (self._pixmap.height() - 1) / (rect.height() - 1))
        return max(0, min(x, self._pixmap.width() - 1)), max(0, min(y, self._pixmap.height() - 1))

    def _source_to_display(self, point: tuple[int, int]) -> QPoint:
        rect = self._image_rect()
        if self._pixmap.width() <= 1 or self._pixmap.height() <= 1:
            return QPoint(rect.x(), rect.y())
        x = rect.x() + round(point[0] * (rect.width() - 1) / (self._pixmap.width() - 1))
        y = rect.y() + round(point[1] * (rect.height() - 1) / (self._pixmap.height() - 1))
        return QPoint(x, y)

    def _point_at(self, point: QPoint) -> int | None:
        for index, source_point in enumerate(self._points):
            if (self._source_to_display(source_point) - point).manhattanLength() <= 18:
                return index
        return None


class InspectionRegionPage(QWidget):
    """Present fixed ROI controls and original/overlay/rectified previews."""

    def __init__(self) -> None:
        super().__init__()
        self._source_paths: tuple[Path, ...] = ()
        self._source_index = -1
        self._source_size = (0, 0)
        self._saved_config = InspectionRegionConfig()
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 28)
        root.setSpacing(16)

        controls = QGroupBox("Inspection Region")
        controls_layout = QHBoxLayout(controls)
        self.enable_checkbox = QCheckBox("Enable fixed ROI")
        self.previous_button = QPushButton()
        self.previous_button.setObjectName("RoiPreviousButton")
        self.previous_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowBack))
        self.previous_button.setToolTip("Previous dataset image")
        self.next_button = QPushButton()
        self.next_button.setObjectName("RoiNextButton")
        self.next_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowForward))
        self.next_button.setToolTip("Next dataset image")
        self.random_button = QPushButton("Random")
        self.reset_button = QPushButton("Reset ROI")
        self.reset_button.setObjectName("AlertButton")
        self.save_button = QPushButton("Save ROI")
        self.save_button.setObjectName("PrimaryButton")
        controls_layout.addWidget(self.enable_checkbox)
        controls_layout.addSpacing(14)
        controls_layout.addWidget(self.previous_button)
        controls_layout.addWidget(self.next_button)
        controls_layout.addWidget(self.random_button)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.reset_button)
        controls_layout.addWidget(self.save_button)
        root.addWidget(controls)

        self.source_label = QLabel("No dataset image available")
        self.source_label.setObjectName("ModelSupport")
        self.source_label.setWordWrap(True)
        root.addWidget(self.source_label)

        previews = QHBoxLayout()
        self.original_preview = self._preview_panel("Original")
        self.overlay_canvas = InspectionRegionCanvas()
        self.overlay_canvas.setObjectName("DatasetThumbnail")
        self.rectified_preview = self._preview_panel("Rectified ROI")
        previews.addWidget(self.original_preview, stretch=1)
        previews.addWidget(self.overlay_canvas, stretch=2)
        previews.addWidget(self.rectified_preview, stretch=1)
        root.addLayout(previews, stretch=1)

        self.status_label = QLabel("ROI disabled")
        self.status_label.setObjectName("ModelSupport")
        root.addWidget(self.status_label)
        root.addStretch(1)

        self.enable_checkbox.toggled.connect(self._update_controls)
        self.previous_button.clicked.connect(lambda: self._select_source(-1))
        self.next_button.clicked.connect(lambda: self._select_source(1))
        self.random_button.clicked.connect(self._select_random_source)
        self.reset_button.clicked.connect(self._reset)
        self.overlay_canvas.points_changed.connect(self._update_rectified_preview)
        self._update_controls()

    def set_dataset_images(self, source_paths: tuple[Path, ...]) -> None:
        """Update source-image navigation without copying or cropping any source file."""
        self._source_paths = source_paths
        if not source_paths:
            self._source_index = -1
            self._load_current_source()
            return
        matching_index = next(
            (
                index
                for index, path in enumerate(source_paths)
                if self._image_size(path) == (self._saved_config.source_width, self._saved_config.source_height)
            ),
            0,
        )
        self._source_index = matching_index
        self._load_current_source()

    def set_inspection_region(self, config: InspectionRegionConfig) -> None:
        """Load saved ROI metadata into the editor without changing project state."""
        self._saved_config = config
        self.enable_checkbox.blockSignals(True)
        self.enable_checkbox.setChecked(config.enabled)
        self.enable_checkbox.blockSignals(False)
        self.overlay_canvas.set_points(config.points_px)
        if self._source_paths:
            self.set_dataset_images(self._source_paths)
        self._update_controls()
        self._update_rectified_preview()

    def inspection_region(self) -> InspectionRegionConfig:
        """Return validated editor data, keeping pixel corners authoritative."""
        if not self.enable_checkbox.isChecked():
            return InspectionRegionConfig()
        config = InspectionRegionConfig(
            enabled=True,
            source_width=self._source_size[0],
            source_height=self._source_size[1],
            points_px=self.overlay_canvas.points,
        )
        config.validate()
        return config

    @staticmethod
    def _preview_panel(title: str) -> QLabel:
        preview = QLabel(title)
        preview.setObjectName("DatasetThumbnail")
        preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview.setMinimumSize(240, 220)
        return preview

    def _update_controls(self) -> None:
        enabled = self.enable_checkbox.isChecked()
        has_sources = bool(self._source_paths)
        self.overlay_canvas.setEnabled(enabled and has_sources)
        for button in (self.previous_button, self.next_button, self.random_button, self.reset_button, self.save_button):
            button.setEnabled(has_sources or button is self.save_button)
        self.status_label.setText("Select four corners" if enabled else "ROI disabled")
        self._update_rectified_preview()

    def _select_source(self, offset: int) -> None:
        if self._source_paths:
            self._source_index = (self._source_index + offset) % len(self._source_paths)
            self._load_current_source()

    def _select_random_source(self) -> None:
        if self._source_paths:
            self._source_index = random.randrange(len(self._source_paths))
            self._load_current_source()

    def _reset(self) -> None:
        self.overlay_canvas.clear_points()
        self._update_rectified_preview()

    def _load_current_source(self) -> None:
        if self._source_index < 0 or not self._source_paths:
            self._source_size = (0, 0)
            self.source_label.setText("No dataset image available")
            self.original_preview.setText("Original")
            self.rectified_preview.setText("Rectified ROI")
            self.overlay_canvas.set_pixmap(QPixmap())
            self._update_controls()
            return
        source_path = self._source_paths[self._source_index]
        pixmap = QPixmap(str(source_path))
        self._source_size = (pixmap.width(), pixmap.height()) if not pixmap.isNull() else (0, 0)
        self.source_label.setText(f"{source_path.name}  |  {self._source_size[0]}x{self._source_size[1]}")
        self.source_label.setToolTip(str(source_path))
        self.overlay_canvas.set_pixmap(pixmap)
        self._set_preview(self.original_preview, pixmap, "Original")
        self._update_rectified_preview()

    def _update_rectified_preview(self, *_args: object) -> None:
        if not self.enable_checkbox.isChecked():
            self.rectified_preview.setPixmap(QPixmap())
            self.rectified_preview.setText("Rectified ROI")
            return
        try:
            config = self.inspection_region()
            source_path = self._source_paths[self._source_index]
            rectified = InspectionRegionProcessor(config).apply_path(source_path)
        except (IndexError, OSError, ValueError):
            self.rectified_preview.setPixmap(QPixmap())
            self.rectified_preview.setText("Rectified ROI")
            return
        image = QImage(
            np.ascontiguousarray(rectified).data,
            rectified.shape[1],
            rectified.shape[0],
            rectified.strides[0],
            QImage.Format.Format_RGB888,
        ).copy()
        self._set_preview(self.rectified_preview, QPixmap.fromImage(image), "Rectified ROI")
        width, height = config.rectified_size()
        self.status_label.setText(f"ROI {width}x{height} | TL, TR, BR, BL")

    @staticmethod
    def _set_preview(preview: QLabel, pixmap: QPixmap, placeholder: str) -> None:
        preview.setPixmap(
            pixmap.scaled(
                preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        preview.setText("" if not pixmap.isNull() else placeholder)

    @staticmethod
    def _image_size(path: Path) -> tuple[int, int]:
        pixmap = QPixmap(str(path))
        return pixmap.width(), pixmap.height()