"""Image viewer widgets for the pattern mixer GUI."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from src.utils.logging import get_logger

_LOGGER = get_logger(__name__)


@dataclass
class ViewLevels:
    """Contrast window levels."""

    min_value: float
    max_value: float


class LevelsDialog(QtWidgets.QDialog):
    """Dialog for adjusting contrast levels."""

    levels_changed = QtCore.Signal(float, float)

    def __init__(self, parent: QtWidgets.QWidget | None, levels: ViewLevels) -> None:
        super().__init__(parent)
        self.setWindowTitle("Levels")
        self.setModal(False)

        self.min_spin = QtWidgets.QDoubleSpinBox()
        self.min_spin.setDecimals(4)
        self.min_spin.setRange(-10.0, 10.0)
        self.min_spin.setValue(levels.min_value)

        self.max_spin = QtWidgets.QDoubleSpinBox()
        self.max_spin.setDecimals(4)
        self.max_spin.setRange(-10.0, 10.0)
        self.max_spin.setValue(levels.max_value)

        apply_button = QtWidgets.QPushButton("Apply")
        auto_button = QtWidgets.QPushButton("Auto")

        apply_button.clicked.connect(self._apply)
        auto_button.clicked.connect(self._auto)

        layout = QtWidgets.QFormLayout()
        layout.addRow("Min", self.min_spin)
        layout.addRow("Max", self.max_spin)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(apply_button)
        buttons.addWidget(auto_button)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.addLayout(layout)
        main_layout.addLayout(buttons)

    def _apply(self) -> None:
        self.levels_changed.emit(self.min_spin.value(), self.max_spin.value())

    def _auto(self) -> None:
        self.levels_changed.emit(float("nan"), float("nan"))


class ImageGraphicsView(QtWidgets.QGraphicsView):
    """Graphics view with zoom and pan support."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._zoom = 0

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if event.angleDelta().y() == 0:
            return
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)
        self._zoom += 1 if event.angleDelta().y() > 0 else -1

    def reset_zoom(self) -> None:
        self._zoom = 0
        self.resetTransform()


class ImageViewer(QtWidgets.QWidget):
    """Widget for displaying a single image with controls."""

    def __init__(self, title: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = title
        self._image: Optional[np.ndarray] = None
        self._levels = ViewLevels(0.0, 1.0)
        self._display_buffer: Optional[np.ndarray] = None
        self._qimage: Optional[QtGui.QImage] = None

        self._scene = QtWidgets.QGraphicsScene(self)
        self._pixmap_item = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)

        self._view = ImageGraphicsView()
        self._view.setScene(self._scene)
        self._view.viewport().installEventFilter(self)

        self._toolbar = QtWidgets.QHBoxLayout()
        self._toolbar.setContentsMargins(0, 0, 0, 0)
        self._toolbar.setSpacing(4)

        self._title_label = QtWidgets.QLabel(title)
        self._title_label.setStyleSheet("font-weight: 600;")

        self._zoom_in_btn = QtWidgets.QToolButton()
        self._zoom_in_btn.setText("+")
        self._zoom_out_btn = QtWidgets.QToolButton()
        self._zoom_out_btn.setText("-")
        self._fit_btn = QtWidgets.QToolButton()
        self._fit_btn.setText("Fit")
        self._reset_btn = QtWidgets.QToolButton()
        self._reset_btn.setText("1:1")
        self._levels_btn = QtWidgets.QToolButton()
        self._levels_btn.setText("Levels")

        self._toolbar.addWidget(self._title_label)
        self._toolbar.addStretch()
        self._toolbar.addWidget(self._zoom_in_btn)
        self._toolbar.addWidget(self._zoom_out_btn)
        self._toolbar.addWidget(self._fit_btn)
        self._toolbar.addWidget(self._reset_btn)
        self._toolbar.addWidget(self._levels_btn)

        self._pixel_label = QtWidgets.QLabel("x: -, y: -, val: -")
        self._pixel_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self._pixel_label.setStyleSheet("color: #666;")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addLayout(self._toolbar)
        layout.addWidget(self._view, stretch=1)
        layout.addWidget(self._pixel_label)

        self._zoom_in_btn.clicked.connect(lambda: self._view.scale(1.25, 1.25))
        self._zoom_out_btn.clicked.connect(lambda: self._view.scale(0.8, 0.8))
        self._fit_btn.clicked.connect(self.fit_to_window)
        self._reset_btn.clicked.connect(self.reset_view)
        self._levels_btn.clicked.connect(self._open_levels)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if obj is self._view.viewport() and event.type() == QtCore.QEvent.Type.MouseMove:
            self._update_pixel_readout(event)
        return super().eventFilter(obj, event)

    def _update_pixel_readout(self, event: QtCore.QEvent) -> None:
        if self._image is None:
            return
        mouse_event = event  # type: ignore[assignment]
        pos = self._view.mapToScene(mouse_event.position().toPoint())
        x = int(pos.x())
        y = int(pos.y())
        if 0 <= y < self._image.shape[0] and 0 <= x < self._image.shape[1]:
            value = float(self._image[y, x])
            self._pixel_label.setText(f"x: {x}, y: {y}, val: {value:.4f}")
        else:
            self._pixel_label.setText("x: -, y: -, val: -")

    def _open_levels(self) -> None:
        dialog = LevelsDialog(self, self._levels)
        dialog.levels_changed.connect(self._set_levels)
        dialog.show()

    def _set_levels(self, min_value: float, max_value: float) -> None:
        if np.isnan(min_value) or np.isnan(max_value):
            self.auto_levels()
        else:
            if max_value <= min_value:
                return
            self._levels = ViewLevels(min_value, max_value)
            self._refresh_pixmap()

    def set_image(self, image: Optional[np.ndarray]) -> None:
        self._image = image
        if image is None:
            self._pixmap_item.setPixmap(QtGui.QPixmap())
            return
        self.auto_levels()
        self.fit_to_window()

    def auto_levels(self) -> None:
        if self._image is None:
            return
        min_value = float(np.min(self._image))
        max_value = float(np.max(self._image))
        if max_value <= min_value:
            max_value = min_value + 1e-6
        self._levels = ViewLevels(min_value, max_value)
        self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        if self._image is None:
            return
        min_val = self._levels.min_value
        max_val = self._levels.max_value
        denom = max(max_val - min_val, 1e-8)
        scaled = np.clip((self._image - min_val) / denom, 0.0, 1.0)
        self._display_buffer = (scaled * 255.0).astype(np.uint8)
        height, width = self._display_buffer.shape
        self._qimage = QtGui.QImage(
            self._display_buffer.data,
            width,
            height,
            self._display_buffer.strides[0],
            QtGui.QImage.Format.Format_Grayscale8,
        )
        self._pixmap_item.setPixmap(QtGui.QPixmap.fromImage(self._qimage))
        self._scene.setSceneRect(0, 0, width, height)

    def fit_to_window(self) -> None:
        if self._image is None:
            return
        self._view.fitInView(self._scene.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def reset_view(self) -> None:
        self._view.reset_zoom()
        self._view.centerOn(self._pixmap_item)

    def title(self) -> str:
        return self._title

    def set_title(self, title: str) -> None:
        self._title = title
        self._title_label.setText(title)

    def current_levels(self) -> Tuple[float, float]:
        return self._levels.min_value, self._levels.max_value

    def image_data(self) -> Optional[np.ndarray]:
        return self._image
