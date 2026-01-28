"""Main window for the pattern mixer GUI."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6 import QtCore, QtWidgets

from src.utils.logging import get_logger

from ..core.config import MixSettings, NoiseSettings, NormalizationMode
from ..core.image_io import LoadedImage, load_image, resize_to_match
from ..core.processing import ProcessedImages
from ..core.strategies import build_strategy_registry
from .logging_handler import LogPanel
from .viewer import ImageViewer

_LOGGER = get_logger(__name__)


@dataclass
class LoadedState:
    """Hold loaded images and masks."""

    image_a: Optional[LoadedImage] = None
    image_b: Optional[LoadedImage] = None


class PatternMixerWindow(QtWidgets.QMainWindow):
    """Main GUI window for pattern mixing."""

    def __init__(self, debug: bool = False) -> None:
        super().__init__()
        self.setWindowTitle("Kikuchi Pattern Mixing Playground")
        self.resize(1400, 900)

        self._state = LoadedState()
        self._settings = MixSettings(seed=123 if debug else None)
        self._noise_a = NoiseSettings()
        self._noise_b = NoiseSettings()
        self._registry = build_strategy_registry()
        self._strategy_keys = list(self._registry.keys())
        self._current_strategy = self._registry[self._strategy_keys[0]]

        self._update_timer = QtCore.QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(80)
        self._update_timer.timeout.connect(self._refresh_processing)

        self._viewer_a = ImageViewer("Pattern A")
        self._viewer_b = ImageViewer("Pattern B")
        self._viewer_c = ImageViewer("Mixed C")

        self._log_panel = LogPanel()

        self._build_layout()
        self._build_controls()

    def _build_layout(self) -> None:
        central = QtWidgets.QWidget()
        central_layout = QtWidgets.QVBoxLayout(central)
        central_layout.setContentsMargins(4, 4, 4, 4)
        central_layout.setSpacing(4)

        image_grid = QtWidgets.QGridLayout()
        image_grid.setContentsMargins(0, 0, 0, 0)
        image_grid.setSpacing(6)
        image_grid.addWidget(self._viewer_a, 0, 0)
        image_grid.addWidget(self._viewer_b, 0, 1)
        image_grid.addWidget(self._viewer_c, 1, 0, 1, 2)

        image_container = QtWidgets.QWidget()
        image_container.setLayout(image_grid)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        splitter.addWidget(image_container)
        splitter.addWidget(self._log_panel)
        splitter.setStretchFactor(0, 9)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([800, 120])

        central_layout.addWidget(splitter)
        self.setCentralWidget(central)

    def _build_controls(self) -> None:
        dock = QtWidgets.QDockWidget("Controls", self)
        dock.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
            | QtCore.Qt.DockWidgetArea.RightDockWidgetArea
        )
        controls = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(controls)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        self._load_a_btn = QtWidgets.QPushButton("Load Pattern A")
        self._load_b_btn = QtWidgets.QPushButton("Load Pattern B")
        self._path_a_label = QtWidgets.QLabel("No file loaded")
        self._path_b_label = QtWidgets.QLabel("No file loaded")
        self._path_a_label.setWordWrap(True)
        self._path_b_label.setWordWrap(True)

        self._load_a_btn.clicked.connect(lambda: self._load_image("A"))
        self._load_b_btn.clicked.connect(lambda: self._load_image("B"))

        layout.addWidget(self._load_a_btn)
        layout.addWidget(self._path_a_label)
        layout.addWidget(self._load_b_btn)
        layout.addWidget(self._path_b_label)

        self._weight_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._weight_slider.setRange(0, 100)
        self._weight_slider.setValue(int(self._settings.weight_a * 100))
        self._weight_value = QtWidgets.QLabel(f"{self._settings.weight_a:.2f}")

        self._weight_slider.valueChanged.connect(self._on_weight_change)

        weight_layout = QtWidgets.QHBoxLayout()
        weight_layout.addWidget(QtWidgets.QLabel("Weight A"))
        weight_layout.addWidget(self._weight_slider, stretch=1)
        weight_layout.addWidget(self._weight_value)
        layout.addLayout(weight_layout)

        self._strategy_combo = QtWidgets.QComboBox()
        for strategy in self._registry.values():
            self._strategy_combo.addItem(strategy.label, strategy.key)
        self._strategy_combo.currentIndexChanged.connect(self._on_strategy_change)
        layout.addWidget(QtWidgets.QLabel("Mixing Strategy"))
        layout.addWidget(self._strategy_combo)

        self._norm_combo = QtWidgets.QComboBox()
        self._norm_combo.addItem("Min/Max (mask-aware)", NormalizationMode.MIN_MAX)
        self._norm_combo.addItem("Per-image min/max", NormalizationMode.PER_IMAGE_MIN_MAX)
        self._norm_combo.addItem("Z-score remap", NormalizationMode.ZSCORE_REMAP)
        self._norm_combo.addItem("No normalization", NormalizationMode.NONE)
        self._norm_combo.currentIndexChanged.connect(self._on_norm_change)
        layout.addWidget(QtWidgets.QLabel("Normalization Mode"))
        layout.addWidget(self._norm_combo)

        self._noise_group_a = self._build_noise_group(
            "Noise A",
            self._noise_a,
            self._on_noise_change,
        )
        self._noise_group_b = self._build_noise_group(
            "Noise B",
            self._noise_b,
            self._on_noise_change,
        )
        layout.addWidget(self._noise_group_a)
        layout.addWidget(self._noise_group_b)

        self._export_btn = QtWidgets.QPushButton("Export Mixed C as PNG")
        self._export_btn.clicked.connect(self._export_image)
        layout.addWidget(self._export_btn)

        layout.addStretch()
        dock.setWidget(controls)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _build_noise_group(
        self,
        title: str,
        settings: NoiseSettings,
        change_callback,
    ) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox(title)
        group.setCheckable(False)
        layout = QtWidgets.QFormLayout(group)
        layout.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)

        gaussian_check = QtWidgets.QCheckBox("Gaussian")
        gaussian_sigma = QtWidgets.QDoubleSpinBox()
        gaussian_sigma.setRange(0.0, 0.5)
        gaussian_sigma.setDecimals(4)
        gaussian_sigma.setSingleStep(0.001)
        gaussian_sigma.setValue(settings.gaussian_sigma)

        poisson_check = QtWidgets.QCheckBox("Poisson-like")
        poisson_scale = QtWidgets.QDoubleSpinBox()
        poisson_scale.setRange(100.0, 100000.0)
        poisson_scale.setDecimals(0)
        poisson_scale.setSingleStep(500.0)
        poisson_scale.setValue(settings.poisson_scale)

        offset_check = QtWidgets.QCheckBox("Offset")
        offset_value = QtWidgets.QDoubleSpinBox()
        offset_value.setRange(-1.0, 1.0)
        offset_value.setDecimals(4)
        offset_value.setSingleStep(0.001)
        offset_value.setValue(settings.offset_value)

        def update_settings() -> None:
            settings.enable_gaussian = gaussian_check.isChecked()
            settings.gaussian_sigma = gaussian_sigma.value()
            settings.enable_poisson = poisson_check.isChecked()
            settings.poisson_scale = poisson_scale.value()
            settings.enable_offset = offset_check.isChecked()
            settings.offset_value = offset_value.value()
            change_callback()

        gaussian_check.toggled.connect(update_settings)
        gaussian_sigma.valueChanged.connect(lambda _: update_settings())
        poisson_check.toggled.connect(update_settings)
        poisson_scale.valueChanged.connect(lambda _: update_settings())
        offset_check.toggled.connect(update_settings)
        offset_value.valueChanged.connect(lambda _: update_settings())

        layout.addRow(gaussian_check, gaussian_sigma)
        layout.addRow(poisson_check, poisson_scale)
        layout.addRow(offset_check, offset_value)
        return group

    def _load_image(self, slot: str) -> None:
        path_str, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Pattern",
            str(Path.cwd()),
            "Images (*.png *.bmp *.jpg *.jpeg *.tif *.tiff)",
        )
        if not path_str:
            return
        path = Path(path_str)
        try:
            loaded = load_image(path)
        except Exception as exc:  # pragma: no cover - GUI feedback
            QtWidgets.QMessageBox.critical(self, "Load Error", str(exc))
            _LOGGER.error("Failed to load %s: %s", path, exc)
            return

        if slot == "A":
            self._state.image_a = loaded
            self._path_a_label.setText(str(path))
            self._viewer_a.set_image(loaded.data)
        else:
            self._state.image_b = loaded
            self._path_b_label.setText(str(path))
            self._viewer_b.set_image(loaded.data)

        self._schedule_update()

    def _on_weight_change(self, value: int) -> None:
        weight = value / 100.0
        self._settings.weight_a = weight
        self._weight_value.setText(f"{weight:.2f}")
        self._schedule_update()

    def _on_strategy_change(self, index: int) -> None:
        key = self._strategy_combo.currentData()
        if key in self._registry:
            self._current_strategy = self._registry[key]
        self._schedule_update()

    def _on_norm_change(self, index: int) -> None:
        mode = self._norm_combo.currentData()
        if isinstance(mode, NormalizationMode):
            self._settings.normalization_mode = mode
        self._schedule_update()

    def _on_noise_change(self) -> None:
        self._schedule_update()

    def _schedule_update(self) -> None:
        if self._update_timer.isActive():
            self._update_timer.stop()
        self._update_timer.start()

    def _refresh_processing(self) -> None:
        if self._state.image_a is None or self._state.image_b is None:
            return

        image_a = self._state.image_a.data
        image_b = self._state.image_b.data
        mask_a = self._state.image_a.mask
        mask_b = self._state.image_b.mask

        if image_a.shape != image_b.shape:
            _LOGGER.info(
                "Resizing B from %s to %s to match A.",
                image_b.shape,
                image_a.shape,
            )
            image_b = resize_to_match(image_b, image_a.shape)
            mask_b = resize_to_match(mask_b.astype(np.float32), image_a.shape) > 0.5

        processed = self._current_strategy.apply(
            image_a=image_a,
            image_b=image_b,
            mask_a=mask_a,
            mask_b=mask_b,
            weight_a=self._settings.weight_a,
            normalization_mode=self._settings.normalization_mode,
            noise_a=self._noise_a,
            noise_b=self._noise_b,
            seed=self._settings.seed,
        )

        self._update_viewers(processed)

        _LOGGER.info(
            "Updated mix | strategy=%s | weight_a=%.3f | norm=%s",
            self._current_strategy.key,
            self._settings.weight_a,
            self._settings.normalization_mode.value,
        )

    def _update_viewers(self, processed: ProcessedImages) -> None:
        self._viewer_a.set_image(processed.image_a)
        self._viewer_b.set_image(processed.image_b)
        self._viewer_c.set_image(processed.image_c)

    def _export_image(self) -> None:
        if self._state.image_a is None or self._state.image_b is None:
            QtWidgets.QMessageBox.warning(self, "Export", "Load both A and B first.")
            return

        image_c = self._viewer_c.image_data()
        if image_c is None:
            QtWidgets.QMessageBox.warning(self, "Export", "No mixed image available.")
            return
        path_str, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Mixed C",
            "mixed_c.png",
            "PNG Images (*.png)",
        )
        if not path_str:
            return
        path = Path(path_str)
        export = np.clip(image_c, 0.0, 1.0)
        export_16bit = (export * 65535.0).astype(np.uint16)
        from PIL import Image

        Image.fromarray(export_16bit).save(path)
        _LOGGER.info("Exported mixed image to %s", path)
        QtWidgets.QMessageBox.information(self, "Export", f"Saved to {path}")
