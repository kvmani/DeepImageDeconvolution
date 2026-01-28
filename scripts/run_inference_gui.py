"""Desktop GUI for running inference."""
from __future__ import annotations

import csv
import json
import logging
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QGraphicsScene,
    QGraphicsView,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.api import (  # noqa: E402
    InferenceRunResult,
    build_gt_lookup,
    merge_inference_config,
    run_inference_core,
)
from src.inference.model_manager import ModelManager  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.io import read_image_16bit, to_float01  # noqa: E402
from src.utils.logging import (  # noqa: E402
    StructuredFormatter,
    get_git_commit,
    setup_logging,
    write_manifest,
)


DEFAULT_CONFIG = REPO_ROOT / "configs/infer_default.yaml"


@dataclass
class GuiInferenceState:
    """Container for the latest inference outputs."""

    result: Optional[InferenceRunResult] = None
    config: Optional[Dict[str, Any]] = None
    input_map: Optional[Dict[str, Path]] = None
    gt_lookup: Optional[Dict[str, Dict[str, Path]]] = None


class GuiLogHandler(QObject, logging.Handler):
    """Send log messages to a Qt signal."""

    log_message = Signal(str, int)

    def __init__(self) -> None:
        QObject.__init__(self)
        logging.Handler.__init__(self)

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self.log_message.emit(msg, record.levelno)


class ViewLinker:
    """Synchronize pan/zoom across views."""

    def __init__(self) -> None:
        self._views: List["LinkedImageView"] = []
        self._syncing = False

    def register(self, view: "LinkedImageView") -> None:
        self._views.append(view)

    def sync(self, source: "LinkedImageView") -> None:
        if self._syncing:
            return
        self._syncing = True
        for view in self._views:
            if view is source:
                continue
            view.setTransform(source.transform())
            view.horizontalScrollBar().setValue(source.horizontalScrollBar().value())
            view.verticalScrollBar().setValue(source.verticalScrollBar().value())
        self._syncing = False


class LinkedImageView(QGraphicsView):
    """Image view with linked pan/zoom."""

    def __init__(self, title: str, linker: ViewLinker) -> None:
        super().__init__()
        self._title = title
        self._linker = linker
        self._image: Optional[np.ndarray] = None
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = self._scene.addPixmap(QPixmap())
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setMinimumSize(200, 200)
        self.setStyleSheet("border: 1px solid #444;")
        self._linker.register(self)
        self.horizontalScrollBar().valueChanged.connect(lambda _: self._linker.sync(self))
        self.verticalScrollBar().valueChanged.connect(lambda _: self._linker.sync(self))

    def set_image(self, image: Optional[np.ndarray], low: float, high: float) -> None:
        """Set a new image with contrast scaling."""
        self._image = image
        if image is None:
            self._scene.clear()
            self._scene.addText(self._title)
            return
        scaled = _scale_for_display(image, low, high)
        qimage = QImage(
            scaled.data,
            scaled.shape[1],
            scaled.shape[0],
            scaled.strides[0],
            QImage.Format_Grayscale8,
        )
        pixmap = QPixmap.fromImage(qimage.copy())
        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self.setSceneRect(pixmap.rect())
        self.resetTransform()

    def wheelEvent(self, event) -> None:  # noqa: D401
        zoom_factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(zoom_factor, zoom_factor)
        self._linker.sync(self)

    def mouseMoveEvent(self, event) -> None:  # noqa: D401
        super().mouseMoveEvent(event)
        if self._image is None:
            return
        scene_pos = self.mapToScene(event.pos())
        x = int(scene_pos.x())
        y = int(scene_pos.y())
        if 0 <= y < self._image.shape[0] and 0 <= x < self._image.shape[1]:
            value = float(self._image[y, x])
            window = self.window()
            if hasattr(window, "update_pixel_readout"):
                window.update_pixel_readout(self._title, x, y, value)


class ImagePanel(QWidget):
    """Labeled image panel."""

    def __init__(self, title: str, linker: ViewLinker) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(title))
        self.view = LinkedImageView(title, linker)
        layout.addWidget(self.view, stretch=1)

    def set_image(self, image: Optional[np.ndarray], low: float, high: float) -> None:
        self.view.set_image(image, low, high)


class DiagnosticsPanel(QWidget):
    """Panel showing diagnostics metrics and diff map."""

    def __init__(self, linker: ViewLinker) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        self.metrics_label = QLabel("Diagnostics")
        self.metrics_label.setWordWrap(True)
        self.image_view = LinkedImageView("Diagnostics", linker)
        layout.addWidget(self.metrics_label)
        layout.addWidget(self.image_view, stretch=1)

    def update_metrics(self, text: str) -> None:
        self.metrics_label.setText(text)

    def set_image(self, image: Optional[np.ndarray], low: float, high: float) -> None:
        self.image_view.set_image(image, low, high)


class InferenceWorker(QThread):
    """Background worker for inference."""

    progress = Signal(int, int, float)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        config: Dict[str, Any],
        input_paths: List[Path],
        gt_lookup: Optional[Dict[str, Dict[str, Path]]],
        model_manager: ModelManager,
        output_dir: Path,
    ) -> None:
        super().__init__()
        self._config = config
        self._input_paths = input_paths
        self._gt_lookup = gt_lookup
        self._model_manager = model_manager
        self._output_dir = output_dir
        self._stop_event = threading.Event()

    def run(self) -> None:
        logger = logging.getLogger("inference_gui")
        start_time = time.perf_counter()
        try:
            result = run_inference_core(
                self._config,
                input_paths=self._input_paths,
                output_dir=self._output_dir,
                model_manager=self._model_manager,
                progress_callback=self._handle_progress,
                stop_event=self._stop_event,
                gt_lookup=self._gt_lookup,
            )
            duration = time.perf_counter() - start_time
            self._write_manifest(result, duration)
            self._write_metrics(result)
            self.finished.emit(result)
        except Exception as exc:
            logger.exception("GUI inference failed: %s", exc)
            self.failed.emit(str(exc))

    def stop(self) -> None:
        self._stop_event.set()

    def _handle_progress(self, snapshot) -> None:
        self.progress.emit(snapshot.processed, snapshot.total, snapshot.eta_s)

    def _write_manifest(self, result: InferenceRunResult, duration: float) -> None:
        manifest = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "git_commit": get_git_commit(REPO_ROOT),
            "config": self._config,
            "checkpoint": self._config.get("inference", {}).get("checkpoint"),
            "device": self._config.get("inference", {}).get("device", "auto"),
            "timings": {"wall_time_s": duration},
            "summary": {
                "processed": result.processed,
                "failed": result.failed,
                "cancelled": result.cancelled,
                "output_counts": result.output_counts,
                "metrics_summary": result.sample_metrics_summary,
            },
        }
        write_manifest(result.output_dir, manifest, filename="run_manifest.json")

    def _write_metrics(self, result: InferenceRunResult) -> None:
        rows = []
        for sample in result.sample_results:
            if not sample.metrics:
                continue
            row = {"sample_id": sample.sample_id}
            row.update(sample.metrics)
            rows.append(row)
        if not rows:
            return
        metrics_csv = result.output_dir / "metrics.csv"
        metrics_json = result.output_dir / "metrics.json"
        fieldnames = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        if "sample_id" in fieldnames:
            fieldnames.remove("sample_id")
            fieldnames.insert(0, "sample_id")
        with metrics_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fieldnames})
        payload = {
            "summary": result.sample_metrics_summary,
            "samples": rows,
        }
        metrics_json.write_text(json.dumps(payload, indent=2))


class InferenceGui(QMainWindow):
    """Main inference GUI window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DeepImageDeconvolution Inference GUI")
        self.resize(1400, 900)
        self._state = GuiInferenceState()
        self._model_manager = ModelManager()
        self._worker: Optional[InferenceWorker] = None
        self._current_sample = None

        self._init_logging()
        self._build_ui()
        self._load_default_config()

    def _init_logging(self) -> None:
        setup_logging("inference_gui", level="INFO")
        self._log_handler = GuiLogHandler()
        self._log_handler.setFormatter(StructuredFormatter())
        logging.getLogger().addHandler(self._log_handler)
        self._log_handler.log_message.connect(self._append_log)

    def _build_ui(self) -> None:
        central = QWidget()
        main_layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_control_panel())
        splitter.addWidget(self._build_viewer_panel())
        splitter.setStretchFactor(1, 3)
        main_layout.addWidget(splitter)

        self.setCentralWidget(central)

    def _build_control_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.config_path_edit = QLineEdit()
        config_btn = QPushButton("Browse")
        config_btn.clicked.connect(self._choose_config)
        config_layout = QHBoxLayout()
        config_layout.addWidget(self.config_path_edit)
        config_layout.addWidget(config_btn)

        config_group = QGroupBox("Config")
        config_form = QFormLayout(config_group)
        config_form.addRow("YAML config", config_layout)

        self.checkpoint_edit = QLineEdit()
        ckpt_btn = QPushButton("Browse")
        ckpt_btn.clicked.connect(lambda: self._choose_file(self.checkpoint_edit))
        ckpt_layout = QHBoxLayout()
        ckpt_layout.addWidget(self.checkpoint_edit)
        ckpt_layout.addWidget(ckpt_btn)

        self.output_dir_edit = QLineEdit()
        out_btn = QPushButton("Browse")
        out_btn.clicked.connect(lambda: self._choose_directory(self.output_dir_edit))
        out_layout = QHBoxLayout()
        out_layout.addWidget(self.output_dir_edit)
        out_layout.addWidget(out_btn)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu", "cuda"])

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 128)
        self.batch_size_spin.setValue(4)

        self.clamp_checkbox = QCheckBox("Clamp outputs to [0, 1]")
        self.clamp_checkbox.setChecked(True)

        self.save_recon_checkbox = QCheckBox("Save reconstructed C_hat")
        self.save_recon_checkbox.setChecked(True)

        self.save_weights_checkbox = QCheckBox("Save weights.csv")
        self.save_weights_checkbox.setChecked(True)

        config_form.addRow("Checkpoint", ckpt_layout)
        config_form.addRow("Output dir", out_layout)
        config_form.addRow("Device", self.device_combo)
        config_form.addRow("Batch size", self.batch_size_spin)
        config_form.addRow("", self.clamp_checkbox)
        config_form.addRow("", self.save_recon_checkbox)
        config_form.addRow("", self.save_weights_checkbox)

        input_group = QGroupBox("Inputs")
        input_layout = QFormLayout(input_group)
        self.single_mode_checkbox = QCheckBox("Single image mode")
        self.single_mode_checkbox.setChecked(True)
        self.single_mode_checkbox.stateChanged.connect(self._toggle_input_mode)

        self.input_c_edit = QLineEdit()
        input_c_btn = QPushButton("Browse")
        input_c_btn.clicked.connect(lambda: self._choose_file(self.input_c_edit))
        input_c_layout = QHBoxLayout()
        input_c_layout.addWidget(self.input_c_edit)
        input_c_layout.addWidget(input_c_btn)

        self.batch_dir_edit = QLineEdit()
        batch_dir_btn = QPushButton("Browse")
        batch_dir_btn.clicked.connect(lambda: self._choose_directory(self.batch_dir_edit))
        batch_dir_layout = QHBoxLayout()
        batch_dir_layout.addWidget(self.batch_dir_edit)
        batch_dir_layout.addWidget(batch_dir_btn)

        self.batch_pattern_edit = QLineEdit("*.png")
        self.batch_recursive_checkbox = QCheckBox("Recursive")

        input_layout.addRow(self.single_mode_checkbox)
        input_layout.addRow("Input C", input_c_layout)
        input_layout.addRow("Batch dir", batch_dir_layout)
        input_layout.addRow("Batch pattern", self.batch_pattern_edit)
        input_layout.addRow("", self.batch_recursive_checkbox)

        gt_group = QGroupBox("Ground Truth (optional)")
        gt_layout = QFormLayout(gt_group)
        self.gt_a_edit = QLineEdit()
        gt_a_btn = QPushButton("Browse")
        gt_a_btn.clicked.connect(lambda: self._choose_file(self.gt_a_edit))
        gt_a_layout = QHBoxLayout()
        gt_a_layout.addWidget(self.gt_a_edit)
        gt_a_layout.addWidget(gt_a_btn)

        self.gt_b_edit = QLineEdit()
        gt_b_btn = QPushButton("Browse")
        gt_b_btn.clicked.connect(lambda: self._choose_file(self.gt_b_edit))
        gt_b_layout = QHBoxLayout()
        gt_b_layout.addWidget(self.gt_b_edit)
        gt_b_layout.addWidget(gt_b_btn)

        self.gt_dir_a_edit = QLineEdit()
        gt_dir_a_btn = QPushButton("Browse")
        gt_dir_a_btn.clicked.connect(lambda: self._choose_directory(self.gt_dir_a_edit))
        gt_dir_a_layout = QHBoxLayout()
        gt_dir_a_layout.addWidget(self.gt_dir_a_edit)
        gt_dir_a_layout.addWidget(gt_dir_a_btn)

        self.gt_dir_b_edit = QLineEdit()
        gt_dir_b_btn = QPushButton("Browse")
        gt_dir_b_btn.clicked.connect(lambda: self._choose_directory(self.gt_dir_b_edit))
        gt_dir_b_layout = QHBoxLayout()
        gt_dir_b_layout.addWidget(self.gt_dir_b_edit)
        gt_dir_b_layout.addWidget(gt_dir_b_btn)

        self.gt_pattern_edit = QLineEdit("*.png")
        self.gt_recursive_checkbox = QCheckBox("Recursive")

        gt_layout.addRow("GT A (single)", gt_a_layout)
        gt_layout.addRow("GT B (single)", gt_b_layout)
        gt_layout.addRow("GT A dir", gt_dir_a_layout)
        gt_layout.addRow("GT B dir", gt_dir_b_layout)
        gt_layout.addRow("GT pattern", self.gt_pattern_edit)
        gt_layout.addRow("", self.gt_recursive_checkbox)

        preprocess_group = QGroupBox("Preprocess")
        preprocess_layout = QFormLayout(preprocess_group)
        self.mask_checkbox = QCheckBox("Apply circular mask")
        self.mask_checkbox.setChecked(True)
        self.normalize_checkbox = QCheckBox("Normalize inputs")
        self.normalize_checkbox.setChecked(False)
        self.norm_low_spin = QDoubleSpinBox()
        self.norm_low_spin.setRange(0.0, 100.0)
        self.norm_low_spin.setValue(1.0)
        self.norm_high_spin = QDoubleSpinBox()
        self.norm_high_spin.setRange(0.0, 100.0)
        self.norm_high_spin.setValue(99.0)
        preprocess_layout.addRow("", self.mask_checkbox)
        preprocess_layout.addRow("", self.normalize_checkbox)
        preprocess_layout.addRow("Percentile low", self.norm_low_spin)
        preprocess_layout.addRow("Percentile high", self.norm_high_spin)

        self.run_button = QPushButton("Run Inference")
        self.run_button.clicked.connect(self._run_inference)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._cancel_inference)
        self.cancel_button.setEnabled(False)

        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("Idle")

        layout.addWidget(config_group)
        layout.addWidget(input_group)
        layout.addWidget(gt_group)
        layout.addWidget(preprocess_group)
        layout.addWidget(self.run_button)
        layout.addWidget(self.cancel_button)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.progress_label)
        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        return scroll

    def _build_viewer_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        contrast_layout = QHBoxLayout()
        self.contrast_low_spin = QDoubleSpinBox()
        self.contrast_low_spin.setRange(0.0, 100.0)
        self.contrast_low_spin.setValue(1.0)
        self.contrast_high_spin = QDoubleSpinBox()
        self.contrast_high_spin.setRange(0.0, 100.0)
        self.contrast_high_spin.setValue(99.0)
        self.contrast_low_spin.valueChanged.connect(self._refresh_views)
        self.contrast_high_spin.valueChanged.connect(self._refresh_views)
        self.pixel_label = QLabel("Pixel: -")
        contrast_layout.addWidget(QLabel("Preview low %"))
        contrast_layout.addWidget(self.contrast_low_spin)
        contrast_layout.addWidget(QLabel("Preview high %"))
        contrast_layout.addWidget(self.contrast_high_spin)
        contrast_layout.addStretch()
        contrast_layout.addWidget(self.pixel_label)

        self._linker = ViewLinker()
        self.input_view = ImagePanel("Input C", self._linker)
        self.a_view = ImagePanel("Pred A_hat", self._linker)
        self.b_view = ImagePanel("Pred B_hat", self._linker)
        self.diagnostics_panel = DiagnosticsPanel(self._linker)

        grid = QGridLayout()
        grid.addWidget(self.input_view, 0, 0)
        grid.addWidget(self.a_view, 0, 1)
        grid.addWidget(self.b_view, 1, 0)
        grid.addWidget(self.diagnostics_panel, 1, 1)

        self.metrics_table = QTableWidget(0, 9)
        self.metrics_table.setHorizontalHeaderLabels(
            [
                "sample_id",
                "a_l1",
                "a_l2",
                "a_psnr",
                "a_ssim",
                "b_l1",
                "b_l2",
                "b_psnr",
                "b_ssim",
            ]
        )
        self.metrics_table.itemSelectionChanged.connect(self._on_metric_selection)

        log_group = QGroupBox("Logs")
        log_layout = QVBoxLayout(log_group)
        log_controls = QHBoxLayout()
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self.log_level_combo.setCurrentText("INFO")
        copy_btn = QPushButton("Copy")
        copy_btn.clicked.connect(self._copy_logs)
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._save_logs)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_logs)
        log_controls.addWidget(QLabel("Level"))
        log_controls.addWidget(self.log_level_combo)
        log_controls.addStretch()
        log_controls.addWidget(copy_btn)
        log_controls.addWidget(save_btn)
        log_controls.addWidget(clear_btn)

        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addLayout(log_controls)
        log_layout.addWidget(self.log_text)

        layout.addLayout(contrast_layout)
        layout.addLayout(grid, stretch=3)
        layout.addWidget(self.metrics_table)
        layout.addWidget(log_group, stretch=1)
        return panel

    def _load_default_config(self) -> None:
        self.config_path_edit.setText(str(DEFAULT_CONFIG))
        self._load_config()

    def _load_config(self) -> None:
        config_path = Path(self.config_path_edit.text())
        config = load_config(config_path)
        self._state.config = config
        inference_cfg = config.get("inference", {})
        output_cfg = config.get("output", {})
        data_cfg = config.get("data", {})
        preprocess_cfg = data_cfg.get("preprocess", {})
        mask_cfg = preprocess_cfg.get("mask", {})
        normalize_cfg = preprocess_cfg.get("normalize", {})

        self.checkpoint_edit.setText(str(inference_cfg.get("checkpoint", "")))
        self.output_dir_edit.setText(str(output_cfg.get("out_dir", "")))
        self.batch_size_spin.setValue(int(inference_cfg.get("batch_size", 4)))
        self.clamp_checkbox.setChecked(bool(inference_cfg.get("clamp_outputs", True)))
        self.save_recon_checkbox.setChecked(bool(output_cfg.get("save_recon", True)))
        self.save_weights_checkbox.setChecked(bool(output_cfg.get("save_weights", True)))
        self.device_combo.setCurrentText(str(inference_cfg.get("device", "auto")))
        self.mask_checkbox.setChecked(bool(mask_cfg.get("enabled", True)))
        self.normalize_checkbox.setChecked(bool(normalize_cfg.get("enabled", False)))
        if "percentile" in normalize_cfg:
            self.norm_low_spin.setValue(float(normalize_cfg["percentile"][0]))
            self.norm_high_spin.setValue(float(normalize_cfg["percentile"][1]))

    def _choose_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select config", str(REPO_ROOT), "YAML files (*.yaml *.yml)"
        )
        if path:
            self.config_path_edit.setText(path)
            self._load_config()

    def _choose_file(self, line_edit: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select file", str(REPO_ROOT))
        if path:
            line_edit.setText(path)

    def _choose_directory(self, line_edit: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select folder", str(REPO_ROOT))
        if path:
            line_edit.setText(path)

    def _toggle_input_mode(self) -> None:
        is_single = self.single_mode_checkbox.isChecked()
        self.input_c_edit.setEnabled(is_single)
        self.batch_dir_edit.setEnabled(not is_single)
        self.batch_pattern_edit.setEnabled(not is_single)
        self.batch_recursive_checkbox.setEnabled(not is_single)
        self.gt_a_edit.setEnabled(is_single)
        self.gt_b_edit.setEnabled(is_single)
        self.gt_dir_a_edit.setEnabled(not is_single)
        self.gt_dir_b_edit.setEnabled(not is_single)
        self.gt_pattern_edit.setEnabled(not is_single)
        self.gt_recursive_checkbox.setEnabled(not is_single)

    def _build_gui_overrides(self) -> Dict[str, Any]:
        overrides: Dict[str, Any] = {
            "inference": {
                "checkpoint": self.checkpoint_edit.text().strip(),
                "batch_size": int(self.batch_size_spin.value()),
                "clamp_outputs": bool(self.clamp_checkbox.isChecked()),
                "device": self.device_combo.currentText(),
            },
            "output": {
                "out_dir": self.output_dir_edit.text().strip(),
                "save_recon": bool(self.save_recon_checkbox.isChecked()),
                "save_weights": bool(self.save_weights_checkbox.isChecked()),
            },
            "data": {
                "preprocess": {
                    "mask": {"enabled": bool(self.mask_checkbox.isChecked())},
                    "normalize": {
                        "enabled": bool(self.normalize_checkbox.isChecked()),
                        "percentile": [
                            float(self.norm_low_spin.value()),
                            float(self.norm_high_spin.value()),
                        ],
                    },
                },
            },
            "postprocess": {"apply_mask": bool(self.mask_checkbox.isChecked())},
        }
        return overrides

    def _collect_input_paths(self, config: Dict[str, Any]) -> List[Path]:
        if self.single_mode_checkbox.isChecked():
            raw_path = self.input_c_edit.text().strip()
            if not raw_path:
                raise FileNotFoundError("Input C path is required.")
            path = Path(raw_path)
            if not path.exists():
                raise FileNotFoundError(f"Input C not found: {path}")
            return [path]

        base_dir = Path(self.batch_dir_edit.text().strip())
        if not base_dir.exists():
            raise FileNotFoundError(f"Batch directory not found: {base_dir}")
        pattern = self.batch_pattern_edit.text().strip() or "*"
        recursive = self.batch_recursive_checkbox.isChecked()
        if recursive:
            paths = sorted(base_dir.rglob(pattern))
        else:
            paths = sorted(base_dir.glob(pattern))
        extensions = [ext.lower() for ext in config.get("data", {}).get("extensions", [])]
        if extensions:
            paths = [path for path in paths if path.suffix.lower() in extensions]
        if not paths:
            raise ValueError("No input images found with the selected pattern.")
        return paths

    def _collect_gt_lookup(
        self, input_paths: List[Path]
    ) -> Optional[Dict[str, Dict[str, Path]]]:
        if self.single_mode_checkbox.isChecked():
            gt_a = self.gt_a_edit.text().strip()
            gt_b = self.gt_b_edit.text().strip()
            if not gt_a and not gt_b:
                return None
            if gt_a and not Path(gt_a).exists():
                raise FileNotFoundError(f"GT A not found: {gt_a}")
            if gt_b and not Path(gt_b).exists():
                raise FileNotFoundError(f"GT B not found: {gt_b}")
            sample_id = build_gt_lookup(input_paths, "C").keys()
            sample_id = list(sample_id)[0]
            lookup = {sample_id: {}}
            if gt_a:
                lookup[sample_id]["A"] = Path(gt_a)
            if gt_b:
                lookup[sample_id]["B"] = Path(gt_b)
            return lookup

        gt_dir_a = self.gt_dir_a_edit.text().strip()
        gt_dir_b = self.gt_dir_b_edit.text().strip()
        if not gt_dir_a and not gt_dir_b:
            return None
        if gt_dir_a and not Path(gt_dir_a).exists():
            raise FileNotFoundError(f"GT A directory not found: {gt_dir_a}")
        if gt_dir_b and not Path(gt_dir_b).exists():
            raise FileNotFoundError(f"GT B directory not found: {gt_dir_b}")
        pattern = self.gt_pattern_edit.text().strip() or "*"
        recursive = self.gt_recursive_checkbox.isChecked()
        input_map = build_gt_lookup(input_paths, "C")
        lookup = {sample_id: {} for sample_id in input_map.keys()}
        if gt_dir_a:
            base_a = Path(gt_dir_a)
            paths_a = (
                base_a.rglob(pattern) if recursive else base_a.glob(pattern)
            )
            map_a = build_gt_lookup(paths_a, "A")
            for sample_id, path in map_a.items():
                if sample_id in lookup:
                    lookup[sample_id]["A"] = path
        if gt_dir_b:
            base_b = Path(gt_dir_b)
            paths_b = (
                base_b.rglob(pattern) if recursive else base_b.glob(pattern)
            )
            map_b = build_gt_lookup(paths_b, "B")
            for sample_id, path in map_b.items():
                if sample_id in lookup:
                    lookup[sample_id]["B"] = path
        return lookup

    def _run_inference(self) -> None:
        try:
            base_config = load_config(Path(self.config_path_edit.text().strip()))
            gui_overrides = self._build_gui_overrides()
            config = merge_inference_config(base_config, gui_overrides=gui_overrides)
            input_paths = self._collect_input_paths(config)
            gt_lookup = self._collect_gt_lookup(input_paths)
        except Exception as exc:
            QMessageBox.critical(self, "Input error", str(exc))
            return

        output_dir = Path(config.get("output", {}).get("out_dir", "outputs/gui_infer"))
        output_dir.mkdir(parents=True, exist_ok=True)

        self._state = GuiInferenceState(config=config)
        self._state.input_map = build_gt_lookup(input_paths, "C")
        self._state.gt_lookup = gt_lookup

        self.progress_bar.setValue(0)
        self.progress_label.setText("Running...")
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)

        self._worker = InferenceWorker(
            config=config,
            input_paths=input_paths,
            gt_lookup=gt_lookup,
            model_manager=self._model_manager,
            output_dir=output_dir,
        )
        self._worker.progress.connect(self._update_progress)
        self._worker.finished.connect(self._handle_finished)
        self._worker.failed.connect(self._handle_failed)
        self._worker.start()

    def _cancel_inference(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self.progress_label.setText("Cancelling...")

    def _update_progress(self, processed: int, total: int, eta: float) -> None:
        if total > 0:
            self.progress_bar.setValue(int(processed / total * 100))
        self.progress_label.setText(f"Processed {processed}/{total} | ETA {eta:.1f}s")

    def _handle_finished(self, result: InferenceRunResult) -> None:
        self._state.result = result
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        status = "Cancelled" if result.cancelled else "Complete"
        self.progress_label.setText(f"{status}: {result.processed} processed")
        self._populate_metrics_table(result)
        self._load_sample_for_display()

    def _handle_failed(self, message: str) -> None:
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.progress_label.setText("Failed")
        QMessageBox.critical(self, "Inference failed", message)

    def _populate_metrics_table(self, result: InferenceRunResult) -> None:
        rows = [sample for sample in result.sample_results if sample.metrics]
        self.metrics_table.setRowCount(len(rows))
        for row_idx, sample in enumerate(rows):
            values = {
                "sample_id": sample.sample_id,
                "a_l1": sample.metrics.get("a_l1"),
                "a_l2": sample.metrics.get("a_l2"),
                "a_psnr": sample.metrics.get("a_psnr"),
                "a_ssim": sample.metrics.get("a_ssim"),
                "b_l1": sample.metrics.get("b_l1"),
                "b_l2": sample.metrics.get("b_l2"),
                "b_psnr": sample.metrics.get("b_psnr"),
                "b_ssim": sample.metrics.get("b_ssim"),
            }
            for col_idx, key in enumerate(values.keys()):
                value = values[key]
                item_text = "" if value is None else f"{value:.4f}"
                if key == "sample_id":
                    item_text = str(value)
                item = QTableWidgetItem(item_text)
                self.metrics_table.setItem(row_idx, col_idx, item)

    def _load_sample_for_display(self) -> None:
        if self._state.result is None or not self._state.result.sample_results:
            return
        self._display_sample(self._state.result.sample_results[0])

    def _display_sample(self, sample) -> None:
        self._current_sample = sample
        low = self.contrast_low_spin.value()
        high = self.contrast_high_spin.value()
        input_img = to_float01(read_image_16bit(sample.input_path))
        pred_a = to_float01(read_image_16bit(sample.output_a))
        pred_b = to_float01(read_image_16bit(sample.output_b))
        self.input_view.set_image(input_img, low, high)
        self.a_view.set_image(pred_a, low, high)
        self.b_view.set_image(pred_b, low, high)

        diag_text = self._build_metrics_text(sample)
        self.diagnostics_panel.update_metrics(diag_text)
        diff_map = self._build_diff_map(sample, pred_a, pred_b)
        self.diagnostics_panel.set_image(diff_map, low, high)

    def _build_metrics_text(self, sample) -> str:
        lines = [f"Sample: {sample.sample_id}"]
        if sample.metrics:
            for key, value in sample.metrics.items():
                lines.append(f"{key}: {value:.4f}")
        if self._state.result and self._state.result.sample_metrics_summary:
            lines.append("\nSummary:")
            for key, value in self._state.result.sample_metrics_summary.items():
                lines.append(f"{key}: {value:.4f}")
        return "\n".join(lines)

    def _build_diff_map(
        self, sample, pred_a: np.ndarray, pred_b: np.ndarray
    ) -> Optional[np.ndarray]:
        if self._state.gt_lookup is None:
            return None
        gt_entry = self._state.gt_lookup.get(sample.sample_id, {})
        diffs = []
        if "A" in gt_entry:
            gt_a = to_float01(read_image_16bit(gt_entry["A"]))
            diffs.append(np.abs(pred_a - gt_a))
        if "B" in gt_entry:
            gt_b = to_float01(read_image_16bit(gt_entry["B"]))
            diffs.append(np.abs(pred_b - gt_b))
        if not diffs:
            return None
        return np.mean(np.stack(diffs, axis=0), axis=0)

    def _on_metric_selection(self) -> None:
        items = self.metrics_table.selectedItems()
        if not items or self._state.result is None:
            return
        row = items[0].row()
        sample = [
            s
            for s in self._state.result.sample_results
            if s.metrics
        ][row]
        self._display_sample(sample)

    def _refresh_views(self) -> None:
        if self._current_sample is None:
            return
        self._display_sample(self._current_sample)

    def _append_log(self, message: str, level: int) -> None:
        level_name = logging.getLevelName(level)
        filter_level = self.log_level_combo.currentText()
        if logging._nameToLevel.get(filter_level, logging.INFO) > level:
            return
        self.log_text.appendPlainText(f"{level_name} {message}")

    def _copy_logs(self) -> None:
        QApplication.clipboard().setText(self.log_text.toPlainText())

    def _save_logs(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save logs", str(REPO_ROOT))
        if path:
            Path(path).write_text(self.log_text.toPlainText(), encoding="utf-8")

    def _clear_logs(self) -> None:
        self.log_text.clear()

    def update_pixel_readout(self, title: str, x: int, y: int, value: float) -> None:
        self.pixel_label.setText(f"{title}: ({x}, {y}) = {value:.4f}")


def _scale_for_display(image: np.ndarray, low: float, high: float) -> np.ndarray:
    low = float(np.clip(low, 0.0, 100.0))
    high = float(np.clip(high, 0.0, 100.0))
    if high <= low:
        high = low + 0.1
    vmin, vmax = np.percentile(image, [low, high])
    if vmax <= vmin:
        vmax = vmin + 1e-6
    scaled = (image - vmin) / (vmax - vmin)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def main() -> None:
    app = QApplication(sys.argv)
    window = InferenceGui()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
