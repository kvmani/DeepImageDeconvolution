"""Main window for deterministic pair-identification GUI."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import csv
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from apps.pattern_mixer_gui.gui.logging_handler import LogPanel
from apps.pattern_mixer_gui.gui.viewer import ImageViewer
from src.deterministic_mixing_inversion import (
    IdentificationResult,
    SyntheticCase,
    SyntheticNoiseConfig,
    build_synthetic_case,
    identify_pair_from_candidates,
    sample_random_candidates,
)
from src.deterministic_mixing_inversion.alignment import (
    RigidAlignment,
    apply_rigid_alignment,
    parse_alignment_settings,
)
from src.deterministic_mixing_inversion.io import PatternRecord
from src.deterministic_mixing_inversion.preprocess import (
    build_centered_mask,
    match_shape_to_target,
    parse_preprocess_settings,
)
from src.deterministic_mixing_inversion.reporting import update_progress_report
from src.utils.config import load_config
from src.utils.io import write_image_16bit
from src.utils.logging import collect_environment, get_git_commit, get_logger, write_manifest
from src.utils.run import resolve_run_dir


@dataclass
class GuiState:
    """Runtime GUI state."""

    candidates: List[PatternRecord]
    synthetic_case: Optional[SyntheticCase]
    identification: Optional[IdentificationResult]
    last_run_dir: Optional[Path]


@dataclass
class WorkerOutput:
    """Output payload from identification worker."""

    result: IdentificationResult
    winner_a_image: np.ndarray
    winner_b_image: np.ndarray
    winner_c_hat: np.ndarray
    winner_residual_abs: np.ndarray


def _build_winner_artifacts(
    candidates: List[PatternRecord],
    mixed_image: np.ndarray,
    result: IdentificationResult,
    inversion_cfg: Dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    preprocess_settings = parse_preprocess_settings(inversion_cfg.get("preprocess", {}))
    alignment_settings = parse_alignment_settings(inversion_cfg.get("alignment", {}))
    target_shape = mixed_image.shape
    mask = build_centered_mask(target_shape) if preprocess_settings.mask_enabled else None

    winner_a = candidates[result.winner.index_a]
    winner_b = candidates[result.winner.index_b]
    image_a = match_shape_to_target(
        winner_a.image,
        target_shape=target_shape,
        auto_crop_to_target=preprocess_settings.auto_crop_to_target,
    ).astype(np.float32)
    image_b = match_shape_to_target(
        winner_b.image,
        target_shape=target_shape,
        auto_crop_to_target=preprocess_settings.auto_crop_to_target,
    ).astype(np.float32)
    if mask is not None:
        image_a = image_a.copy()
        image_b = image_b.copy()
        image_a[~mask] = 0.0
        image_b[~mask] = 0.0

    c_hat = (result.winner.x_hat * image_a + (1.0 - result.winner.x_hat) * image_b).astype(np.float32)
    rigid = RigidAlignment(
        angle_deg=float(result.winner.alignment.get("angle_deg", 0.0)),
        shift_y=float(result.winner.alignment.get("shift_y", 0.0)),
        shift_x=float(result.winner.alignment.get("shift_x", 0.0)),
        score=0.0,
    )
    c_hat_aligned = apply_rigid_alignment(
        c_hat,
        rigid,
        interpolation_order=alignment_settings.interpolation_order,
        mask=mask,
    ).astype(np.float32)
    residual_abs = np.abs(c_hat_aligned - mixed_image).astype(np.float32)
    return image_a, image_b, np.clip(c_hat_aligned, 0.0, 1.0), np.clip(residual_abs, 0.0, 1.0)


class CandidateCard(QtWidgets.QFrame):
    """Thumbnail card for one candidate pattern."""

    def __init__(self, index: int, title: str, image: np.ndarray, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.index = index
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.setLineWidth(1)
        self.setMinimumWidth(120)
        self.setMaximumWidth(170)

        image_label = QtWidgets.QLabel()
        image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        image_label.setPixmap(self._to_pixmap(image))
        image_label.setMinimumHeight(96)

        title_label = QtWidgets.QLabel(title)
        title_label.setWordWrap(True)
        title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        title_label.setToolTip(title)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addWidget(image_label)
        layout.addWidget(title_label)

    @staticmethod
    def _to_pixmap(image: np.ndarray) -> QtGui.QPixmap:
        clipped = np.clip(image.astype(np.float32), 0.0, 1.0)
        uint8_image = (clipped * 255.0).astype(np.uint8)
        qimage = QtGui.QImage(
            uint8_image.data,
            uint8_image.shape[1],
            uint8_image.shape[0],
            uint8_image.strides[0],
            QtGui.QImage.Format.Format_Grayscale8,
        )
        pixmap = QtGui.QPixmap.fromImage(qimage.copy())
        return pixmap.scaled(128, 128, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)


class IdentificationWorker(QtCore.QThread):
    """Background worker for deterministic pair identification."""

    progress = QtCore.Signal(int, int, float, str)
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(
        self,
        candidates: List[PatternRecord],
        mixed_image: np.ndarray,
        inversion_cfg: Dict[str, Any],
        top_k: int,
    ) -> None:
        super().__init__()
        self._candidates = candidates
        self._mixed_image = mixed_image
        self._inversion_cfg = inversion_cfg
        self._top_k = top_k

    def run(self) -> None:
        try:
            result = identify_pair_from_candidates(
                candidates=self._candidates,
                mixed_image=self._mixed_image,
                inversion_cfg=self._inversion_cfg,
                top_k=self._top_k,
                progress_callback=self._on_progress,
                logger=logging.getLogger("deterministic_pair_gui.worker"),
            )
            winner_a, winner_b, winner_c_hat, winner_residual_abs = _build_winner_artifacts(
                candidates=self._candidates,
                mixed_image=self._mixed_image,
                result=result,
                inversion_cfg=self._inversion_cfg,
            )
            self.finished.emit(
                WorkerOutput(
                    result=result,
                    winner_a_image=winner_a,
                    winner_b_image=winner_b,
                    winner_c_hat=winner_c_hat,
                    winner_residual_abs=winner_residual_abs,
                )
            )
        except Exception as exc:
            self.failed.emit(str(exc))

    def _on_progress(self, processed: int, total: int, eta_s: float, message: str) -> None:
        self.progress.emit(processed, total, eta_s, message)


class DeterministicPairWindow(QtWidgets.QMainWindow):
    """Main GUI window for deterministic pair-identification demos."""

    def __init__(self, config_path: Path, debug: bool = False) -> None:
        super().__init__()
        self.setWindowTitle("Deterministic EBSD Pair Identification")
        self.resize(1800, 980)

        self._logger = get_logger(__name__)
        self._config_path = config_path
        self._debug = debug
        self._config = self._load_config(config_path)
        self._inversion_cfg_last: Dict[str, Any] = {}
        self._worker: Optional[IdentificationWorker] = None
        self._candidate_cards: List[CandidateCard] = []
        self._state = GuiState(candidates=[], synthetic_case=None, identification=None, last_run_dir=None)

        self.log_panel = LogPanel()
        self._build_viewers()
        self._build_layout()
        self._build_controls()
        self._apply_defaults()

    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        if not config_path.exists():
            self._logger.warning("Config path not found, using defaults: %s", config_path)
            return {}
        try:
            return load_config(config_path)
        except Exception as exc:
            self._logger.error("Failed to load config %s: %s", config_path, exc)
            return {}

    def _build_viewers(self) -> None:
        self._viewer_true_a = ImageViewer("Selected A (Noisy)")
        self._viewer_true_b = ImageViewer("Selected B (Noisy)")
        self._viewer_mixed = ImageViewer("Synthetic C")
        self._viewer_pred_a = ImageViewer("Predicted A")
        self._viewer_pred_b = ImageViewer("Predicted B")
        self._viewer_c_hat = ImageViewer("Predicted C_hat")
        self._viewer_residual = ImageViewer("Residual |C-C_hat|")

    def _build_layout(self) -> None:
        central = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        self._candidate_scroll = QtWidgets.QScrollArea()
        self._candidate_scroll.setWidgetResizable(True)
        self._candidate_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._candidate_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._candidate_scroll.setMinimumHeight(180)
        self._candidate_scroll.setMaximumHeight(220)
        self._candidate_container = QtWidgets.QWidget()
        self._candidate_layout = QtWidgets.QHBoxLayout(self._candidate_container)
        self._candidate_layout.setContentsMargins(6, 6, 6, 6)
        self._candidate_layout.setSpacing(6)
        self._candidate_layout.addStretch()
        self._candidate_scroll.setWidget(self._candidate_container)

        image_grid = QtWidgets.QGridLayout()
        image_grid.setContentsMargins(0, 0, 0, 0)
        image_grid.setSpacing(4)
        image_grid.addWidget(self._viewer_true_a, 0, 0)
        image_grid.addWidget(self._viewer_true_b, 0, 1)
        image_grid.addWidget(self._viewer_mixed, 0, 2)
        image_grid.addWidget(self._viewer_pred_a, 1, 0)
        image_grid.addWidget(self._viewer_pred_b, 1, 1)
        image_grid.addWidget(self._viewer_c_hat, 1, 2)
        image_grid.addWidget(self._viewer_residual, 0, 3, 2, 1)

        image_container = QtWidgets.QWidget()
        image_container.setLayout(image_grid)

        progress_row = QtWidgets.QHBoxLayout()
        self._progress_label = QtWidgets.QLabel("Idle")
        self._eta_label = QtWidgets.QLabel("ETA: -")
        self._progress_bar = QtWidgets.QProgressBar()
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        progress_row.addWidget(self._progress_label)
        progress_row.addWidget(self._progress_bar, stretch=1)
        progress_row.addWidget(self._eta_label)

        top_content = QtWidgets.QWidget()
        top_layout = QtWidgets.QVBoxLayout(top_content)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(4)
        top_layout.addWidget(self._candidate_scroll)
        top_layout.addWidget(image_container, stretch=1)
        top_layout.addLayout(progress_row)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        splitter.addWidget(top_content)
        splitter.addWidget(self.log_panel)
        splitter.setStretchFactor(0, 8)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([760, 200])

        outer.addWidget(splitter)
        self.setCentralWidget(central)

    def _build_controls(self) -> None:
        dock = QtWidgets.QDockWidget("Controls", self)
        dock.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea | QtCore.Qt.DockWidgetArea.RightDockWidgetArea
        )
        controls = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(controls)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        data_group = QtWidgets.QGroupBox("Candidate Pool")
        data_form = QtWidgets.QFormLayout(data_group)
        data_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)

        self._candidate_dir_edit = QtWidgets.QLineEdit()
        self._candidate_dir_edit.setToolTip("Folder containing candidate pure patterns (e.g., Good Pattern).")
        browse_candidates_btn = QtWidgets.QPushButton("Browse")
        browse_candidates_btn.clicked.connect(self._browse_candidate_dir)
        candidate_dir_row = QtWidgets.QHBoxLayout()
        candidate_dir_row.addWidget(self._candidate_dir_edit, stretch=1)
        candidate_dir_row.addWidget(browse_candidates_btn)

        self._candidate_count_spin = QtWidgets.QSpinBox()
        self._candidate_count_spin.setRange(2, 30)
        self._candidate_count_spin.setToolTip("Number of random candidate patterns to load.")

        self._fixed_seed_check = QtWidgets.QCheckBox("Use fixed seed")
        self._fixed_seed_check.setToolTip("Enable reproducible random candidate sampling/noise.")
        self._seed_spin = QtWidgets.QSpinBox()
        self._seed_spin.setRange(0, 999999)
        self._seed_spin.setValue(7)
        self._seed_spin.setToolTip("Seed used when fixed seed is enabled.")
        self._lock_candidates_check = QtWidgets.QCheckBox("Lock sampled candidates")
        self._lock_candidates_check.setChecked(True)
        self._lock_candidates_check.setToolTip("Keep current candidate set for repeated demo runs.")

        self._load_candidates_btn = QtWidgets.QPushButton("Load / Resample Candidates")
        self._load_candidates_btn.clicked.connect(lambda: self._load_candidates(force_resample=True))

        data_form.addRow("Candidate folder", candidate_dir_row)
        data_form.addRow("Candidate count", self._candidate_count_spin)
        data_form.addRow(self._fixed_seed_check, self._seed_spin)
        data_form.addRow(self._lock_candidates_check)
        data_form.addRow(self._load_candidates_btn)
        layout.addWidget(data_group)

        synth_group = QtWidgets.QGroupBox("Synthetic Case")
        synth_form = QtWidgets.QFormLayout(synth_group)
        synth_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self._index_a_combo = QtWidgets.QComboBox()
        self._index_b_combo = QtWidgets.QComboBox()
        self._index_a_combo.currentIndexChanged.connect(self._refresh_candidate_highlights)
        self._index_b_combo.currentIndexChanged.connect(self._refresh_candidate_highlights)
        self._x_spin = QtWidgets.QDoubleSpinBox()
        self._x_spin.setRange(0.0, 1.0)
        self._x_spin.setDecimals(4)
        self._x_spin.setSingleStep(0.01)
        self._x_spin.setValue(0.5)
        self._x_spin.setToolTip("Mixing fraction for selected A in C = x*A + (1-x)*B.")
        self._generate_btn = QtWidgets.QPushButton("Generate Synthetic C")
        self._generate_btn.clicked.connect(self._generate_synthetic_case)
        synth_form.addRow("A index", self._index_a_combo)
        synth_form.addRow("B index", self._index_b_combo)
        synth_form.addRow("x (A fraction)", self._x_spin)
        synth_form.addRow(self._generate_btn)
        layout.addWidget(synth_group)

        noise_group = QtWidgets.QGroupBox("Synthetic Noise")
        noise_form = QtWidgets.QFormLayout(noise_group)
        noise_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self._gaussian_check = QtWidgets.QCheckBox("Enable Gaussian")
        self._gaussian_sigma_spin = QtWidgets.QDoubleSpinBox()
        self._gaussian_sigma_spin.setRange(0.0, 0.2)
        self._gaussian_sigma_spin.setDecimals(4)
        self._gaussian_sigma_spin.setSingleStep(0.001)
        self._gaussian_sigma_spin.setValue(0.01)
        self._gaussian_sigma_spin.setToolTip("Standard deviation for additive Gaussian noise.")
        self._saltpepper_check = QtWidgets.QCheckBox("Enable salt-pepper")
        self._saltpepper_amount_spin = QtWidgets.QDoubleSpinBox()
        self._saltpepper_amount_spin.setRange(0.0, 0.2)
        self._saltpepper_amount_spin.setDecimals(4)
        self._saltpepper_amount_spin.setSingleStep(0.001)
        self._saltpepper_amount_spin.setValue(0.01)
        self._saltpepper_amount_spin.setToolTip("Pixel fraction replaced by salt/pepper.")
        self._salt_ratio_spin = QtWidgets.QDoubleSpinBox()
        self._salt_ratio_spin.setRange(0.0, 1.0)
        self._salt_ratio_spin.setDecimals(3)
        self._salt_ratio_spin.setSingleStep(0.05)
        self._salt_ratio_spin.setValue(0.5)
        self._salt_ratio_spin.setToolTip("Fraction of noisy pixels set to white (rest black).")
        self._rotation_check = QtWidgets.QCheckBox("Enable rotation")
        self._rotation_max_spin = QtWidgets.QDoubleSpinBox()
        self._rotation_max_spin.setRange(0.0, 2.0)
        self._rotation_max_spin.setDecimals(3)
        self._rotation_max_spin.setSingleStep(0.1)
        self._rotation_max_spin.setValue(2.0)
        self._rotation_max_spin.setToolTip("Random in-plane rotation sampled from [-max_deg, +max_deg].")
        noise_form.addRow(self._gaussian_check, self._gaussian_sigma_spin)
        noise_form.addRow(self._saltpepper_check, self._saltpepper_amount_spin)
        noise_form.addRow("Salt ratio", self._salt_ratio_spin)
        noise_form.addRow(self._rotation_check, self._rotation_max_spin)
        layout.addWidget(noise_group)

        id_group = QtWidgets.QGroupBox("Identification")
        id_form = QtWidgets.QFormLayout(id_group)
        id_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self._top_k_spin = QtWidgets.QSpinBox()
        self._top_k_spin.setRange(1, 20)
        self._top_k_spin.setValue(5)
        self._top_k_spin.setToolTip("Number of top-ranked candidate pairs shown.")
        self._coarse_top_m_spin = QtWidgets.QSpinBox()
        self._coarse_top_m_spin.setRange(2, 500)
        self._coarse_top_m_spin.setValue(20)
        self._coarse_top_m_spin.setToolTip("Two-stage search: refine top-M pairs after coarse screening.")
        self._align_translation_check = QtWidgets.QCheckBox("Enable translation alignment")
        self._align_translation_check.setChecked(True)
        self._align_rotation_check = QtWidgets.QCheckBox("Enable rotation alignment")
        self._align_rotation_check.setChecked(True)
        self._align_rot_deg_spin = QtWidgets.QDoubleSpinBox()
        self._align_rot_deg_spin.setRange(0.0, 2.0)
        self._align_rot_deg_spin.setDecimals(2)
        self._align_rot_deg_spin.setSingleStep(0.25)
        self._align_rot_deg_spin.setValue(2.0)
        self._align_rot_deg_spin.setToolTip("Alignment rotation search range in degrees.")
        self._identify_btn = QtWidgets.QPushButton("Identify Pair")
        self._identify_btn.clicked.connect(self._start_identification)
        self._run_demo_btn = QtWidgets.QPushButton("Run Full Demo")
        self._run_demo_btn.clicked.connect(self._run_full_demo)
        id_form.addRow("Top-K", self._top_k_spin)
        id_form.addRow("Stage-2 top-M", self._coarse_top_m_spin)
        id_form.addRow(self._align_translation_check)
        id_form.addRow(self._align_rotation_check, self._align_rot_deg_spin)
        id_form.addRow(self._identify_btn)
        id_form.addRow(self._run_demo_btn)
        layout.addWidget(id_group)

        output_group = QtWidgets.QGroupBox("Output")
        output_form = QtWidgets.QFormLayout(output_group)
        output_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self._output_dir_edit = QtWidgets.QLineEdit("outputs/gui_pair_demo")
        self._output_dir_edit.setToolTip("Base output directory for run artifacts.")
        browse_output_btn = QtWidgets.QPushButton("Browse")
        browse_output_btn.clicked.connect(self._browse_output_dir)
        output_row = QtWidgets.QHBoxLayout()
        output_row.addWidget(self._output_dir_edit, stretch=1)
        output_row.addWidget(browse_output_btn)
        self._run_tag_edit = QtWidgets.QLineEdit("gui_demo")
        self._run_tag_edit.setToolTip("Tag used in output folder suffix.")
        self._open_report_btn = QtWidgets.QPushButton("Open Last Report Folder")
        self._open_report_btn.clicked.connect(self._open_last_report_dir)
        output_form.addRow("Output dir", output_row)
        output_form.addRow("Run tag", self._run_tag_edit)
        output_form.addRow(self._open_report_btn)
        layout.addWidget(output_group)

        result_group = QtWidgets.QGroupBox("Result Summary")
        result_layout = QtWidgets.QVBoxLayout(result_group)
        self._result_banner = QtWidgets.QFrame()
        self._result_banner.setObjectName("resultBanner")
        self._result_banner.setMinimumHeight(110)
        banner_layout = QtWidgets.QHBoxLayout(self._result_banner)
        banner_layout.setContentsMargins(12, 12, 12, 12)
        banner_layout.setSpacing(12)
        self._result_icon = QtWidgets.QLabel()
        self._result_icon.setFixedSize(44, 44)
        self._result_icon.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._result_icon.setScaledContents(True)
        banner_text_layout = QtWidgets.QVBoxLayout()
        banner_text_layout.setContentsMargins(0, 0, 0, 0)
        banner_text_layout.setSpacing(4)
        self._result_title = QtWidgets.QLabel("Result: -")
        self._result_title.setWordWrap(True)
        self._result_title.setStyleSheet("font-weight:700; font-size:15px;")
        self._result_detail = QtWidgets.QLabel("Run identification to see a summary here.")
        self._result_detail.setWordWrap(True)
        self._result_detail.setStyleSheet("color:#444; font-size:13px;")
        banner_text_layout.addWidget(self._result_title)
        banner_text_layout.addWidget(self._result_detail)
        banner_layout.addWidget(self._result_icon)
        banner_layout.addLayout(banner_text_layout, stretch=1)

        self._winner_label = QtWidgets.QLabel("Winner: -")
        self._x_label = QtWidgets.QLabel("x_hat: -")
        self._score_label = QtWidgets.QLabel("NCC: - | L2: -")
        self._pair_match_label = QtWidgets.QLabel("Pair match: -")
        self._topk_table = QtWidgets.QTableWidget(0, 6)
        self._topk_table.setHorizontalHeaderLabels(["Rank", "A", "B", "x_hat", "NCC", "L2"])
        self._topk_table.horizontalHeader().setStretchLastSection(True)
        self._topk_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._topk_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self._topk_table.setMaximumHeight(200)
        result_layout.addWidget(self._result_banner)
        result_layout.addWidget(self._winner_label)
        result_layout.addWidget(self._x_label)
        result_layout.addWidget(self._score_label)
        result_layout.addWidget(self._pair_match_label)
        result_layout.addWidget(self._topk_table)
        layout.addWidget(result_group)

        layout.addStretch()
        dock.setWidget(controls)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self._set_result_banner(
            title="Result: -",
            detail="Run identification to see a summary here.",
            status="idle",
        )

    def _apply_defaults(self) -> None:
        data_cfg = self._config.get("data", {}) if isinstance(self._config, dict) else {}
        synth_cfg = self._config.get("synthetic_case", {}) if isinstance(self._config, dict) else {}
        inv_cfg = self._config.get("deterministic_inversion", {}) if isinstance(self._config, dict) else {}
        pair_cfg = inv_cfg.get("pair_search", {}) if isinstance(inv_cfg, dict) else {}
        align_cfg = inv_cfg.get("alignment", {}) if isinstance(inv_cfg, dict) else {}
        noise_cfg = synth_cfg.get("noise", {}) if isinstance(synth_cfg, dict) else {}
        output_cfg = self._config.get("output", {}) if isinstance(self._config, dict) else {}

        self._candidate_dir_edit.setText(str(data_cfg.get("candidate_dir", "data/raw/Double Pattern Data/Good Pattern")))
        self._candidate_count_spin.setValue(int(data_cfg.get("candidate_count", 10)))
        self._lock_candidates_check.setChecked(bool(data_cfg.get("lock_candidates", True)))
        seed_value = data_cfg.get("sample_seed", 7)
        if seed_value is None:
            self._fixed_seed_check.setChecked(False)
        else:
            self._fixed_seed_check.setChecked(True)
            self._seed_spin.setValue(int(seed_value))
        self._x_spin.setValue(float(synth_cfg.get("x_true", 0.5)))

        self._gaussian_check.setChecked(bool(noise_cfg.get("gaussian_enabled", True)))
        self._gaussian_sigma_spin.setValue(float(noise_cfg.get("gaussian_std", 0.01)))
        self._saltpepper_check.setChecked(bool(noise_cfg.get("salt_pepper_enabled", False)))
        self._saltpepper_amount_spin.setValue(float(noise_cfg.get("salt_pepper_amount", 0.01)))
        self._salt_ratio_spin.setValue(float(noise_cfg.get("salt_vs_pepper", 0.5)))
        self._rotation_check.setChecked(bool(noise_cfg.get("rotation_enabled", True)))
        self._rotation_max_spin.setValue(float(noise_cfg.get("rotation_max_deg", 2.0)))

        self._top_k_spin.setValue(5)
        self._coarse_top_m_spin.setValue(int(pair_cfg.get("coarse_top_m", 20)))
        translation_cfg = align_cfg.get("translation", {}) if isinstance(align_cfg, dict) else {}
        rotation_cfg = align_cfg.get("rotation", {}) if isinstance(align_cfg, dict) else {}
        self._align_translation_check.setChecked(bool(translation_cfg.get("enabled", True)))
        self._align_rotation_check.setChecked(bool(rotation_cfg.get("enabled", True)))
        self._align_rot_deg_spin.setValue(float(rotation_cfg.get("search_range_deg", 2.0)))

        self._output_dir_edit.setText(str(output_cfg.get("out_dir", "outputs/gui_pair_demo")))
        self._run_tag_edit.setText("gui_demo")

        candidate_root = Path(self._candidate_dir_edit.text()).expanduser()
        if candidate_root.exists():
            self._load_candidates(force_resample=True)
        elif self._debug:
            self._logger.warning("Candidate directory does not exist: %s", candidate_root)

    def _browse_candidate_dir(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select candidate folder", self._candidate_dir_edit.text())
        if path:
            self._candidate_dir_edit.setText(path)

    def _browse_output_dir(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output folder", self._output_dir_edit.text())
        if path:
            self._output_dir_edit.setText(path)

    def _open_last_report_dir(self) -> None:
        if self._state.last_run_dir is None:
            QtWidgets.QMessageBox.information(self, "No run", "No completed run available yet.")
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self._state.last_run_dir.resolve())))

    def _current_seed(self) -> Optional[int]:
        return int(self._seed_spin.value()) if self._fixed_seed_check.isChecked() else None

    def _load_candidates(self, force_resample: bool = False) -> None:
        if self._lock_candidates_check.isChecked() and self._state.candidates and not force_resample:
            self._logger.info("Using locked candidate set (%d patterns).", len(self._state.candidates))
            return
        try:
            candidate_dir = Path(self._candidate_dir_edit.text()).expanduser()
            count = int(self._candidate_count_spin.value())
            candidates = sample_random_candidates(
                candidate_dir=candidate_dir,
                sample_count=count,
                seed=self._current_seed(),
                recursive=False,
                logger=self._logger,
            )
        except Exception as exc:
            self._show_error("Candidate loading failed", str(exc))
            return

        self._state.candidates = candidates
        self._state.synthetic_case = None
        self._state.identification = None
        self._render_candidate_cards()
        self._populate_candidate_selectors()
        self._clear_identification_views()
        self._logger.info("Loaded %d candidate patterns.", len(candidates))

    def _render_candidate_cards(self) -> None:
        while self._candidate_layout.count():
            item = self._candidate_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._candidate_cards = []
        for idx, record in enumerate(self._state.candidates):
            title = f"{idx}: {record.path.stem}"
            card = CandidateCard(index=idx, title=title, image=record.image)
            self._candidate_layout.addWidget(card)
            self._candidate_cards.append(card)
        self._candidate_layout.addStretch()
        self._refresh_candidate_highlights()

    def _populate_candidate_selectors(self) -> None:
        prev_a = self._index_a_combo.currentData()
        prev_b = self._index_b_combo.currentData()
        self._index_a_combo.blockSignals(True)
        self._index_b_combo.blockSignals(True)
        self._index_a_combo.clear()
        self._index_b_combo.clear()
        for idx, record in enumerate(self._state.candidates):
            label = f"{idx}: {record.path.stem}"
            self._index_a_combo.addItem(label, idx)
            self._index_b_combo.addItem(label, idx)
        if self._state.candidates:
            self._index_a_combo.setCurrentIndex(0)
            self._index_b_combo.setCurrentIndex(1 if len(self._state.candidates) > 1 else 0)
        if prev_a is not None:
            pos = self._index_a_combo.findData(prev_a)
            if pos >= 0:
                self._index_a_combo.setCurrentIndex(pos)
        if prev_b is not None:
            pos = self._index_b_combo.findData(prev_b)
            if pos >= 0:
                self._index_b_combo.setCurrentIndex(pos)
        self._index_a_combo.blockSignals(False)
        self._index_b_combo.blockSignals(False)
        self._refresh_candidate_highlights()

    def _refresh_candidate_highlights(self) -> None:
        selected_a = self._index_a_combo.currentData()
        selected_b = self._index_b_combo.currentData()
        winner_indices = set()
        if self._state.identification is not None:
            winner_indices = {self._state.identification.winner.index_a, self._state.identification.winner.index_b}
        for card in self._candidate_cards:
            border = "#b5b5b5"
            bg = "#ffffff"
            if card.index in winner_indices:
                border = "#d62728"
                bg = "#fff2f2"
            elif card.index == selected_a:
                border = "#1f77b4"
                bg = "#f0f7ff"
            elif card.index == selected_b:
                border = "#2ca02c"
                bg = "#f1fff2"
            card.setStyleSheet(
                f"QFrame{{border:2px solid {border}; border-radius:6px; background:{bg};}} QLabel{{background:transparent;}}"
            )

    def _noise_config_from_ui(self) -> SyntheticNoiseConfig:
        return SyntheticNoiseConfig(
            gaussian_enabled=self._gaussian_check.isChecked(),
            gaussian_std=float(max(self._gaussian_sigma_spin.value(), 0.0)),
            salt_pepper_enabled=self._saltpepper_check.isChecked(),
            salt_pepper_amount=float(max(self._saltpepper_amount_spin.value(), 0.0)),
            salt_vs_pepper=float(np.clip(self._salt_ratio_spin.value(), 0.0, 1.0)),
            rotation_enabled=self._rotation_check.isChecked(),
            rotation_max_deg=float(np.clip(self._rotation_max_spin.value(), 0.0, 2.0)),
        )

    def _generate_synthetic_case(self) -> bool:
        if not self._state.candidates:
            self._load_candidates(force_resample=False)
        if not self._state.candidates:
            return False
        idx_a = self._index_a_combo.currentData()
        idx_b = self._index_b_combo.currentData()
        if idx_a is None or idx_b is None:
            self._show_error("Invalid selection", "Select both A and B candidates.")
            return False
        if int(idx_a) == int(idx_b):
            self._show_error("Invalid selection", "A and B must be different candidates.")
            return False
        try:
            case = build_synthetic_case(
                candidates=self._state.candidates,
                index_a=int(idx_a),
                index_b=int(idx_b),
                mix_fraction=float(np.clip(self._x_spin.value(), 0.0, 1.0)),
                noise=self._noise_config_from_ui(),
                seed=self._current_seed(),
                mask_enabled=True,
            )
        except Exception as exc:
            self._show_error("Synthetic generation failed", str(exc))
            return False

        self._state.synthetic_case = case
        self._state.identification = None
        self._viewer_true_a.set_image(case.pattern_a_noisy)
        self._viewer_true_b.set_image(case.pattern_b_noisy)
        self._viewer_mixed.set_image(case.mixed_c)
        self._clear_identification_views()
        self._winner_label.setText("Winner: -")
        self._x_label.setText("x_hat: -")
        self._score_label.setText("NCC: - | L2: -")
        self._pair_match_label.setText("Pair match: -")
        self._set_result_banner(
            title="Result: -",
            detail="Synthetic case ready. Run identification to see results.",
            status="idle",
        )
        self._logger.info(
            "Synthetic case generated with A=%s B=%s x=%.4f (rotA=%.3f°, rotB=%.3f°).",
            case.candidate_a.pattern_id,
            case.candidate_b.pattern_id,
            case.mix_fraction_true,
            case.angle_a_deg,
            case.angle_b_deg,
        )
        self._refresh_candidate_highlights()
        return True

    def _inversion_config_from_ui(self) -> Dict[str, Any]:
        config = deepcopy(self._config.get("deterministic_inversion", {}))
        if not isinstance(config, dict):
            config = {}
        config.setdefault("metrics", {})
        if isinstance(config["metrics"], dict):
            config["metrics"]["enabled"] = ["ncc", "l2"]
            config["metrics"]["primary"] = "ncc"

        config.setdefault("pair_search", {})
        if isinstance(config["pair_search"], dict):
            config["pair_search"]["two_stage_enabled"] = True
            config["pair_search"]["coarse_top_m"] = int(self._coarse_top_m_spin.value())
            config["pair_search"]["coarse_metric"] = "ncc"

        config.setdefault("alignment", {})
        if isinstance(config["alignment"], dict):
            config["alignment"]["enabled"] = bool(
                self._align_translation_check.isChecked() or self._align_rotation_check.isChecked()
            )
            config["alignment"].setdefault("translation", {})
            config["alignment"].setdefault("rotation", {})
            if isinstance(config["alignment"]["translation"], dict):
                config["alignment"]["translation"]["enabled"] = self._align_translation_check.isChecked()
            if isinstance(config["alignment"]["rotation"], dict):
                config["alignment"]["rotation"]["enabled"] = self._align_rotation_check.isChecked()
                config["alignment"]["rotation"]["search_range_deg"] = float(self._align_rot_deg_spin.value())
                config["alignment"]["rotation"]["hard_max_deg"] = 5.0
        return config

    def _start_identification(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._show_error("Busy", "Identification is already running.")
            return
        if self._state.synthetic_case is None:
            if not self._generate_synthetic_case():
                return
        if self._state.synthetic_case is None:
            return

        self._inversion_cfg_last = self._inversion_config_from_ui()
        self._worker = IdentificationWorker(
            candidates=list(self._state.candidates),
            mixed_image=self._state.synthetic_case.mixed_c,
            inversion_cfg=self._inversion_cfg_last,
            top_k=int(self._top_k_spin.value()),
        )
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.failed.connect(self._on_worker_failed)

        self._set_controls_enabled(False)
        self._set_result_banner(
            title="Identification running...",
            detail="Searching candidate pairs. Progress appears below.",
            status="running",
        )
        self._progress_label.setText("Running identification...")
        self._eta_label.setText("ETA: -")
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        self._worker.start()
        self._logger.info("Started deterministic identification with top_k=%d.", self._top_k_spin.value())

    def _run_full_demo(self) -> None:
        if self._generate_synthetic_case():
            self._start_identification()

    @QtCore.Slot(int, int, float, str)
    def _on_worker_progress(self, processed: int, total: int, eta_s: float, message: str) -> None:
        self._progress_bar.setRange(0, max(total, 1))
        self._progress_bar.setValue(min(processed, total))
        self._progress_label.setText(message)
        self._eta_label.setText(f"ETA: {eta_s:.1f}s")
        self._logger.info("%s | ETA %.2fs", message, eta_s)

    @QtCore.Slot(object)
    def _on_worker_finished(self, payload: WorkerOutput) -> None:
        self._set_controls_enabled(True)
        self._worker = None
        self._state.identification = payload.result

        self._viewer_pred_a.set_image(payload.winner_a_image)
        self._viewer_pred_b.set_image(payload.winner_b_image)
        self._viewer_c_hat.set_image(payload.winner_c_hat)
        self._viewer_residual.set_image(payload.winner_residual_abs)
        self._refresh_candidate_highlights()
        self._update_topk_table(payload.result)

        pair_match = False
        x_hat_true_order = payload.result.winner.x_hat
        x_abs_error = None
        if self._state.synthetic_case is not None:
            true_a = self._index_a_combo.currentData()
            true_b = self._index_b_combo.currentData()
            if true_a is not None and true_b is not None:
                true_pair = tuple(sorted((int(true_a), int(true_b))))
                pred_pair = tuple(sorted((payload.result.winner.index_a, payload.result.winner.index_b)))
                pair_match = bool(true_pair == pred_pair)
                same_order = bool(
                    payload.result.winner.index_a == int(true_a) and payload.result.winner.index_b == int(true_b)
                )
                x_hat_true_order = float(payload.result.winner.x_hat if same_order else 1.0 - payload.result.winner.x_hat)
                x_abs_error = abs(x_hat_true_order - float(self._state.synthetic_case.mix_fraction_true))

        self._winner_label.setText(f"Winner: {payload.result.winner.id_a} + {payload.result.winner.id_b}")
        if x_abs_error is None:
            self._x_label.setText(f"x_hat: {payload.result.winner.x_hat:.4f}")
        else:
            self._x_label.setText(
                f"x_hat: {payload.result.winner.x_hat:.4f} | x_hat(true-order): {x_hat_true_order:.4f} | |err|={x_abs_error:.4f}"
            )
        self._score_label.setText(
            f"NCC: {payload.result.winner.primary_score:.6f} | L2: {payload.result.winner.l2_score if payload.result.winner.l2_score is not None else '-'}"
        )
        self._pair_match_label.setText(f"Pair match: {pair_match}")
        self._progress_label.setText("Completed")
        self._eta_label.setText("ETA: 0.0s")
        self._progress_bar.setValue(self._progress_bar.maximum())

        if self._state.synthetic_case is not None:
            x_true = float(self._state.synthetic_case.mix_fraction_true)
            if pair_match:
                title = "Pair correctly identified"
                if x_abs_error is None:
                    detail = (
                        f"Predicted: {payload.result.winner.id_a} + {payload.result.winner.id_b}. "
                        f"x_hat={x_hat_true_order:.4f} vs x_true={x_true:.4f}."
                    )
                else:
                    detail = (
                        f"Predicted: {payload.result.winner.id_a} + {payload.result.winner.id_b}. "
                        f"x_hat={x_hat_true_order:.4f} vs x_true={x_true:.4f} (|err|={x_abs_error:.4f})."
                    )
                self._set_result_banner(title=title, detail=detail, status="success")
            else:
                title = "Pair mismatch"
                detail = (
                    f"Predicted: {payload.result.winner.id_a} + {payload.result.winner.id_b}. "
                    f"True: {self._state.synthetic_case.candidate_a.pattern_id} + "
                    f"{self._state.synthetic_case.candidate_b.pattern_id}."
                )
                if x_abs_error is not None:
                    detail = f"{detail} x_hat={payload.result.winner.x_hat:.4f} vs x_true={x_true:.4f}."
                self._set_result_banner(title=title, detail=detail, status="mismatch")
        else:
            self._set_result_banner(
                title="Identification complete",
                detail=(
                    f"Winner: {payload.result.winner.id_a} + {payload.result.winner.id_b} | "
                    f"x_hat={payload.result.winner.x_hat:.4f}."
                ),
                status="success",
            )

        run_dir = self._persist_run_outputs(
            payload=payload,
            pair_match=pair_match,
            x_hat_true_order=x_hat_true_order,
            x_abs_error=x_abs_error,
        )
        self._state.last_run_dir = run_dir
        self._logger.info(
            "Identification complete. Winner=%s + %s x_hat=%.4f run_dir=%s",
            payload.result.winner.id_a,
            payload.result.winner.id_b,
            payload.result.winner.x_hat,
            run_dir,
        )

    @QtCore.Slot(str)
    def _on_worker_failed(self, message: str) -> None:
        self._set_controls_enabled(True)
        self._worker = None
        self._progress_label.setText("Failed")
        self._eta_label.setText("ETA: -")
        self._set_result_banner(title="Identification failed", detail=message, status="error")
        self._show_error("Identification failed", message)
        self._logger.error("Identification failed: %s", message)

    def _update_topk_table(self, result: IdentificationResult) -> None:
        self._topk_table.setRowCount(len(result.top_k))
        for row_idx, row in enumerate(result.top_k):
            values = [
                str(row.rank),
                row.id_a,
                row.id_b,
                f"{row.x_hat:.4f}",
                f"{row.primary_score:.6f}",
                "-" if row.l2_score is None else f"{row.l2_score:.6f}",
            ]
            for col_idx, value in enumerate(values):
                self._topk_table.setItem(row_idx, col_idx, QtWidgets.QTableWidgetItem(value))
        self._topk_table.resizeColumnsToContents()

    def _persist_run_outputs(
        self,
        payload: WorkerOutput,
        pair_match: bool,
        x_hat_true_order: float,
        x_abs_error: Optional[float],
    ) -> Path:
        base_out = Path(self._output_dir_edit.text()).expanduser()
        run_tag = self._run_tag_edit.text().strip() or "gui_demo"
        run_dir = resolve_run_dir(base_out, run_tag)
        run_dir.mkdir(parents=True, exist_ok=True)

        case = self._state.synthetic_case
        if case is None:
            raise RuntimeError("Synthetic case is missing while persisting outputs.")

        synth_dir = run_dir / "synthetic"
        recon_dir = run_dir / "reconstructions"
        report_dir = run_dir / "report"
        synth_dir.mkdir(parents=True, exist_ok=True)
        recon_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)

        write_image_16bit(synth_dir / "a_noisy.png", case.pattern_a_noisy)
        write_image_16bit(synth_dir / "b_noisy.png", case.pattern_b_noisy)
        write_image_16bit(synth_dir / "c_synthetic.png", case.mixed_c)
        write_image_16bit(recon_dir / "winner_a.png", payload.winner_a_image)
        write_image_16bit(recon_dir / "winner_b.png", payload.winner_b_image)
        write_image_16bit(recon_dir / "winner_c_hat.png", payload.winner_c_hat)
        write_image_16bit(recon_dir / "winner_residual_abs.png", payload.winner_residual_abs)

        candidate_manifest = {
            "created_at": datetime.now().isoformat(),
            "candidate_paths": [str(record.path) for record in self._state.candidates],
            "candidate_ids": [record.pattern_id for record in self._state.candidates],
        }
        (run_dir / "candidate_manifest.json").write_text(json.dumps(candidate_manifest, indent=2), encoding="utf-8")

        result_payload = {
            "created_at": datetime.now().isoformat(),
            "true_pair": {
                "index_a": int(self._index_a_combo.currentData()),
                "index_b": int(self._index_b_combo.currentData()),
                "id_a": case.candidate_a.pattern_id,
                "id_b": case.candidate_b.pattern_id,
                "x_true": case.mix_fraction_true,
            },
            "noise": {
                "gaussian_enabled": self._gaussian_check.isChecked(),
                "gaussian_std": self._gaussian_sigma_spin.value(),
                "salt_pepper_enabled": self._saltpepper_check.isChecked(),
                "salt_pepper_amount": self._saltpepper_amount_spin.value(),
                "salt_vs_pepper": self._salt_ratio_spin.value(),
                "rotation_enabled": self._rotation_check.isChecked(),
                "rotation_max_deg": self._rotation_max_spin.value(),
                "angle_a_deg": case.angle_a_deg,
                "angle_b_deg": case.angle_b_deg,
            },
            "pair_match": pair_match,
            "x_hat_in_true_order": x_hat_true_order,
            "x_abs_error": x_abs_error,
            "winner": {
                "rank": payload.result.winner.rank,
                "index_a": payload.result.winner.index_a,
                "index_b": payload.result.winner.index_b,
                "id_a": payload.result.winner.id_a,
                "id_b": payload.result.winner.id_b,
                "x_hat": payload.result.winner.x_hat,
                "primary_score": payload.result.winner.primary_score,
                "l2_score": payload.result.winner.l2_score,
                "metric_scores": payload.result.winner.metric_scores,
                "alignment": payload.result.winner.alignment,
                "top_margin": payload.result.winner.top_margin,
            },
            "top_k": [
                {
                    "rank": row.rank,
                    "index_a": row.index_a,
                    "index_b": row.index_b,
                    "id_a": row.id_a,
                    "id_b": row.id_b,
                    "x_hat": row.x_hat,
                    "primary_score": row.primary_score,
                    "l2_score": row.l2_score,
                    "metric_scores": row.metric_scores,
                    "alignment": row.alignment,
                    "top_margin": row.top_margin,
                }
                for row in payload.result.top_k
            ],
            "runtime_s": payload.result.runtime_s,
            "total_pairs": payload.result.total_pairs,
            "primary_metric": payload.result.primary_metric,
        }
        (run_dir / "demo_result.json").write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
        self._write_topk_csv(run_dir / "top_k_pairs.csv", payload.result)
        self._write_html_report(run_dir / "report" / "index.html", result_payload)

        summary = {
            "true_a": case.candidate_a.pattern_id,
            "true_b": case.candidate_b.pattern_id,
            "pred_a": payload.result.winner.id_a,
            "pred_b": payload.result.winner.id_b,
            "pair_match": pair_match,
            "x_true": case.mix_fraction_true,
            "x_hat": payload.result.winner.x_hat,
            "x_hat_in_true_order": x_hat_true_order,
            "x_abs_error": x_abs_error,
            "primary_metric": payload.result.primary_metric,
            "primary_score": payload.result.winner.primary_score,
            "runtime_s": payload.result.runtime_s,
            "total_pairs": payload.result.total_pairs,
        }
        report_payload = {
            "status": "completed",
            "progress": {"current": 1, "total": 1, "percent": 100.0, "eta_s": 0.0},
            "summary": summary,
        }
        update_progress_report(run_dir, report_payload)
        manifest = {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "git_commit": get_git_commit(Path.cwd()),
            "environment": collect_environment(),
            "config_path": str(self._config_path),
            "inversion_cfg": self._inversion_cfg_last,
            "output_dir": str(run_dir),
            "summary": summary,
            "failures": [],
        }
        write_manifest(run_dir, manifest)
        return run_dir

    def _write_topk_csv(self, path: Path, result: IdentificationResult) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "rank",
                    "index_a",
                    "index_b",
                    "id_a",
                    "id_b",
                    "x_hat",
                    "primary_score",
                    "l2_score",
                    "top_margin",
                    "angle_deg",
                    "shift_y",
                    "shift_x",
                ],
            )
            writer.writeheader()
            for row in result.top_k:
                writer.writerow(
                    {
                        "rank": row.rank,
                        "index_a": row.index_a,
                        "index_b": row.index_b,
                        "id_a": row.id_a,
                        "id_b": row.id_b,
                        "x_hat": row.x_hat,
                        "primary_score": row.primary_score,
                        "l2_score": row.l2_score,
                        "top_margin": row.top_margin,
                        "angle_deg": row.alignment.get("angle_deg"),
                        "shift_y": row.alignment.get("shift_y"),
                        "shift_x": row.alignment.get("shift_x"),
                    }
                )

    def _write_html_report(self, path: Path, result_payload: Dict[str, Any]) -> None:
        winner = result_payload.get("winner", {})
        html = [
            "<!doctype html>",
            "<html><head><meta charset='utf-8'><title>Deterministic GUI Pair Demo</title>",
            "<style>",
            "body{font-family:Arial,sans-serif;margin:20px;}",
            "table{border-collapse:collapse;width:100%;margin-top:16px;}",
            "th,td{border:1px solid #ddd;padding:6px;text-align:left;}",
            "th{background:#f5f5f5;}",
            ".row{display:flex;gap:12px;flex-wrap:wrap;}",
            ".card{border:1px solid #ddd;border-radius:8px;padding:8px;}",
            ".img{width:220px;height:auto;border:1px solid #eee;}",
            "</style></head><body>",
            "<h1>Deterministic GUI Pair Demo</h1>",
            (
                f"<p>Winner: {winner.get('id_a')} + {winner.get('id_b')} | "
                f"x_hat={winner.get('x_hat')} | "
                f"{result_payload.get('primary_metric')}={winner.get('primary_score')}</p>"
            ),
            "<div class='row'>",
            "<div class='card'><h3>Synthetic A noisy</h3><img class='img' src='../synthetic/a_noisy.png'/></div>",
            "<div class='card'><h3>Synthetic B noisy</h3><img class='img' src='../synthetic/b_noisy.png'/></div>",
            "<div class='card'><h3>Synthetic C</h3><img class='img' src='../synthetic/c_synthetic.png'/></div>",
            "<div class='card'><h3>Winner C_hat</h3><img class='img' src='../reconstructions/winner_c_hat.png'/></div>",
            "<div class='card'><h3>|Residual|</h3><img class='img' src='../reconstructions/winner_residual_abs.png'/></div>",
            "</div>",
            "<table><tr><th>Rank</th><th>A</th><th>B</th><th>x_hat</th><th>NCC</th><th>L2</th></tr>",
        ]
        for row in result_payload.get("top_k", []):
            html.append(
                "<tr>"
                f"<td>{row.get('rank')}</td>"
                f"<td>{row.get('id_a')}</td>"
                f"<td>{row.get('id_b')}</td>"
                f"<td>{row.get('x_hat')}</td>"
                f"<td>{row.get('primary_score')}</td>"
                f"<td>{row.get('l2_score')}</td>"
                "</tr>"
            )
        html.extend(["</table>", "</body></html>"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(html), encoding="utf-8")

    def _clear_identification_views(self) -> None:
        self._viewer_pred_a.set_image(None)
        self._viewer_pred_b.set_image(None)
        self._viewer_c_hat.set_image(None)
        self._viewer_residual.set_image(None)
        self._topk_table.setRowCount(0)
        self._state.identification = None
        self._set_result_banner(
            title="Result: -",
            detail="Run identification to see a summary here.",
            status="idle",
        )
        self._refresh_candidate_highlights()

    def _set_controls_enabled(self, enabled: bool) -> None:
        self._load_candidates_btn.setEnabled(enabled)
        self._generate_btn.setEnabled(enabled)
        self._identify_btn.setEnabled(enabled)
        self._run_demo_btn.setEnabled(enabled)

    def _make_status_pixmap(self, color: QtGui.QColor, mark: Optional[bool]) -> QtGui.QPixmap:
        size = 44
        pixmap = QtGui.QPixmap(size, size)
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(QtGui.QBrush(color))
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawEllipse(0, 0, size - 1, size - 1)

        if mark is not None:
            pen = QtGui.QPen(QtGui.QColor("white"))
            pen.setWidth(3)
            pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            if mark:
                start = QtCore.QPointF(size * 0.26, size * 0.55)
                mid = QtCore.QPointF(size * 0.45, size * 0.72)
                end = QtCore.QPointF(size * 0.76, size * 0.30)
                painter.drawLine(start, mid)
                painter.drawLine(mid, end)
            else:
                p1 = QtCore.QPointF(size * 0.30, size * 0.30)
                p2 = QtCore.QPointF(size * 0.70, size * 0.70)
                p3 = QtCore.QPointF(size * 0.70, size * 0.30)
                p4 = QtCore.QPointF(size * 0.30, size * 0.70)
                painter.drawLine(p1, p2)
                painter.drawLine(p3, p4)
        painter.end()
        return pixmap

    def _set_result_banner(self, title: str, detail: str, status: str) -> None:
        palette = {
            "idle": {"bg": "#f5f5f5", "border": "#c7c7c7", "dot": "#9e9e9e", "mark": None},
            "running": {"bg": "#eef4ff", "border": "#b5c9ef", "dot": "#1f77b4", "mark": None},
            "success": {"bg": "#edf7ee", "border": "#b6e2bf", "dot": "#2ca02c", "mark": True},
            "mismatch": {"bg": "#fdecea", "border": "#f3c2bf", "dot": "#d62728", "mark": False},
            "error": {"bg": "#fdecea", "border": "#f3c2bf", "dot": "#d62728", "mark": False},
        }
        style = palette.get(status, palette["idle"])
        self._result_banner.setStyleSheet(
            "QFrame#resultBanner{"
            f"border:1px solid {style['border']};"
            f"border-radius:6px;"
            f"background:{style['bg']};"
            "}"
        )
        self._result_icon.setPixmap(
            self._make_status_pixmap(QtGui.QColor(style["dot"]), style["mark"])
        )
        self._result_title.setText(title)
        self._result_detail.setText(detail)

    def _show_error(self, title: str, message: str) -> None:
        QtWidgets.QMessageBox.critical(self, title, message)
