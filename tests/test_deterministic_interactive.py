"""Tests for deterministic interactive pair-identification helpers."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.deterministic_mixing_inversion.interactive import (
    SyntheticNoiseConfig,
    build_synthetic_case,
    identify_pair_from_candidates,
)
from src.deterministic_mixing_inversion.io import PatternRecord


def _build_pattern(size: int, center_y: float, center_x: float, spread: float) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    distance = ((yy - center_y) ** 2 + (xx - center_x) ** 2) / max(spread, 1e-6)
    pattern = np.exp(-distance)
    return np.clip(pattern, 0.0, 1.0).astype(np.float32)


def _candidate(path: Path, image: np.ndarray) -> PatternRecord:
    return PatternRecord(pattern_id=path.stem, path=path, image=image, source_dtype="uint16")


def test_identify_pair_two_stage_recovers_true_pair(tmp_path: Path) -> None:
    candidates = [
        _candidate(tmp_path / "cand_0.png", _build_pattern(64, 18.0, 18.0, 150.0)),
        _candidate(tmp_path / "cand_1.png", _build_pattern(64, 44.0, 18.0, 170.0)),
        _candidate(tmp_path / "cand_2.png", _build_pattern(64, 18.0, 42.0, 160.0)),
        _candidate(tmp_path / "cand_3.png", _build_pattern(64, 42.0, 42.0, 175.0)),
    ]
    true_a = 1
    true_b = 3
    true_x = 0.65

    case = build_synthetic_case(
        candidates=candidates,
        index_a=true_a,
        index_b=true_b,
        mix_fraction=true_x,
        noise=SyntheticNoiseConfig(
            gaussian_enabled=False,
            salt_pepper_enabled=False,
            rotation_enabled=False,
        ),
        seed=7,
        mask_enabled=False,
    )

    inversion_cfg = {
        "preprocess": {
            "auto_crop_to_target": True,
            "mask": {"enabled": False, "detect_existing": False},
            "background_correction": {"enabled": False, "mode": "subtractive", "sigma": 5.0},
            "standardize": {"enabled": False, "epsilon": 1.0e-6},
            "dog": {"enabled": False, "sigma_low": 1.0, "sigma_high": 3.0},
            "fft_filter": {"enabled": False, "low_cut": 0.0, "high_cut": 1.0, "rolloff": 0.02},
        },
        "metrics": {"enabled": ["ncc", "l2"], "primary": "ncc"},
        "search": {
            "strategy": "coarse_to_fine",
            "grid_steps": [0.1, 0.02, 0.005],
            "refine_window_steps": 3,
            "exhaustive_step": 0.005,
        },
        "alignment": {
            "enabled": False,
            "translation": {"enabled": False, "max_shift_px": 0},
            "rotation": {"enabled": False, "search_range_deg": 0, "hard_max_deg": 5, "step_deg": 1},
            "interpolation_order": 3,
        },
        "pair_search": {
            "two_stage_enabled": True,
            "coarse_top_m": 3,
            "coarse_metric": "ncc",
            "coarse_search": {
                "strategy": "coarse_to_fine",
                "grid_steps": [0.2, 0.05],
                "refine_window_steps": 2,
                "exhaustive_step": 0.05,
            },
            "coarse_alignment": {
                "enabled": False,
                "translation": {"enabled": False, "max_shift_px": 0},
                "rotation": {"enabled": False, "search_range_deg": 0, "hard_max_deg": 5, "step_deg": 1},
                "interpolation_order": 3,
            },
        },
    }

    result = identify_pair_from_candidates(
        candidates=candidates,
        mixed_image=case.mixed_c,
        inversion_cfg=inversion_cfg,
        top_k=3,
    )

    assert result.total_pairs == 6
    predicted_pair = {result.winner.index_a, result.winner.index_b}
    assert predicted_pair == {true_a, true_b}

    order_matches = result.winner.index_a == true_a and result.winner.index_b == true_b
    x_hat_true_order = result.winner.x_hat if order_matches else 1.0 - result.winner.x_hat
    assert x_hat_true_order == pytest.approx(true_x, abs=0.05)


def test_gui_module_import_has_no_qapplication_side_effects() -> None:
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    assert QApplication.instance() is None
    __import__("apps.deterministic_pair_gui.main")
    __import__("scripts.run_deterministic_pair_gui")
    assert QApplication.instance() is None
