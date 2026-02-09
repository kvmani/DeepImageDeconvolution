"""Tests for deterministic inversion runner."""
from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.deterministic_mixing_inversion.runner import run_deterministic_inversion
from src.utils.io import write_image_16bit


def _build_test_pattern(size: int, center_y: float, center_x: float, spread: float) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    distance = ((yy - center_y) ** 2 + (xx - center_x) ** 2) / max(spread, 1e-6)
    pattern = np.exp(-distance)
    return pattern.astype(np.float32)


def test_runner_recovers_known_mixing_fraction(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    mixed_dir = data_dir / "mixed"
    pool_dir = data_dir / "pool"
    mixed_dir.mkdir(parents=True, exist_ok=True)
    pool_dir.mkdir(parents=True, exist_ok=True)

    pattern_a = _build_test_pattern(size=64, center_y=20.0, center_x=24.0, spread=220.0)
    pattern_b = _build_test_pattern(size=64, center_y=44.0, center_x=40.0, spread=180.0)
    true_fraction = 0.72
    mixed_pattern = true_fraction * pattern_a + (1.0 - true_fraction) * pattern_b

    write_image_16bit(pool_dir / "sample_bcc_001.png", np.clip(pattern_a, 0.0, 1.0))
    write_image_16bit(pool_dir / "sample_fcc_001.png", np.clip(pattern_b, 0.0, 1.0))
    write_image_16bit(mixed_dir / "mixed_001.png", np.clip(mixed_pattern, 0.0, 1.0))

    output_dir = tmp_path / "out"
    config = {
        "data": {
            "mixed_dir": str(mixed_dir),
            "mixed_recursive": False,
            "candidate_pool": {
                "root_dir": str(pool_dir),
                "a_pattern": "(?i)bcc",
                "b_pattern": "(?i)fcc",
                "recursive": False,
            },
        },
        "deterministic_inversion": {
            "preprocess": {
                "auto_crop_to_target": False,
                "mask": {"enabled": False, "detect_existing": False},
                "background_correction": {"enabled": False, "mode": "subtractive"},
                "standardize": {"enabled": False},
                "dog": {"enabled": False},
            },
            "metrics": {"enabled": ["ncc", "l2", "l1"], "primary": "ncc"},
            "search": {
                "strategy": "exhaustive_grid",
                "exhaustive_step": 0.01,
                "grid_steps": [0.1, 0.02],
                "refine_window_steps": 4,
            },
            "alignment": {
                "enabled": False,
                "translation": {"enabled": False, "max_shift_px": 0},
                "rotation": {"enabled": False, "search_range_deg": 0, "hard_max_deg": 5, "step_deg": 1},
                "interpolation_order": 3,
            },
        },
        "output": {"out_dir": str(output_dir), "save_curves": True, "write_html_report": True},
        "debug": {"enabled": False},
    }

    logger = logging.getLogger("deterministic_test_runner")
    summary = run_deterministic_inversion(config, logger=logger)
    assert summary["processed"] == 1
    assert summary["candidate_pairs"] == 1

    results_path = output_dir / "results.jsonl"
    assert results_path.exists()
    result_line = results_path.read_text(encoding="utf-8").strip().splitlines()[0]
    result_payload = json.loads(result_line)
    recovered_fraction = float(result_payload["best_by_metric"]["ncc"]["x_hat"])
    assert abs(recovered_fraction - true_fraction) <= 0.03

    report_path = output_dir / "report.json"
    assert report_path.exists()
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_payload["status"] == "completed"
    assert report_payload["summary"]["processed"] == 1

    assert (output_dir / "summary_metrics.csv").exists()
    assert (output_dir / "reconstructions").exists()


def test_candidate_pool_synthetic_runs(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    pool_dir = data_dir / "pool"
    pool_dir.mkdir(parents=True, exist_ok=True)

    pattern_1 = _build_test_pattern(size=48, center_y=16.0, center_x=20.0, spread=160.0)
    pattern_2 = _build_test_pattern(size=48, center_y=32.0, center_x=28.0, spread=180.0)
    pattern_3 = _build_test_pattern(size=48, center_y=20.0, center_x=32.0, spread=140.0)
    pattern_4 = _build_test_pattern(size=48, center_y=30.0, center_x=14.0, spread=200.0)

    write_image_16bit(pool_dir / "cand_001.png", np.clip(pattern_1, 0.0, 1.0))
    write_image_16bit(pool_dir / "cand_002.png", np.clip(pattern_2, 0.0, 1.0))
    write_image_16bit(pool_dir / "cand_003.png", np.clip(pattern_3, 0.0, 1.0))
    write_image_16bit(pool_dir / "cand_004.png", np.clip(pattern_4, 0.0, 1.0))

    output_dir = tmp_path / "out_candidate"
    config = {
        "data": {
            "mode": "candidate_pool_synthetic",
            "candidate_pool": {
                "root_dir": str(pool_dir),
                "recursive": False,
                "max_candidates": 4,
                "sample_seed": 5,
                "synthetic_pairs": 2,
                "synthetic_seed": 5,
                "x_true": [0.3, 0.7],
                "noise": {"enabled": False},
            },
        },
        "deterministic_inversion": {
            "preprocess": {
                "auto_crop_to_target": False,
                "mask": {"enabled": False, "detect_existing": False},
                "background_correction": {"enabled": False, "mode": "subtractive"},
                "standardize": {"enabled": False},
                "dog": {"enabled": False},
                "fft_filter": {"enabled": False},
            },
            "metrics": {"enabled": ["ncc", "l2", "l1"], "primary": "ncc"},
            "search": {
                "strategy": "exhaustive_grid",
                "exhaustive_step": 0.05,
                "grid_steps": [0.1, 0.05],
                "refine_window_steps": 2,
            },
            "alignment": {
                "enabled": False,
                "translation": {"enabled": False, "max_shift_px": 0},
                "rotation": {"enabled": False, "search_range_deg": 0, "hard_max_deg": 5, "step_deg": 1},
                "interpolation_order": 3,
            },
        },
        "output": {"out_dir": str(output_dir), "save_curves": False, "write_html_report": False},
        "debug": {"enabled": False},
    }

    logger = logging.getLogger("deterministic_candidate_pool_test")
    summary = run_deterministic_inversion(config, logger=logger)
    assert summary["processed"] == 2
    assert summary["candidate_pairs"] == 6

    assert (output_dir / "results.jsonl").exists()
    assert (output_dir / "summary_metrics.csv").exists()
    assert (output_dir / "candidate_pool.csv").exists()
    assert (output_dir / "candidate_trials" / "trial_01").exists()
    assert (output_dir / "report.json").exists()
