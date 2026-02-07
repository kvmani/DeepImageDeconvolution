"""Tests for deterministic synthetic robustness benchmark."""
from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.deterministic_mixing_inversion.benchmark import run_synthetic_robustness_benchmark
from src.utils.io import write_image_16bit


def _make_pattern(size: int, y0: float, x0: float, sigma: float) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    dist = ((yy - y0) ** 2 + (xx - x0) ** 2) / max(sigma, 1e-6)
    return np.exp(-dist).astype(np.float32)


def test_synthetic_benchmark_writes_outputs(tmp_path: Path) -> None:
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir(parents=True, exist_ok=True)

    pattern_a = _make_pattern(size=64, y0=18.0, x0=22.0, sigma=180.0)
    pattern_b = _make_pattern(size=64, y0=46.0, x0=40.0, sigma=200.0)
    write_image_16bit(pool_dir / "case_bcc_1.png", np.clip(pattern_a, 0.0, 1.0))
    write_image_16bit(pool_dir / "case_fcc_1.png", np.clip(pattern_b, 0.0, 1.0))

    out_dir = tmp_path / "out"
    config = {
        "data": {
            "candidate_pool": {
                "root_dir": str(pool_dir),
                "a_pattern": "(?i)bcc",
                "b_pattern": "(?i)fcc",
                "recursive": False,
            }
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
        "synthetic_benchmark": {
            "use_true_pair_only": True,
            "max_candidate_pairs": 1,
            "max_samples": 6,
            "x_values": [0.2, 0.8],
            "save_synthetic_images": True,
            "save_synthetic_limit": 2,
            "nuisance": {
                "gain_levels": [1.0],
                "offset_levels": [0.0],
                "noise_std_levels": [0.0],
                "blur_sigma_levels": [0.0],
                "translation_x_levels": [0.0],
                "translation_y_levels": [0.0],
                "rotation_deg_levels": [0.0],
            },
        },
        "output": {"out_dir": str(out_dir)},
        "debug": {"enabled": False, "seed": 5},
    }

    summary = run_synthetic_robustness_benchmark(config=config, logger=logging.getLogger("det_benchmark_test"))
    assert summary["records"] == 2
    assert summary["pairs_used"] == 1

    results_path = out_dir / "benchmark_results.jsonl"
    assert results_path.exists()
    lines = [line for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2

    summary_csv = out_dir / "summary_metrics.csv"
    assert summary_csv.exists()
    csv_text = summary_csv.read_text(encoding="utf-8")
    assert "metric" in csv_text
    assert "mae" in csv_text

    report_path = out_dir / "report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "completed"
    assert report["summary"]["records"] == 2

