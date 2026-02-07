"""Tests for deterministic mixing-fraction search."""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.deterministic_mixing_inversion.metrics import parse_metric_config
from src.deterministic_mixing_inversion.search import parse_search_settings, search_mixing_fraction


def test_search_recovers_fraction_with_exhaustive_grid() -> None:
    true_fraction = 0.37
    metric_names, primary_metric, metric_specs = parse_metric_config(
        {"enabled": ["l2", "ncc"], "primary": "l2"}
    )
    assert metric_names == ["l2", "ncc"]

    settings = parse_search_settings(
        {
            "strategy": "exhaustive_grid",
            "exhaustive_step": 0.005,
            "grid_steps": [0.05, 0.01],
            "refine_window_steps": 4,
        }
    )

    def evaluate_fraction(fraction: float) -> dict[str, float]:
        l2_score = (fraction - true_fraction) ** 2
        ncc_score = 1.0 - l2_score
        return {"l2": l2_score, "ncc": ncc_score}

    result = search_mixing_fraction(
        evaluate_fraction=evaluate_fraction,
        metric_specs=metric_specs,
        primary_metric=primary_metric,
        settings=settings,
    )
    recovered_fraction = result.metric_best["l2"].fraction
    assert abs(recovered_fraction - true_fraction) <= 0.01
    assert result.evaluated_points > 100
