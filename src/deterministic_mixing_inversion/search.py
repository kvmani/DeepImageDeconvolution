"""Mixing-fraction search routines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

from src.deterministic_mixing_inversion.metrics import MetricSpec, is_better


@dataclass(frozen=True)
class SearchSettings:
    """Search settings for mixing-fraction inversion."""

    strategy: str
    exhaustive_step: float
    coarse_steps: Tuple[float, ...]
    refine_window_steps: float


@dataclass(frozen=True)
class MetricBest:
    """Best search result for one metric."""

    fraction: float
    score: float
    top_margin: Optional[float]


@dataclass(frozen=True)
class SearchResult:
    """Aggregate search output."""

    metric_best: Dict[str, MetricBest]
    score_curves: Dict[str, List[Tuple[float, float]]]
    evaluated_points: int


def parse_search_settings(search_cfg: Dict[str, object]) -> SearchSettings:
    """Parse search settings from configuration."""
    strategy = str(search_cfg.get("strategy", "coarse_to_fine")).lower()
    coarse_steps_raw = search_cfg.get("grid_steps", [0.05, 0.01, 0.002])
    if not isinstance(coarse_steps_raw, list) or not coarse_steps_raw:
        raise ValueError("search.grid_steps must be a non-empty list of positive steps.")
    coarse_steps = tuple(float(step) for step in coarse_steps_raw)
    if any(step <= 0.0 for step in coarse_steps):
        raise ValueError("search.grid_steps values must be > 0.")
    exhaustive_step = float(search_cfg.get("exhaustive_step", coarse_steps[-1]))
    if exhaustive_step <= 0.0:
        raise ValueError("search.exhaustive_step must be > 0.")
    refine_window_steps = float(search_cfg.get("refine_window_steps", 4.0))
    if refine_window_steps <= 0.0:
        raise ValueError("search.refine_window_steps must be > 0.")
    if strategy not in {"coarse_to_fine", "exhaustive_grid"}:
        raise ValueError("search.strategy must be one of {'coarse_to_fine', 'exhaustive_grid'}.")
    return SearchSettings(
        strategy=strategy,
        exhaustive_step=exhaustive_step,
        coarse_steps=coarse_steps,
        refine_window_steps=refine_window_steps,
    )


def _build_dense_grid(step: float, lower: float = 0.0, upper: float = 1.0) -> np.ndarray:
    point_count = int(np.floor((upper - lower) / step + 0.5)) + 1
    grid = np.linspace(lower, upper, point_count, dtype=np.float32)
    return np.clip(grid, 0.0, 1.0)


def _metric_top_margin(scores: np.ndarray, objective: str) -> Optional[float]:
    if scores.size < 2:
        return None
    if objective == "max":
        order = np.sort(scores)[::-1]
        return float(order[0] - order[1])
    if objective == "min":
        order = np.sort(scores)
        return float(order[1] - order[0])
    raise ValueError(f"Unknown objective '{objective}'.")


def search_mixing_fraction(
    evaluate_fraction: Callable[[float], Dict[str, float]],
    metric_specs: Mapping[str, MetricSpec],
    primary_metric: str,
    settings: SearchSettings,
) -> SearchResult:
    """Search for best mixing fractions across configured metrics.

    Parameters
    ----------
    evaluate_fraction:
        Callable returning metric values for a requested fraction.
    metric_specs:
        Metric definitions keyed by metric name.
    primary_metric:
        Metric used to guide coarse-to-fine refinement.
    settings:
        Search settings.

    Returns
    -------
    SearchResult
        Search curves and best values per metric.
    """
    cache: Dict[int, Dict[str, float]] = {}
    cache_fraction: Dict[int, float] = {}

    def evaluate_once(fraction: float) -> Dict[str, float]:
        clipped_fraction = float(np.clip(fraction, 0.0, 1.0))
        key = int(round(clipped_fraction * 1_000_000))
        if key not in cache:
            cache[key] = evaluate_fraction(clipped_fraction)
            cache_fraction[key] = clipped_fraction
        return cache[key]

    if settings.strategy == "exhaustive_grid":
        exhaustive_grid = _build_dense_grid(settings.exhaustive_step)
        for fraction in exhaustive_grid:
            evaluate_once(float(fraction))
    else:
        best_fraction = 0.5
        previous_step = settings.coarse_steps[0]
        for level_index, step in enumerate(settings.coarse_steps):
            if level_index == 0:
                lower = 0.0
                upper = 1.0
            else:
                half_width = settings.refine_window_steps * previous_step
                lower = max(0.0, best_fraction - half_width)
                upper = min(1.0, best_fraction + half_width)
            level_grid = _build_dense_grid(step, lower=lower, upper=upper)
            best_score: Optional[float] = None
            for fraction in level_grid:
                metric_values = evaluate_once(float(fraction))
                score_value = float(metric_values[primary_metric])
                if is_better(score_value, best_score, metric_specs[primary_metric].objective):
                    best_score = score_value
                    best_fraction = float(fraction)
            previous_step = step

    sorted_items = sorted(cache.items(), key=lambda item: cache_fraction[item[0]])
    score_curves: Dict[str, List[Tuple[float, float]]] = {name: [] for name in metric_specs}
    for key, metric_values in sorted_items:
        fraction = cache_fraction[key]
        for metric_name in metric_specs:
            score_curves[metric_name].append((fraction, float(metric_values[metric_name])))

    metric_best: Dict[str, MetricBest] = {}
    for metric_name, spec in metric_specs.items():
        fractions = np.asarray([point[0] for point in score_curves[metric_name]], dtype=np.float32)
        scores = np.asarray([point[1] for point in score_curves[metric_name]], dtype=np.float32)
        if scores.size == 0:
            raise ValueError(f"No scores computed for metric '{metric_name}'.")
        if spec.objective == "max":
            best_index = int(np.argmax(scores))
        else:
            best_index = int(np.argmin(scores))
        metric_best[metric_name] = MetricBest(
            fraction=float(fractions[best_index]),
            score=float(scores[best_index]),
            top_margin=_metric_top_margin(scores, spec.objective),
        )

    return SearchResult(
        metric_best=metric_best,
        score_curves=score_curves,
        evaluated_points=len(cache),
    )

