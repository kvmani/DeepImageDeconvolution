"""Metrics for deterministic mixing-fraction inversion."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping

import numpy as np
import torch

from src.utils.metrics import ssim as torch_ssim
from src.utils.metrics import ssim_masked as torch_ssim_masked


@dataclass(frozen=True)
class MetricSpec:
    """Metric definition with optimization objective."""

    name: str
    objective: str


SUPPORTED_METRICS: Mapping[str, MetricSpec] = {
    "l1": MetricSpec(name="l1", objective="min"),
    "l2": MetricSpec(name="l2", objective="min"),
    "ncc": MetricSpec(name="ncc", objective="max"),
    "ssim": MetricSpec(name="ssim", objective="max"),
}


def parse_metric_config(metrics_cfg: Dict[str, object]) -> tuple[List[str], str, Dict[str, MetricSpec]]:
    """Parse metric configuration.

    Parameters
    ----------
    metrics_cfg:
        Metrics configuration dictionary.

    Returns
    -------
    tuple
        `(enabled_metric_names, primary_metric_name, metric_specs)`.
    """
    enabled_raw = metrics_cfg.get("enabled", ["ncc", "ssim", "l2", "l1"])
    enabled_metrics: List[str] = []
    for metric_name in enabled_raw:
        normalized = str(metric_name).lower()
        if normalized not in SUPPORTED_METRICS:
            raise ValueError(f"Unsupported metric '{metric_name}'.")
        if normalized not in enabled_metrics:
            enabled_metrics.append(normalized)
    if not enabled_metrics:
        raise ValueError("At least one metric must be enabled.")

    primary_metric = str(metrics_cfg.get("primary", enabled_metrics[0])).lower()
    if primary_metric not in enabled_metrics:
        raise ValueError("metrics.primary must be present in metrics.enabled.")

    metric_specs = {name: SUPPORTED_METRICS[name] for name in enabled_metrics}
    return enabled_metrics, primary_metric, metric_specs


def is_better(candidate_score: float, incumbent_score: float | None, objective: str) -> bool:
    """Return whether a candidate score is better than incumbent score."""
    if incumbent_score is None:
        return True
    if objective == "max":
        return candidate_score > incumbent_score
    if objective == "min":
        return candidate_score < incumbent_score
    raise ValueError(f"Unknown objective '{objective}'.")


def _masked_values(image: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return image.reshape(-1)
    return image[mask]


def masked_l1(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray | None) -> float:
    """Compute masked L1."""
    diff = np.abs(prediction - target)
    values = _masked_values(diff, mask)
    if values.size == 0:
        return 0.0
    return float(values.mean())


def masked_l2(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray | None) -> float:
    """Compute masked L2 (MSE)."""
    diff = prediction - target
    values = _masked_values(diff * diff, mask)
    if values.size == 0:
        return 0.0
    return float(values.mean())


def masked_ncc(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray | None) -> float:
    """Compute masked normalized cross-correlation (Pearson)."""
    prediction_values = _masked_values(prediction, mask)
    target_values = _masked_values(target, mask)
    if prediction_values.size == 0 or target_values.size == 0:
        return 0.0
    pred_centered = prediction_values - prediction_values.mean()
    target_centered = target_values - target_values.mean()
    numerator = float(np.sum(pred_centered * target_centered))
    denom_pred = float(np.sqrt(np.sum(pred_centered * pred_centered)))
    denom_target = float(np.sqrt(np.sum(target_centered * target_centered)))
    denominator = max(denom_pred * denom_target, 1e-12)
    return numerator / denominator


def masked_ssim(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray | None) -> float:
    """Compute SSIM using shared torch metric implementation."""
    prediction_tensor = torch.from_numpy(prediction.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    target_tensor = torch.from_numpy(target.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    if mask is None:
        return float(torch_ssim(prediction_tensor, target_tensor).item())
    mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return float(torch_ssim_masked(prediction_tensor, target_tensor, mask_tensor).item())


def compute_metric_values(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None,
    enabled_metrics: Iterable[str],
) -> Dict[str, float]:
    """Compute all configured metric scores for one prediction/target pair."""
    results: Dict[str, float] = {}
    for metric_name in enabled_metrics:
        if metric_name == "l1":
            results["l1"] = masked_l1(prediction, target, mask)
        elif metric_name == "l2":
            results["l2"] = masked_l2(prediction, target, mask)
        elif metric_name == "ncc":
            results["ncc"] = masked_ncc(prediction, target, mask)
        elif metric_name == "ssim":
            results["ssim"] = masked_ssim(prediction, target, mask)
        else:
            raise ValueError(f"Unsupported metric '{metric_name}'.")
    return results

