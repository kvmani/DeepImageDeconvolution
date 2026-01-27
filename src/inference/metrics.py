"""Inference metrics helpers."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from src.utils.metrics import psnr, ssim


def compute_image_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute L1, L2, PSNR, and SSIM metrics for a single image.

    Parameters
    ----------
    pred:
        Predicted image as float32 in [0, 1].
    target:
        Target image as float32 in [0, 1].
    mask:
        Optional binary mask to apply before computing metrics.

    Returns
    -------
    dict
        Dictionary with l1, l2, psnr, and ssim entries.
    """
    pred_t = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0)
    target_t = torch.from_numpy(target).unsqueeze(0).unsqueeze(0)
    if mask is not None:
        mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
        pred_t = pred_t * mask_t
        target_t = target_t * mask_t

    diff = pred_t - target_t
    l1 = float(diff.abs().mean().item())
    l2 = float((diff**2).mean().item())
    psnr_val = float(psnr(pred_t, target_t).mean().item())
    ssim_val = float(ssim(pred_t, target_t).mean().item())
    return {"l1": l1, "l2": l2, "psnr": psnr_val, "ssim": ssim_val}


def summarize_metrics(metrics_list: list[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate metrics across samples."""
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    summary: Dict[str, float] = {}
    for key in keys:
        values = [metrics[key] for metrics in metrics_list if key in metrics]
        summary[key] = float(np.mean(values)) if values else 0.0
    return summary
