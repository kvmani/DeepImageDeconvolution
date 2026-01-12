"""Training-time metrics and physics checks."""
from __future__ import annotations

from typing import Dict

import torch
from torch.nn import functional as F

from src.utils.metrics import aggregate_metrics, mse


def compute_reconstruction_metrics(
    pred_a: torch.Tensor,
    pred_b: torch.Tensor,
    x_hat: torch.Tensor,
    target_a: torch.Tensor,
    target_b: torch.Tensor,
    mixed_c: torch.Tensor,
    x_true: torch.Tensor | None = None,
) -> Dict[str, float]:
    """Compute PSNR/SSIM, reconstruction, and weight metrics for a batch."""
    psnr_a, ssim_a = aggregate_metrics(pred_a, target_a)
    psnr_b, ssim_b = aggregate_metrics(pred_b, target_b)

    metrics = {
        "psnr_a": psnr_a,
        "ssim_a": ssim_a,
        "psnr_b": psnr_b,
        "ssim_b": ssim_b,
        "l2_a": float(mse(pred_a, target_a).mean().item()),
        "l2_b": float(mse(pred_b, target_b).mean().item()),
    }

    metrics.update(physics_consistency(pred_a, pred_b, x_hat, mixed_c))
    if x_true is not None:
        metrics["x_mae"] = float((x_true - x_hat).abs().mean().item())
    return metrics


def physics_consistency(
    pred_a: torch.Tensor,
    pred_b: torch.Tensor,
    x_hat: torch.Tensor,
    mixed_c: torch.Tensor,
) -> Dict[str, float]:
    """Compute physics-driven consistency checks for xA + yB ≈ C."""
    x_weight = x_hat.view(-1, 1, 1, 1)
    recon = (x_weight * pred_a) + ((1.0 - x_weight) * pred_b)
    return {
        "l1_recon": float(F.l1_loss(recon, mixed_c).item()),
        "l2_recon": float(mse(recon, mixed_c).mean().item()),
        "max_abs_recon_error": float((recon - mixed_c).abs().max().item()),
    }
