"""Metrics for image reconstruction quality."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Compute PSNR for a batch of images.

    Parameters
    ----------
    pred:
        Predicted images, shape (B, 1, H, W).
    target:
        Target images, shape (B, 1, H, W).
    data_range:
        Max value of the data range (default 1.0).

    Returns
    -------
    torch.Tensor
        PSNR values per batch element.
    """
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2, 3))
    psnr_val = 20.0 * torch.log10(pred.new_tensor(data_range))
    psnr_val = psnr_val - 10.0 * torch.log10(mse + 1e-8)
    return psnr_val


def _gaussian_kernel(window_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(window_size, device=device) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel = g[:, None] * g[None, :]
    return kernel.view(1, 1, window_size, window_size)


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Compute SSIM for a batch of images.

    Parameters
    ----------
    pred:
        Predicted images, shape (B, 1, H, W).
    target:
        Target images, shape (B, 1, H, W).
    data_range:
        Max value of the data range (default 1.0).
    window_size:
        Window size for local statistics.
    sigma:
        Gaussian sigma.

    Returns
    -------
    torch.Tensor
        SSIM values per batch element.
    """
    if pred.size(1) != 1 or target.size(1) != 1:
        raise ValueError("SSIM expects single-channel images.")

    device = pred.device
    kernel = _gaussian_kernel(window_size, sigma, device)
    padding = window_size // 2

    mu_x = F.conv2d(pred, kernel, padding=padding)
    mu_y = F.conv2d(target, kernel, padding=padding)

    sigma_x = F.conv2d(pred * pred, kernel, padding=padding) - mu_x**2
    sigma_y = F.conv2d(target * target, kernel, padding=padding) - mu_y**2
    sigma_xy = F.conv2d(pred * target, kernel, padding=padding) - mu_x * mu_y

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    ssim_map = numerator / (denominator + 1e-8)

    return ssim_map.mean(dim=(1, 2, 3))


def aggregate_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> Tuple[float, float]:
    """Aggregate PSNR and SSIM over a batch."""
    psnr_vals = psnr(pred, target, data_range=data_range)
    ssim_vals = ssim(pred, target, data_range=data_range)
    return float(psnr_vals.mean().item()), float(ssim_vals.mean().item())
