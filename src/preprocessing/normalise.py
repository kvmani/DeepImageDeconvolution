"""Normalization utilities for 16-bit images."""
from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np


NormalizeMethod = Literal["min_max", "max", "percentile", "histogram_equalization"]


def normalize_image(
    image: np.ndarray,
    method: NormalizeMethod = "min_max",
    eps: float = 1e-6,
    percentile: Tuple[float, float] = (1.0, 99.0),
    histogram_bins: int = 4096,
    mask: Optional[np.ndarray] = None,
    smart_minmax: bool = False,
) -> np.ndarray:
    """Normalize an image to [0, 1].

    Parameters
    ----------
    image:
        Input image, float32.
    method:
        Normalization method.
    eps:
        Small constant to avoid division by zero.
    percentile:
        Low and high percentiles for percentile normalization.
    histogram_bins:
        Number of bins for histogram equalization.
    mask:
        Optional boolean mask for the active region.
    smart_minmax:
        When True and mask is provided, compute normalization statistics from the active region.

    Returns
    -------
    numpy.ndarray
        Normalized image in [0, 1].
    """
    if image.dtype not in (np.float32, np.float64):
        raise ValueError("normalize_image expects float input in [0, 1].")

    active = image
    if mask is not None:
        if mask.shape != image.shape:
            raise ValueError("Mask shape must match image shape.")
        active = image[mask]

    if method == "min_max":
        if smart_minmax and mask is not None:
            min_val = float(active.min()) if active.size else 0.0
            max_val = float(active.max()) if active.size else 0.0
        else:
            min_val = float(image.min())
            max_val = float(image.max())
        if max_val - min_val < eps:
            normalized = np.zeros_like(image, dtype=np.float32)
        else:
            normalized = ((image - min_val) / (max_val - min_val + eps)).astype(np.float32)
        if mask is not None:
            normalized[~mask] = 0.0
        return normalized

    if method == "max":
        if smart_minmax and mask is not None:
            max_val = float(active.max()) if active.size else 0.0
        else:
            max_val = float(image.max())
        if max_val < eps:
            normalized = np.zeros_like(image, dtype=np.float32)
        else:
            normalized = (image / (max_val + eps)).astype(np.float32)
        if mask is not None:
            normalized[~mask] = 0.0
        return normalized

    if method == "percentile":
        if smart_minmax and mask is not None:
            low, high = np.percentile(active, percentile) if active.size else (0.0, 0.0)
        else:
            low, high = np.percentile(image, percentile)
        if high - low < eps:
            normalized = np.zeros_like(image, dtype=np.float32)
        else:
            clipped = np.clip(image, low, high)
            normalized = ((clipped - low) / (high - low + eps)).astype(np.float32)
        if mask is not None:
            normalized[~mask] = 0.0
        return normalized

    if method == "histogram_equalization":
        normalized = histogram_equalization(
            image,
            bins=histogram_bins,
            mask=mask if smart_minmax and mask is not None else None,
        ).astype(np.float32)
        if mask is not None:
            normalized[~mask] = 0.0
        return normalized

    raise ValueError(f"Unknown normalization method: {method}")


def histogram_equalization(
    image: np.ndarray, bins: int = 4096, mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Apply histogram equalization to a float image in [0, 1].

    Parameters
    ----------
    image:
        Input image, float32 in [0, 1].
    bins:
        Number of histogram bins.
    mask:
        Optional boolean mask to compute statistics on the active region only.

    Returns
    -------
    numpy.ndarray
        Histogram-equalized image in [0, 1].
    """
    if image.min() < 0 or image.max() > 1:
        raise ValueError("histogram_equalization expects image in [0, 1].")

    if mask is not None and mask.shape != image.shape:
        raise ValueError("Mask shape must match image shape.")

    if mask is None:
        flat = image.flatten()
    else:
        flat = image[mask].flatten()
    hist, bin_edges = np.histogram(flat, bins=bins, range=(0.0, 1.0))
    cdf = hist.cumsum().astype(np.float32)
    if cdf[-1] == 0:
        return np.zeros_like(image, dtype=np.float32)
    cdf /= cdf[-1]
    if mask is None:
        equalized = np.interp(flat, bin_edges[:-1], cdf)
        return equalized.reshape(image.shape).astype(np.float32)
    mapped = np.interp(flat, bin_edges[:-1], cdf)
    output = np.zeros_like(image, dtype=np.float32)
    output[mask] = mapped.astype(np.float32)
    return output
