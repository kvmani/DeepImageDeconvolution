"""Circular mask utilities for Kikuchi patterns."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def build_circular_mask(
    shape: Tuple[int, int],
    radius: Optional[float] = None,
    center: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Build a boolean circular mask for a 2D image.

    Parameters
    ----------
    shape:
        Image shape as (height, width).
    radius:
        Radius of the circle. Defaults to the maximum inscribed circle.
    center:
        Center of the circle (row, col). Defaults to image center.

    Returns
    -------
    numpy.ndarray
        Boolean mask with True inside the circle.
    """
    height, width = shape
    if center is None:
        center = ((height - 1) / 2.0, (width - 1) / 2.0)
    if radius is None:
        radius = min(height, width) / 2.0

    yy, xx = np.ogrid[:height, :width]
    dist_sq = (yy - center[0]) ** 2 + (xx - center[1]) ** 2
    return dist_sq <= radius**2


def detect_circular_mask(
    image: np.ndarray,
    mask: np.ndarray,
    zero_tolerance: float = 1e-6,
    outside_zero_fraction: float = 0.98,
) -> Tuple[bool, float]:
    """Detect whether an image is already masked outside the circle.

    Parameters
    ----------
    image:
        Input image, float32 in [0, 1].
    mask:
        Boolean mask for the active circular region.
    zero_tolerance:
        Absolute tolerance to treat a pixel as zero.
    outside_zero_fraction:
        Fraction of outside pixels that must be near zero to be considered masked.

    Returns
    -------
    tuple
        (is_masked, outside_zero_fraction_measured)
    """
    if image.shape != mask.shape:
        raise ValueError("Mask shape must match image shape.")
    outside = image[~mask]
    if outside.size == 0:
        return True, 1.0
    zero_fraction = float(np.mean(np.abs(outside) <= zero_tolerance))
    return zero_fraction >= outside_zero_fraction, zero_fraction


def apply_circular_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply a circular mask to an image.

    Parameters
    ----------
    image:
        Input image.
    mask:
        Boolean mask for the active region.

    Returns
    -------
    numpy.ndarray
        Masked image with outside pixels set to zero.
    """
    if image.shape != mask.shape:
        raise ValueError("Mask shape must match image shape.")
    masked = image.copy()
    masked[~mask] = 0.0
    return masked
