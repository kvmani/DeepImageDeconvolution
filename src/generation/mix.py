"""Synthetic mixing functions for Kikuchi patterns."""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from src.preprocessing.normalise import NormalizeMethod, normalize_image


MixPipeline = Literal["normalize_then_mix", "mix_then_normalize"]


def mix_normalize_then_mix(
    image_a: np.ndarray,
    image_b: np.ndarray,
    weight_a: float,
    normalize_after_mix: bool,
    normalize_method: NormalizeMethod,
    mask: Optional[np.ndarray] = None,
    normalize_smart: bool = False,
) -> np.ndarray:
    """Mix two normalized images with optional re-normalization.

    Parameters
    ----------
    image_a:
        Normalized image A in [0, 1].
    image_b:
        Normalized image B in [0, 1].
    weight_a:
        Weight for image A.
    normalize_after_mix:
        Whether to normalize the mixed image.
    normalize_method:
        Method used if normalize_after_mix is True.
    mask:
        Optional boolean mask for smart normalization on the mixed image.
    normalize_smart:
        When True and mask is provided, compute normalization statistics from the active region.

    Returns
    -------
    numpy.ndarray
        Mixed image in [0, 1].
    """
    weight_b = 1.0 - weight_a
    mixed = (weight_a * image_a) + (weight_b * image_b)
    if normalize_after_mix:
        mixed = normalize_image(
            mixed, method=normalize_method, mask=mask, smart_minmax=normalize_smart
        )
    return mixed.astype(np.float32)


def mix_then_normalize(
    image_a: np.ndarray,
    image_b: np.ndarray,
    weight_a: float,
    normalize_method: NormalizeMethod,
    mask: Optional[np.ndarray] = None,
    normalize_smart: bool = False,
) -> np.ndarray:
    """Mix raw images then normalize the mixture.

    Parameters
    ----------
    image_a:
        Image A in [0, 1].
    image_b:
        Image B in [0, 1].
    weight_a:
        Weight for image A.
    normalize_method:
        Normalization method for the mixture.
    mask:
        Optional boolean mask for smart normalization on the mixed image.
    normalize_smart:
        When True and mask is provided, compute normalization statistics from the active region.

    Returns
    -------
    numpy.ndarray
        Mixed image in [0, 1].
    """
    weight_b = 1.0 - weight_a
    mixed = (weight_a * image_a) + (weight_b * image_b)
    mixed = normalize_image(
        mixed, method=normalize_method, mask=mask, smart_minmax=normalize_smart
    )
    return mixed.astype(np.float32)
