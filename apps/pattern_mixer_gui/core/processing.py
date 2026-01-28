"""Core processing utilities for pattern mixing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from src.preprocessing.mask import apply_circular_mask
from src.utils.logging import get_logger

from .config import NormalizationMode, NoiseSettings

_LOGGER = get_logger(__name__)


@dataclass
class ProcessedImages:
    """Processed images for mixing."""

    image_a: np.ndarray
    image_b: np.ndarray
    image_c: np.ndarray


@dataclass
class NormalizationResult:
    """Normalization output container."""

    image: np.ndarray
    min_value: float
    max_value: float


def _compute_min_max(image: np.ndarray, mask: Optional[np.ndarray]) -> Tuple[float, float]:
    if mask is not None:
        values = image[mask]
    else:
        values = image.ravel()
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    return min_value, max_value


def min_max_normalize(image: np.ndarray, mask: Optional[np.ndarray] = None) -> NormalizationResult:
    """Normalize to [0, 1] using min/max values.

    Parameters
    ----------
    image:
        Input image.
    mask:
        Optional boolean mask to restrict statistics.

    Returns
    -------
    NormalizationResult
        Normalized image and min/max values.
    """
    min_value, max_value = _compute_min_max(image, mask)
    denom = max(max_value - min_value, 1e-8)
    normalized = (image - min_value) / denom
    return NormalizationResult(np.clip(normalized, 0.0, 1.0), min_value, max_value)


def zscore_remap(image: np.ndarray, mask: Optional[np.ndarray] = None) -> NormalizationResult:
    """Z-score normalize then remap to [0, 1].

    The remap uses mean ± 3 * std as the default window.
    """
    if mask is not None:
        values = image[mask]
    else:
        values = image.ravel()
    mean = float(np.mean(values))
    std = float(np.std(values))
    std = max(std, 1e-8)
    z = (image - mean) / std
    window_min = -3.0
    window_max = 3.0
    normalized = (z - window_min) / (window_max - window_min)
    return NormalizationResult(np.clip(normalized, 0.0, 1.0), window_min, window_max)


def apply_noise(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    settings: NoiseSettings,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply configured noise to an image.

    Parameters
    ----------
    image:
        Input image in [0, 1].
    mask:
        Optional mask to preserve outside zeros.
    settings:
        Noise settings.
    rng:
        Random number generator.

    Returns
    -------
    numpy.ndarray
        Noisy image.
    """
    noisy = image.copy()
    if settings.enable_gaussian:
        noisy += rng.normal(0.0, settings.gaussian_sigma, size=noisy.shape).astype(np.float32)
    if settings.enable_poisson:
        scaled = np.clip(noisy, 0.0, 1.0) * settings.poisson_scale
        noisy = rng.poisson(scaled).astype(np.float32) / settings.poisson_scale
    if settings.enable_offset:
        noisy += settings.offset_value
    noisy = np.clip(noisy, 0.0, 1.0)
    if mask is not None:
        noisy = apply_circular_mask(noisy, mask)
    return noisy


def mix_images(image_a: np.ndarray, image_b: np.ndarray, weight_a: float) -> np.ndarray:
    """Mix two images with weight A and weight B = 1 - weight A."""
    weight_a = float(np.clip(weight_a, 0.0, 1.0))
    weight_b = 1.0 - weight_a
    return np.clip(weight_a * image_a + weight_b * image_b, 0.0, 1.0)


def process_images(
    image_a: np.ndarray,
    image_b: np.ndarray,
    mask_a: Optional[np.ndarray],
    mask_b: Optional[np.ndarray],
    weight_a: float,
    normalization_mode: NormalizationMode,
    normalize_output: bool,
    noise_a: NoiseSettings,
    noise_b: NoiseSettings,
    seed: Optional[int] = None,
    normalize_inputs: bool = True,
) -> ProcessedImages:
    """Process and mix images using the specified settings.

    Parameters
    ----------
    image_a:
        Input image A in [0, 1].
    image_b:
        Input image B in [0, 1].
    mask_a:
        Circular mask for A (optional).
    mask_b:
        Circular mask for B (optional).
    weight_a:
        Mixing weight for A.
    normalization_mode:
        Normalization mode to apply.
    normalize_output:
        Whether to normalize C after mixing.
    noise_a:
        Noise settings for A.
    noise_b:
        Noise settings for B.
    seed:
        Seed for deterministic noise.
    normalize_inputs:
        Whether to normalize inputs before mixing.

    Returns
    -------
    ProcessedImages
        Processed A, B, and mixed C arrays.
    """
    rng = np.random.default_rng(seed)

    noisy_a = apply_noise(image_a, mask_a, noise_a, rng)
    noisy_b = apply_noise(image_b, mask_b, noise_b, rng)

    norm_fn: Callable[[np.ndarray, Optional[np.ndarray]], NormalizationResult]
    if normalization_mode in (NormalizationMode.MIN_MAX, NormalizationMode.PER_IMAGE_MIN_MAX):
        norm_fn = min_max_normalize
    elif normalization_mode == NormalizationMode.ZSCORE_REMAP:
        norm_fn = zscore_remap
    else:
        norm_fn = lambda img, _mask=None: NormalizationResult(img, 0.0, 1.0)

    if normalize_inputs and normalization_mode != NormalizationMode.NONE:
        norm_a = norm_fn(noisy_a, mask_a)
        norm_b = norm_fn(noisy_b, mask_b)
        proc_a = norm_a.image
        proc_b = norm_b.image
    else:
        proc_a = noisy_a
        proc_b = noisy_b

    mixed = mix_images(proc_a, proc_b, weight_a)

    if normalize_output and normalization_mode != NormalizationMode.NONE:
        mixed = norm_fn(mixed, None).image

    _LOGGER.debug(
        "Processed images with weight_a=%.3f | norm=%s | normalize_output=%s",
        weight_a,
        normalization_mode,
        normalize_output,
    )

    return ProcessedImages(proc_a, proc_b, mixed)
