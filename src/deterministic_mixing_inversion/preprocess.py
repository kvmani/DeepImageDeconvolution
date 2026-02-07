"""Preprocessing for deterministic mixing-fraction inversion."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from src.preprocessing.mask import apply_circular_mask, build_circular_mask, detect_circular_mask


@dataclass(frozen=True)
class PreprocessSettings:
    """Preprocessing settings for deterministic inversion."""

    auto_crop_to_target: bool
    mask_enabled: bool
    detect_existing_mask: bool
    zero_tolerance: float
    outside_zero_fraction: float
    background_enabled: bool
    background_mode: str
    background_sigma: float
    background_epsilon: float
    standardize_enabled: bool
    standardize_epsilon: float
    dog_enabled: bool
    dog_sigma_low: float
    dog_sigma_high: float


def parse_preprocess_settings(preprocess_cfg: Dict[str, object]) -> PreprocessSettings:
    """Parse preprocessing settings from configuration."""
    mask_cfg = preprocess_cfg.get("mask", {}) if isinstance(preprocess_cfg, dict) else {}
    background_cfg = preprocess_cfg.get("background_correction", {}) if isinstance(preprocess_cfg, dict) else {}
    standardize_cfg = preprocess_cfg.get("standardize", {}) if isinstance(preprocess_cfg, dict) else {}
    dog_cfg = preprocess_cfg.get("dog", {}) if isinstance(preprocess_cfg, dict) else {}

    return PreprocessSettings(
        auto_crop_to_target=bool(preprocess_cfg.get("auto_crop_to_target", False)),
        mask_enabled=bool(mask_cfg.get("enabled", True)),
        detect_existing_mask=bool(mask_cfg.get("detect_existing", True)),
        zero_tolerance=float(mask_cfg.get("zero_tolerance", 1e-6)),
        outside_zero_fraction=float(mask_cfg.get("outside_zero_fraction", 0.98)),
        background_enabled=bool(background_cfg.get("enabled", True)),
        background_mode=str(background_cfg.get("mode", "subtractive")).lower(),
        background_sigma=float(background_cfg.get("sigma", 21.0)),
        background_epsilon=float(background_cfg.get("epsilon", 1e-6)),
        standardize_enabled=bool(standardize_cfg.get("enabled", True)),
        standardize_epsilon=float(standardize_cfg.get("epsilon", 1e-6)),
        dog_enabled=bool(dog_cfg.get("enabled", False)),
        dog_sigma_low=float(dog_cfg.get("sigma_low", 1.0)),
        dog_sigma_high=float(dog_cfg.get("sigma_high", 3.0)),
    )


def center_crop_to_shape(image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Center-crop an image to the requested shape."""
    target_height, target_width = target_shape
    source_height, source_width = image.shape
    if target_height > source_height or target_width > source_width:
        raise ValueError(
            f"Target shape {target_shape} exceeds source shape {image.shape}; cannot center-crop."
        )
    top = max((source_height - target_height) // 2, 0)
    left = max((source_width - target_width) // 2, 0)
    return image[top : top + target_height, left : left + target_width]


def match_shape_to_target(
    image: np.ndarray,
    target_shape: Tuple[int, int],
    auto_crop_to_target: bool,
) -> np.ndarray:
    """Return an image matching the target shape."""
    if image.shape == target_shape:
        return image
    if not auto_crop_to_target:
        raise ValueError(f"Image shape {image.shape} does not match target shape {target_shape}.")
    return center_crop_to_shape(image, target_shape)


def build_centered_mask(shape: Tuple[int, int]) -> np.ndarray:
    """Return the maximum inscribed centered circular mask."""
    return build_circular_mask(shape).astype(bool)


def _apply_background_correction(
    image: np.ndarray,
    settings: PreprocessSettings,
    mask: Optional[np.ndarray],
) -> np.ndarray:
    if not settings.background_enabled:
        return image
    blurred = gaussian_filter(image, sigma=settings.background_sigma)
    if settings.background_mode == "subtractive":
        corrected = image - blurred
    elif settings.background_mode == "divisive":
        corrected = image / (blurred + settings.background_epsilon)
    else:
        raise ValueError(
            "background_correction.mode must be one of {'subtractive', 'divisive'}."
        )
    if mask is not None:
        corrected = corrected.copy()
        corrected[~mask] = 0.0
    return corrected.astype(np.float32)


def _apply_standardization(
    image: np.ndarray,
    settings: PreprocessSettings,
    mask: Optional[np.ndarray],
) -> np.ndarray:
    if not settings.standardize_enabled:
        return image
    if mask is not None:
        active = image[mask]
    else:
        active = image.reshape(-1)
    if active.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    mean_value = float(active.mean())
    std_value = float(active.std())
    standardized = (image - mean_value) / max(std_value, settings.standardize_epsilon)
    if mask is not None:
        standardized = standardized.copy()
        standardized[~mask] = 0.0
    return standardized.astype(np.float32)


def _apply_dog(
    image: np.ndarray,
    settings: PreprocessSettings,
    mask: Optional[np.ndarray],
) -> np.ndarray:
    if not settings.dog_enabled:
        return image
    low_blur = gaussian_filter(image, sigma=settings.dog_sigma_low)
    high_blur = gaussian_filter(image, sigma=settings.dog_sigma_high)
    dog_image = low_blur - high_blur
    if mask is not None:
        dog_image = dog_image.copy()
        dog_image[~mask] = 0.0
    return dog_image.astype(np.float32)


def preprocess_pattern(
    image: np.ndarray,
    settings: PreprocessSettings,
    mask: Optional[np.ndarray],
) -> tuple[np.ndarray, Dict[str, object]]:
    """Apply deterministic preprocessing pipeline to one pattern.

    Parameters
    ----------
    image:
        Input image in float32 [0, 1].
    settings:
        Parsed preprocessing settings.
    mask:
        Optional circular mask.

    Returns
    -------
    tuple
        `(preprocessed_image, metadata)`.
    """
    working = image.astype(np.float32, copy=True)
    metadata: Dict[str, object] = {
        "mask_enabled": settings.mask_enabled,
        "already_masked": None,
        "outside_zero_fraction": None,
        "background_mode": settings.background_mode,
        "background_sigma": settings.background_sigma,
        "standardize_enabled": settings.standardize_enabled,
        "dog_enabled": settings.dog_enabled,
    }

    if settings.mask_enabled and mask is not None:
        if settings.detect_existing_mask:
            already_masked, outside_fraction = detect_circular_mask(
                working,
                mask,
                zero_tolerance=settings.zero_tolerance,
                outside_zero_fraction=settings.outside_zero_fraction,
            )
            metadata["already_masked"] = already_masked
            metadata["outside_zero_fraction"] = outside_fraction
        working = apply_circular_mask(working, mask)

    working = _apply_background_correction(working, settings, mask if settings.mask_enabled else None)
    working = _apply_standardization(working, settings, mask if settings.mask_enabled else None)
    working = _apply_dog(working, settings, mask if settings.mask_enabled else None)
    return working.astype(np.float32), metadata

