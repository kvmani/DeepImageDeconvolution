"""Preprocessing helpers shared across datasets and inference."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from src.preprocessing.mask import apply_circular_mask
from src.preprocessing.normalise import normalize_with_metadata


@dataclass
class PreprocessConfig:
    """Preprocessing configuration for datasets and inference."""

    crop_enabled: bool
    crop_size: Optional[Tuple[int, int]]
    crop_mode: str
    mask_enabled: bool
    detect_existing: bool
    outside_zero_fraction: float
    zero_tolerance: float
    normalize_enabled: bool
    normalize_method: str
    normalize_smart: bool
    histogram_bins: int
    percentile: Tuple[float, float]
    augment_enabled: bool
    flip_horizontal: bool
    flip_vertical: bool
    rotate90: bool


def parse_preprocess_cfg(cfg: Dict) -> PreprocessConfig:
    """Parse a preprocessing configuration dictionary.

    Parameters
    ----------
    cfg:
        Preprocessing configuration dictionary.

    Returns
    -------
    PreprocessConfig
        Parsed configuration.
    """
    crop_cfg = cfg.get("crop", {})
    mask_cfg = cfg.get("mask", {})
    normalize_cfg = cfg.get("normalize", {})
    augment_cfg = cfg.get("augment", {})
    return PreprocessConfig(
        crop_enabled=bool(crop_cfg.get("enabled", False)),
        crop_size=tuple(crop_cfg.get("size", [])) if crop_cfg.get("size") else None,
        crop_mode=str(crop_cfg.get("mode", "center")),
        mask_enabled=bool(mask_cfg.get("enabled", True)),
        detect_existing=bool(mask_cfg.get("detect_existing", True)),
        outside_zero_fraction=float(mask_cfg.get("outside_zero_fraction", 0.98)),
        zero_tolerance=float(mask_cfg.get("zero_tolerance", 1e-6)),
        normalize_enabled=bool(normalize_cfg.get("enabled", False)),
        normalize_method=str(normalize_cfg.get("method", "min_max")),
        normalize_smart=bool(normalize_cfg.get("smart", True)),
        histogram_bins=int(normalize_cfg.get("histogram_bins", 4096)),
        percentile=tuple(normalize_cfg.get("percentile", (1.0, 99.0))),
        augment_enabled=bool(augment_cfg.get("enabled", False)),
        flip_horizontal=bool(augment_cfg.get("flip_horizontal", True)),
        flip_vertical=bool(augment_cfg.get("flip_vertical", True)),
        rotate90=bool(augment_cfg.get("rotate90", True)),
    )


def apply_preprocess(
    image: np.ndarray,
    cfg: PreprocessConfig,
    mask: Optional[np.ndarray],
) -> np.ndarray:
    """Apply mask and normalization preprocessing steps.

    Parameters
    ----------
    image:
        Input image as a float array in [0, 1].
    cfg:
        Parsed preprocessing configuration.
    mask:
        Optional circular mask to apply.

    Returns
    -------
    numpy.ndarray
        Preprocessed image.
    """
    if cfg.mask_enabled and mask is not None:
        image = apply_circular_mask(image, mask)

    if cfg.normalize_enabled:
        image, _ = normalize_with_metadata(
            image,
            method=cfg.normalize_method,
            histogram_bins=cfg.histogram_bins,
            percentile=cfg.percentile,
            mask=mask,
            smart_minmax=cfg.normalize_smart,
        )

    return image.astype(np.float32)
