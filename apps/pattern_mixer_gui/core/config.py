"""Configuration models for the pattern mixer GUI core."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class NormalizationMode(str, Enum):
    """Supported normalization modes."""

    MIN_MAX = "min_max"
    PER_IMAGE_MIN_MAX = "per_image_min_max"
    ZSCORE_REMAP = "zscore_remap"
    NONE = "none"


@dataclass
class NoiseSettings:
    """Noise configuration for a single image."""

    enable_gaussian: bool = False
    gaussian_sigma: float = 0.01
    enable_poisson: bool = False
    poisson_scale: float = 20000.0
    enable_offset: bool = False
    offset_value: float = 0.0


@dataclass
class MixSettings:
    """Mixing configuration shared across strategies."""

    weight_a: float = 0.5
    normalize_output: bool = False
    normalization_mode: NormalizationMode = NormalizationMode.MIN_MAX
    seed: Optional[int] = None
