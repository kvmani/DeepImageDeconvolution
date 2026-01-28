"""Tests for the pattern mixer GUI core processing."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from apps.pattern_mixer_gui.core.config import NoiseSettings, NormalizationMode
from apps.pattern_mixer_gui.core.processing import (
    apply_noise,
    min_max_normalize,
    mix_images,
    process_images,
    zscore_remap,
)


def test_weights_sum_to_one() -> None:
    image_a = np.ones((4, 4), dtype=np.float32)
    image_b = np.zeros((4, 4), dtype=np.float32)
    mixed = mix_images(image_a, image_b, 0.3)
    assert np.allclose(mixed, 0.3)


def test_min_max_normalization_range() -> None:
    image = np.array([[0.2, 0.6], [0.4, 0.8]], dtype=np.float32)
    result = min_max_normalize(image)
    assert result.image.min() >= 0.0
    assert result.image.max() <= 1.0


def test_zscore_remap_range() -> None:
    image = np.array([[0.0, 0.5], [0.7, 1.0]], dtype=np.float32)
    result = zscore_remap(image)
    assert result.image.min() >= 0.0
    assert result.image.max() <= 1.0


def test_deterministic_noise_with_seed() -> None:
    image = np.full((8, 8), 0.5, dtype=np.float32)
    noise = NoiseSettings(enable_gaussian=True, gaussian_sigma=0.05)
    rng = np.random.default_rng(123)
    out_a = apply_noise(image, None, noise, rng)
    rng = np.random.default_rng(123)
    out_b = apply_noise(image, None, noise, rng)
    assert np.allclose(out_a, out_b)


def test_process_images_with_normalization() -> None:
    image_a = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    image_b = np.linspace(1.0, 0.0, 16, dtype=np.float32).reshape(4, 4)
    processed = process_images(
        image_a=image_a,
        image_b=image_b,
        mask_a=None,
        mask_b=None,
        weight_a=0.5,
        normalization_mode=NormalizationMode.MIN_MAX,
        normalize_output=True,
        noise_a=NoiseSettings(),
        noise_b=NoiseSettings(),
        seed=42,
        normalize_inputs=True,
    )
    assert processed.image_c.min() >= 0.0
    assert processed.image_c.max() <= 1.0
