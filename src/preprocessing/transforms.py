"""Basic preprocessing transforms for Kikuchi patterns."""
from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np


CropMode = Literal["center", "random"]


def center_crop(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Center crop an image to (height, width)."""
    height, width = image.shape
    target_h, target_w = size
    if target_h > height or target_w > width:
        raise ValueError("Crop size must be <= image size.")
    top = (height - target_h) // 2
    left = (width - target_w) // 2
    return image[top : top + target_h, left : left + target_w]


def random_crop(
    image: np.ndarray, size: Tuple[int, int], rng: np.random.Generator
) -> np.ndarray:
    """Random crop an image to (height, width)."""
    height, width = image.shape
    target_h, target_w = size
    if target_h > height or target_w > width:
        raise ValueError("Crop size must be <= image size.")
    top = int(rng.integers(0, height - target_h + 1))
    left = int(rng.integers(0, width - target_w + 1))
    return image[top : top + target_h, left : left + target_w]


def apply_crop(
    image: np.ndarray,
    size: Optional[Tuple[int, int]],
    mode: CropMode,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply optional cropping."""
    if size is None:
        return image
    if mode == "center":
        return center_crop(image, size)
    if mode == "random":
        return random_crop(image, size, rng)
    raise ValueError(f"Unknown crop mode: {mode}")


def apply_flip(image: np.ndarray, flip_horizontal: bool, flip_vertical: bool) -> np.ndarray:
    """Apply flips to the image."""
    if flip_horizontal:
        image = np.fliplr(image)
    if flip_vertical:
        image = np.flipud(image)
    return image


def apply_rotation_90(image: np.ndarray, k: int) -> np.ndarray:
    """Rotate by 90 degrees k times."""
    if k % 4 == 0:
        return image
    return np.rot90(image, k=k)


def apply_gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur to simulate detector point spread."""
    if sigma <= 0:
        return image
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError as exc:
        raise ImportError(
            "scipy is required for Gaussian blur. Install scipy or disable blur."
        ) from exc
    return gaussian_filter(image, sigma=sigma).astype(np.float32)


def apply_gaussian_noise(
    image: np.ndarray, std: float, rng: np.random.Generator
) -> np.ndarray:
    """Add Gaussian noise to an image."""
    if std <= 0:
        return image
    noise = rng.normal(0.0, std, size=image.shape).astype(np.float32)
    return np.clip(image + noise, 0.0, 1.0)
