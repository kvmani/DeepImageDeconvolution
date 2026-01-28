"""Image loading utilities for the pattern mixer GUI."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from src.preprocessing.mask import build_mask_with_metadata
from src.utils.logging import get_logger

_LOGGER = get_logger(__name__)


@dataclass
class LoadedImage:
    """Container for a loaded pattern image."""

    path: Path
    data: np.ndarray
    mask: np.ndarray
    mask_meta: dict
    bit_depth: int
    original_mode: str
    original_shape: Tuple[int, int]


def _infer_bit_depth(array: np.ndarray) -> int:
    if array.dtype == np.uint16:
        return 16
    if array.dtype == np.uint8:
        return 8
    return 16


def _rgb_to_luminance(array: np.ndarray) -> np.ndarray:
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return np.tensordot(array, weights, axes=([-1], [0]))


def load_image(path: Path, apply_mask: bool = True) -> LoadedImage:
    """Load an image and convert to float32 in [0, 1].

    Parameters
    ----------
    path:
        Path to the image.
    apply_mask:
        Whether to apply the maximum inscribed circular mask.

    Returns
    -------
    LoadedImage
        Loaded image container with mask metadata.
    """
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    with Image.open(path) as img:
        original_mode = img.mode
        array = np.array(img)

    bit_depth = _infer_bit_depth(array)
    original_shape = array.shape[:2]

    if array.ndim == 3:
        array = _rgb_to_luminance(array.astype(np.float32))
        _LOGGER.info("Converted RGB image to luminance for %s.", path)

    if bit_depth == 8:
        array = (array.astype(np.float32) / 255.0) * 65535.0
    else:
        array = array.astype(np.float32)

    array = np.clip(array, 0.0, 65535.0)
    float_image = (array / 65535.0).astype(np.float32)

    mask = np.ones_like(float_image, dtype=bool)
    mask_meta = {"enabled": False}
    if apply_mask:
        mask, mask_meta = build_mask_with_metadata(float_image)
        if not mask_meta.get("already_masked", False):
            float_image = np.where(mask, float_image, 0.0)

    _LOGGER.info(
        "Loaded image %s | mode=%s | shape=%s | bit_depth=%s | min=%.4f max=%.4f",
        path,
        original_mode,
        float_image.shape,
        bit_depth,
        float(float_image.min()),
        float(float_image.max()),
    )

    return LoadedImage(
        path=path,
        data=float_image,
        mask=mask,
        mask_meta=mask_meta,
        bit_depth=bit_depth,
        original_mode=original_mode,
        original_shape=original_shape,
    )


def resize_to_match(image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Resize an image to match a target shape.

    Parameters
    ----------
    image:
        Image array.
    target_shape:
        Desired (height, width).

    Returns
    -------
    numpy.ndarray
        Resized image.
    """
    if image.shape == target_shape:
        return image
    pil_img = Image.fromarray(np.clip(image * 65535.0, 0, 65535).astype(np.uint16))
    resized = pil_img.resize((target_shape[1], target_shape[0]), Image.BILINEAR)
    return np.array(resized, dtype=np.float32) / 65535.0
