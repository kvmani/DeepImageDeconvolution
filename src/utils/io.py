"""Image IO utilities with 16-bit normalization and format support."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = (".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg")


def collect_image_paths(directory: Path, recursive: bool = False) -> List[Path]:
    """Collect image paths from a directory.

    Parameters
    ----------
    directory:
        Directory to scan for images.
    recursive:
        When True, scan subdirectories recursively.

    Returns
    -------
    list of Path
        Sorted list of image paths.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Input directory not found: {directory}")
    if recursive:
        iterator = directory.rglob("*")
    else:
        iterator = directory.iterdir()
    paths = [p for p in iterator if p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(paths)


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        if image.shape[2] == 1:
            return image[..., 0]
        weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        gray = image[..., :3].astype(np.float32) @ weights
        if np.issubdtype(image.dtype, np.integer):
            max_val = np.iinfo(image.dtype).max
            return np.clip(np.round(gray), 0, max_val).astype(image.dtype)
        return gray.astype(np.float32)
    raise ValueError(f"Expected 2D/3D image array, got shape={image.shape}.")


def _to_uint16(image: np.ndarray) -> np.ndarray:
    image = _to_grayscale(image)
    if image.dtype == np.uint16:
        return image
    if np.issubdtype(image.dtype, np.integer):
        max_val = np.iinfo(image.dtype).max
        scale = 65535.0 / float(max_val)
        return np.clip(image.astype(np.float32) * scale, 0, 65535).round().astype(np.uint16)
    if np.issubdtype(image.dtype, np.floating):
        max_val = float(image.max()) if image.size else 0.0
        if max_val <= 1.0:
            scale = 65535.0
        else:
            scale = 65535.0 / max_val if max_val > 0 else 0.0
        return np.clip(image.astype(np.float32) * scale, 0, 65535).round().astype(np.uint16)
    raise ValueError(f"Unsupported image dtype: {image.dtype}.")


def read_image_16bit(path: Path) -> np.ndarray:
    """Read a grayscale image and return a 16-bit array.

    Parameters
    ----------
    path:
        Path to the image file.

    Returns
    -------
    numpy.ndarray
        2D uint16 array scaled to full 16-bit range if needed.
    """
    with Image.open(path) as img:
        image = np.array(img)
    return _to_uint16(image)


def write_image_16bit(path: Path, image: np.ndarray) -> None:
    """Write a 16-bit grayscale image.

    Parameters
    ----------
    path:
        Destination path.
    image:
        Image array as uint16 or float32 in [0, 1].
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}.")
    if image.dtype == np.uint16:
        array = image
    else:
        if image.min() < 0 or image.max() > 1:
            raise ValueError(
                "Float image must be in [0, 1] before saving to 16-bit format."
            )
        array = np.clip(image * 65535.0, 0, 65535).astype(np.uint16)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="I;16").save(path)


def write_image_8bit(path: Path, image: np.ndarray) -> None:
    """Write an 8-bit image for visualization.

    Parameters
    ----------
    path:
        Destination path.
    image:
        Image array as float in [0, 1] or uint16.
    """
    if image.dtype == np.uint16:
        scaled = (image.astype(np.float32) / 65535.0)
    else:
        scaled = image.astype(np.float32)
    array = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="L").save(path)


def to_float01(image: np.ndarray) -> np.ndarray:
    """Convert a uint16 image to float32 in [0, 1]."""
    if image.dtype != np.uint16:
        raise ValueError(f"Expected uint16 image, got dtype={image.dtype}.")
    return image.astype(np.float32) / 65535.0


def _to_float01_non16(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    if np.issubdtype(image.dtype, np.integer):
        max_val = np.iinfo(image.dtype).max
        return image.astype(np.float32) / float(max_val)
    if np.issubdtype(image.dtype, np.floating):
        max_val = float(image.max())
        if max_val > 1.0:
            return (image / max_val).astype(np.float32)
        return image.astype(np.float32)
    raise ValueError(f"Unsupported image dtype: {image.dtype}.")


def to_float01_any(image: np.ndarray, allow_non_16bit: bool = False) -> np.ndarray:
    """Convert an image to float32 in [0, 1].

    Parameters
    ----------
    image:
        Input image array.
    allow_non_16bit:
        When False, only uint16 inputs are accepted.
    """
    image = _to_grayscale(image)
    if image.dtype == np.uint16:
        return to_float01(image)
    if not allow_non_16bit:
        raise ValueError(
            f"Expected uint16 image, got dtype={image.dtype}. "
            "Set allow_non_16bit=True for exploratory use."
        )
    return _to_float01_non16(image)


def read_image_float01(path: Path, allow_non_16bit: bool = False) -> np.ndarray:
    """Read a grayscale image and return float32 in [0, 1]."""
    with Image.open(path) as img:
        image = np.array(img)
    return to_float01_any(image, allow_non_16bit=allow_non_16bit)


def read_image_float01_with_meta(
    path: Path, allow_non_16bit: bool = False
) -> tuple[np.ndarray, Dict[str, object]]:
    """Read a grayscale image and return float32 in [0, 1] with metadata."""
    with Image.open(path) as img:
        image = np.array(img)
    meta: Dict[str, object] = {"path": str(path), "dtype": str(image.dtype)}
    meta["is_16bit"] = bool(image.dtype == np.uint16)
    return to_float01_any(image, allow_non_16bit=allow_non_16bit), meta
