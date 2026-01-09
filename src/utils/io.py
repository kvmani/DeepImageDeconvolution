"""Image IO utilities with 16-bit safeguards."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = (".png", ".tif", ".tiff")


def collect_image_paths(directory: Path) -> List[Path]:
    """Collect image paths from a directory.

    Parameters
    ----------
    directory:
        Directory to scan for images.

    Returns
    -------
    list of Path
        Sorted list of image paths.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Input directory not found: {directory}")
    paths = [p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(paths)


def ensure_16bit(image: np.ndarray, path: Path) -> None:
    """Validate that an image is 16-bit.

    Parameters
    ----------
    image:
        Image array.
    path:
        Image path for error context.

    Raises
    ------
    ValueError
        If the image is not 16-bit.
    """
    if image.dtype != np.uint16:
        raise ValueError(
            f"Expected 16-bit image at {path}, got dtype={image.dtype}. "
            "Ensure the source data is 16-bit."
        )


def read_image_16bit(path: Path) -> np.ndarray:
    """Read a 16-bit grayscale image.

    Parameters
    ----------
    path:
        Path to the image file.

    Returns
    -------
    numpy.ndarray
        2D uint16 array.
    """
    with Image.open(path) as img:
        image = np.array(img)
    if image.ndim != 2:
        raise ValueError(f"Expected grayscale image at {path}, got shape {image.shape}.")
    ensure_16bit(image, path)
    return image


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
