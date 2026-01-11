import numpy as np

from src.preprocessing.mask import (
    apply_circular_mask,
    build_circular_mask,
    build_mask_with_metadata,
    detect_circular_mask,
)
from src.preprocessing.normalise import normalize_image


def test_detect_circular_mask_true() -> None:
    shape = (32, 32)
    mask = build_circular_mask(shape)
    image = np.zeros(shape, dtype=np.float32)
    image[mask] = 0.5
    detected, fraction = detect_circular_mask(
        image, mask, zero_tolerance=1e-6, outside_zero_fraction=0.95
    )
    assert detected is True
    assert fraction == 1.0


def test_detect_circular_mask_false() -> None:
    shape = (32, 32)
    mask = build_circular_mask(shape)
    image = np.ones(shape, dtype=np.float32)
    detected, fraction = detect_circular_mask(
        image, mask, zero_tolerance=1e-6, outside_zero_fraction=0.95
    )
    assert detected is False
    assert fraction < 0.95


def test_smart_normalize_uses_mask() -> None:
    shape = (32, 32)
    mask = build_circular_mask(shape)
    image = np.zeros(shape, dtype=np.float32)
    values = np.linspace(0.2, 0.8, num=mask.sum(), dtype=np.float32)
    image[mask] = values

    normalized = normalize_image(
        image,
        method="min_max",
        mask=mask,
        smart_minmax=True,
    )

    assert np.isclose(normalized[mask].min(), 0.0, atol=1e-5)
    assert np.isclose(normalized[mask].max(), 1.0, atol=1e-5)
    assert np.allclose(normalized[~mask], 0.0)


def test_apply_circular_mask() -> None:
    shape = (16, 16)
    mask = build_circular_mask(shape)
    image = np.ones(shape, dtype=np.float32)
    masked = apply_circular_mask(image, mask)
    assert np.allclose(masked[~mask], 0.0)
    assert np.allclose(masked[mask], 1.0)


def test_build_mask_with_metadata_detects_mask() -> None:
    shape = (16, 16)
    mask = build_circular_mask(shape)
    image = np.zeros(shape, dtype=np.float32)
    image[mask] = 0.5

    built_mask, meta = build_mask_with_metadata(
        image, detect_existing=True, zero_tolerance=1e-6, outside_zero_fraction=0.95
    )
    assert built_mask.shape == image.shape
    assert meta["already_masked"] is True
    assert meta["outside_zero_fraction"] == 1.0
