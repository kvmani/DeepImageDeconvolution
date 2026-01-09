import numpy as np

from src.preprocessing.normalise import histogram_equalization, normalize_image


def test_normalize_constant_image() -> None:
    image = np.ones((4, 4), dtype=np.float32)
    normalized = normalize_image(image, method="min_max")
    assert normalized.min() == 0.0
    assert normalized.max() == 0.0


def test_histogram_equalization_range() -> None:
    rng = np.random.default_rng(123)
    image = rng.random((8, 8), dtype=np.float32)
    equalized = histogram_equalization(image, bins=256)
    assert 0.0 <= equalized.min() <= 1.0
    assert 0.0 <= equalized.max() <= 1.0
