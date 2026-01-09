import numpy as np

from src.generation.mix import mix_normalize_then_mix, mix_then_normalize


def test_mix_normalize_then_mix() -> None:
    a = np.full((2, 2), 0.2, dtype=np.float32)
    b = np.full((2, 2), 0.8, dtype=np.float32)
    mixed = mix_normalize_then_mix(a, b, weight_a=0.25, normalize_after_mix=False, normalize_method="max")
    expected = 0.25 * a + 0.75 * b
    assert np.allclose(mixed, expected)


def test_mix_then_normalize_outputs_range() -> None:
    a = np.array([[0.0, 0.5], [0.2, 0.8]], dtype=np.float32)
    b = np.array([[0.1, 0.6], [0.3, 0.9]], dtype=np.float32)
    mixed = mix_then_normalize(a, b, weight_a=0.5, normalize_method="min_max")
    assert 0.0 <= mixed.min() <= 1.0
    assert 0.0 <= mixed.max() <= 1.0
