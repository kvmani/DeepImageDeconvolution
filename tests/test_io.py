from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PIL")

from src.utils.io import read_image_16bit


def test_read_image_16bit() -> None:
    path = Path("data/code_development_data/image_A.png")
    if not path.exists():
        pytest.skip("Sample data not available.")
    image = read_image_16bit(path)
    assert image.dtype == np.uint16
    assert image.ndim == 2
