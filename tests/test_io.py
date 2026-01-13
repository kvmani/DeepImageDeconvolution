from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PIL")

from src.utils.io import read_image_16bit, read_image_float01_with_meta


def test_read_image_16bit() -> None:
    path = Path("data/code_development_data/image_A.png")
    if not path.exists():
        pytest.skip("Sample data not available.")
    image = read_image_16bit(path)
    assert image.dtype == np.uint16
    assert image.ndim == 2


def test_read_image_float01_with_meta_16bit(tmp_path: Path) -> None:
    from PIL import Image

    array = (np.arange(16, dtype=np.uint16).reshape(4, 4))
    path = tmp_path / "test.tif"
    Image.fromarray(array, mode="I;16").save(path)

    image, meta = read_image_float01_with_meta(path)
    assert image.dtype == np.float32
    assert 0.0 <= float(image.min()) <= float(image.max()) <= 1.0
    assert meta["is_16bit"] is True


def test_read_image_float01_with_meta_non16bit(tmp_path: Path) -> None:
    from PIL import Image

    array = (np.arange(16, dtype=np.uint8).reshape(4, 4))
    path = tmp_path / "test.png"
    Image.fromarray(array, mode="L").save(path)

    image, meta = read_image_float01_with_meta(path, allow_non_16bit=True)
    assert image.dtype == np.float32
    assert 0.0 <= float(image.min()) <= float(image.max()) <= 1.0
    assert meta["is_16bit"] is False


def test_read_image_16bit_scales_uint8(tmp_path: Path) -> None:
    from PIL import Image

    array = np.array([[0, 128, 255]], dtype=np.uint8)
    path = tmp_path / "test.png"
    Image.fromarray(array, mode="L").save(path)

    image = read_image_16bit(path)
    assert image.dtype == np.uint16
    assert image[0, 0] == 0
    assert image[0, 2] == 65535
