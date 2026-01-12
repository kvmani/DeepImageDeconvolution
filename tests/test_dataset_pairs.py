import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("PIL")

from pathlib import Path

from src.datasets.kikuchi_pairs import KikuchiPairDataset
from src.utils.io import write_image_16bit


def _write_sample(path: Path, array: np.ndarray) -> None:
    write_image_16bit(path, array)


def test_kikuchi_pair_dataset(tmp_path: Path) -> None:
    a_dir = tmp_path / "A"
    b_dir = tmp_path / "B"
    c_dir = tmp_path / "C"
    a_dir.mkdir()
    b_dir.mkdir()
    c_dir.mkdir()

    rng = np.random.default_rng(0)
    metadata_lines = ["sample_id,x,y,weight_a,weight_b"]
    for idx in range(3):
        a = (rng.random((32, 32)) * 65535).astype(np.uint16)
        b = (rng.random((32, 32)) * 65535).astype(np.uint16)
        x = 0.4 + (0.1 * idx)
        y = 1.0 - x
        c = ((x * a + y * b)).astype(np.uint16)
        _write_sample(a_dir / f"sample_{idx:03d}_A.png", a)
        _write_sample(b_dir / f"sample_{idx:03d}_B.png", b)
        _write_sample(c_dir / f"sample_{idx:03d}_C.png", c)
        metadata_lines.append(
            f"sample_{idx:03d},{x:.4f},{y:.4f},{x:.4f},{y:.4f}"
        )

    (tmp_path / "metadata.csv").write_text("\n".join(metadata_lines))

    dataset = KikuchiPairDataset(
        root_dir=tmp_path,
        a_dir="A",
        b_dir="B",
        c_dir="C",
        extensions=[".png"],
        preprocess_cfg={"normalize": {"enabled": False}},
    )

    sample = dataset[0]
    assert sample["C"].shape == (1, 32, 32)
    assert sample["A"].shape == (1, 32, 32)
    assert sample["B"].shape == (1, 32, 32)
    assert sample["x"].shape == (1,)
    assert 0.0 <= float(sample["x"].item()) <= 1.0
