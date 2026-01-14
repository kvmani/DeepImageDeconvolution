"""Tests for logging utilities."""
from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image

from src.utils.logging import resolve_log_level, summarize_images, write_manifest


def _write_image(path: Path, size: tuple[int, int], dtype: str = "uint16") -> None:
    width, height = size
    if dtype == "uint16":
        array = (np.random.rand(height, width) * 65535).astype(np.uint16)
        Image.fromarray(array, mode="I;16").save(path)
    else:
        array = (np.random.rand(height, width) * 255).astype(np.uint8)
        Image.fromarray(array, mode="L").save(path)


def test_summarize_images(tmp_path: Path) -> None:
    img_a = tmp_path / "a.png"
    img_b = tmp_path / "b.png"
    _write_image(img_a, (32, 24), dtype="uint16")
    _write_image(img_b, (64, 40), dtype="uint8")

    summary = summarize_images([img_a, img_b], sample_n=5)
    assert summary["total"] == 2
    assert summary["extensions"].get(".png") == 2
    assert summary["min_size"] == {"width": 32, "height": 24}
    assert summary["max_size"] == {"width": 64, "height": 40}
    assert "uint16" in summary["sample_dtypes"]


def test_write_manifest(tmp_path: Path) -> None:
    data = {"name": "run", "count": 2}
    manifest_path = write_manifest(tmp_path, data)
    assert manifest_path.exists()
    loaded = json.loads(manifest_path.read_text())
    assert loaded == data


def test_resolve_log_level_quiet() -> None:
    assert resolve_log_level("DEBUG", debug=False, quiet=True) == 30
