from pathlib import Path

import pytest

pytest.importorskip("PIL")

from src.generation.dataset import generate_synthetic_dataset


def test_generate_synthetic_dataset_debug(tmp_path: Path) -> None:
    input_dir = Path("data/code_development_data")
    if not input_dir.exists():
        pytest.skip("Sample data not available.")

    config = {
        "data": {
            "input_dir": str(input_dir),
            "input_recursive": True,
            "output_dir": str(tmp_path),
            "num_samples": 2,
            "allow_same": True,
            "preprocess": {
                "auto_crop_to_min": True,
                "crop": {"enabled": False},
                "denoise": {"enabled": False},
                "normalize": {"enabled": True, "method": "min_max"},
                "augment": {"apply_to": "B", "flip_horizontal": False, "flip_vertical": False, "rotate90": False},
            },
            "mix": {
                "pipeline": "normalize_then_mix",
                "normalize_after_mix": False,
                "normalize_method": "max",
                "weight": {"min": 0.3, "max": 0.7},
                "blur": {"enabled": False},
                "noise": {"enabled": False},
            },
        },
        "debug": {
            "enabled": True,
            "seed": 123,
            "visualize": False,
            "max_visualizations": 0,
            "sample_limit": 2,
            "sum_tolerance": 0.05,
        },
    }

    summary = generate_synthetic_dataset(config)
    assert summary["samples"] == 2
    assert (tmp_path / "A").exists()
    assert (tmp_path / "B").exists()
    assert (tmp_path / "C").exists()
