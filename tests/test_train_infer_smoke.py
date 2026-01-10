import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("PIL")

from pathlib import Path

from src.training.train import train_model
from src.inference.infer import run_inference
from src.utils.io import write_image_16bit


def _write_sample(path: Path, array: np.ndarray) -> None:
    write_image_16bit(path, array)


def test_train_and_infer_smoke(tmp_path: Path) -> None:
    a_dir = tmp_path / "data" / "A"
    b_dir = tmp_path / "data" / "B"
    c_dir = tmp_path / "data" / "C"
    a_dir.mkdir(parents=True)
    b_dir.mkdir(parents=True)
    c_dir.mkdir(parents=True)

    rng = np.random.default_rng(123)
    for idx in range(2):
        a = (rng.random((16, 16)) * 65535).astype(np.uint16)
        b = (rng.random((16, 16)) * 65535).astype(np.uint16)
        c = ((0.5 * a + 0.5 * b)).astype(np.uint16)
        _write_sample(a_dir / f"sample_{idx:03d}_A.png", a)
        _write_sample(b_dir / f"sample_{idx:03d}_B.png", b)
        _write_sample(c_dir / f"sample_{idx:03d}_C.png", c)

    train_out = tmp_path / "train_out"
    train_config = {
        "data": {
            "root_dir": str(tmp_path / "data"),
            "a_dir": "A",
            "b_dir": "B",
            "c_dir": "C",
            "extensions": [".png"],
            "val_split": 0.0,
            "shuffle": True,
            "num_workers": 0,
            "preprocess": {"normalize": {"enabled": False}},
        },
        "model": {
            "name": "unet_dual",
            "in_channels": 1,
            "out_channels": 1,
            "base_channels": 4,
            "depth": 2,
            "up_mode": "transpose",
            "use_batchnorm": False,
            "dropout": 0.0,
            "init": "kaiming",
        },
        "train": {
            "batch_size": 1,
            "epochs": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "optimizer": "adam",
            "scheduler": {"enabled": False},
            "loss": {"lambda_a": 1.0, "lambda_b": 1.0, "lambda_sum": 0.5},
            "grad_clip": 0.0,
        },
        "logging": {"log_to_file": False, "log_level": "ERROR", "log_interval": 1},
        "output": {"out_dir": str(train_out), "save_best": True, "save_every": 1},
        "debug": {"enabled": True, "seed": 1, "sample_limit": 2, "epochs": 1, "batch_size": 1},
    }

    train_model(train_config)
    checkpoint = train_out / "best.pt"
    assert checkpoint.exists()

    infer_out = tmp_path / "infer_out"
    infer_config = {
        "data": {
            "mixed_dir": str(c_dir),
            "extensions": [".png"],
            "num_workers": 0,
            "preprocess": {"normalize": {"enabled": False}},
        },
        "model": train_config["model"],
        "inference": {
            "checkpoint": str(checkpoint),
            "batch_size": 1,
            "clamp_outputs": True,
        },
        "postprocess": {"apply_mask": False},
        "output": {"out_dir": str(infer_out), "format": "png", "save_sum": True},
        "logging": {"log_to_file": False, "log_level": "ERROR"},
        "debug": {"enabled": True, "seed": 1, "sample_limit": 2},
    }

    run_inference(infer_config)
    assert (infer_out / "A").exists()
    assert (infer_out / "B").exists()
