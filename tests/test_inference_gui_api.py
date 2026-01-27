import importlib

import numpy as np
import pytest
import torch

from src.inference.api import merge_inference_config
from src.inference.metrics import compute_image_metrics
from src.inference.model_manager import ModelManager


def test_merge_inference_config_precedence() -> None:
    base = {"data": {"mixed_dir": "base"}, "inference": {"batch_size": 4}}
    gui = {"data": {"mixed_dir": "gui"}, "inference": {"batch_size": 8}}
    cli = {"data": {"mixed_dir": "cli"}}
    merged = merge_inference_config(base, gui_overrides=gui, cli_overrides=cli)
    assert merged["data"]["mixed_dir"] == "cli"
    assert merged["inference"]["batch_size"] == 8


def test_model_manager_caches_by_checkpoint(tmp_path) -> None:
    model_factory = lambda cfg: torch.nn.Conv2d(1, 1, kernel_size=1, bias=False)
    manager = ModelManager(model_factory=model_factory)
    model = model_factory({})
    torch.save({"model_state": model.state_dict()}, tmp_path / "ckpt.pt")

    cfg = {"name": "tiny"}
    first = manager.get_model(cfg, tmp_path / "ckpt.pt", torch.device("cpu"))
    second = manager.get_model(cfg, tmp_path / "ckpt.pt", torch.device("cpu"))
    assert first is second

    new_model = model_factory({})
    torch.nn.init.constant_(new_model.weight, 0.5)
    torch.save({"model_state": new_model.state_dict()}, tmp_path / "ckpt2.pt")
    third = manager.get_model(cfg, tmp_path / "ckpt2.pt", torch.device("cpu"))
    assert third is not first


def test_compute_image_metrics() -> None:
    pred = np.zeros((16, 16), dtype=np.float32)
    target = np.ones((16, 16), dtype=np.float32)
    metrics = compute_image_metrics(pred, target)
    assert metrics["l1"] == pytest.approx(1.0)
    assert metrics["l2"] == pytest.approx(1.0)
    assert metrics["psnr"] == pytest.approx(0.0, abs=1e-3)

    metrics_same = compute_image_metrics(pred, pred)
    assert metrics_same["l1"] == pytest.approx(0.0)
    assert metrics_same["l2"] == pytest.approx(0.0)
    assert metrics_same["psnr"] > 70.0
    assert metrics_same["ssim"] >= 0.9


def test_gui_import_no_side_effects() -> None:
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    assert QApplication.instance() is None
    module = importlib.import_module("scripts.run_inference_gui")
    assert QApplication.instance() is None
    assert hasattr(module, "main")
