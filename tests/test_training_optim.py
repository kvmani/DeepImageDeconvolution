import pytest

pytest.importorskip("torch")

import torch

from src.training.optim import build_optimizer, build_scheduler


def test_build_one_cycle_scheduler() -> None:
    model = torch.nn.Conv2d(1, 1, kernel_size=1)
    train_cfg = {
        "learning_rate": 1e-3,
        "optimizer": "adam",
        "scheduler": {
            "name": "one_cycle",
            "max_lr": 2e-4,
            "pct_start": 0.35,
            "div_factor": 20,
            "final_div_factor": 1e4,
            "anneal_strategy": "cos",
        },
    }
    optimizer = build_optimizer(model, train_cfg)
    scheduler, step_per_batch = build_scheduler(optimizer, train_cfg, steps_per_epoch=5, epochs=2)
    assert scheduler is not None
    assert step_per_batch is True
    for _ in range(3):
        optimizer.step()
        scheduler.step()
