"""Optimizer and scheduler builders."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch import nn


def build_optimizer(model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    """Create an optimizer from configuration."""
    lr = float(cfg.get("learning_rate", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    optimizer_name = str(cfg.get("optimizer", "adam")).lower()
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        momentum = float(cfg.get("momentum", 0.9))
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
        )
    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Dict[str, Any],
    steps_per_epoch: int,
    epochs: int,
) -> Tuple[torch.optim.lr_scheduler._LRScheduler | None, bool]:
    """Create a scheduler and indicate if it should step per batch."""
    sched_cfg = cfg.get("scheduler", {})
    if sched_cfg.get("enabled") is False:
        return None, False

    name = sched_cfg.get("name") or sched_cfg.get("type")
    if not name:
        return None, False

    name = str(name).lower()
    if name == "one_cycle":
        max_lr = float(sched_cfg.get("max_lr", 1e-3))
        pct_start = float(sched_cfg.get("pct_start", 0.3))
        div_factor = float(sched_cfg.get("div_factor", 25.0))
        final_div_factor = float(sched_cfg.get("final_div_factor", 1e4))
        anneal_strategy = str(sched_cfg.get("anneal_strategy", "cos")).lower()
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=max(1, steps_per_epoch * epochs),
            pct_start=pct_start,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            anneal_strategy=anneal_strategy,
        )
        return scheduler, True

    if name == "step":
        step_size = int(sched_cfg.get("step_size", 25))
        gamma = float(sched_cfg.get("gamma", 0.5))
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )
        return scheduler, False

    if name == "cosine":
        t_max = int(sched_cfg.get("t_max", 50))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
        return scheduler, False

    raise ValueError(f"Unknown scheduler name: {name}")
