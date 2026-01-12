"""Checkpoint save/load utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    step: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Save model, optimizer, and scheduler state."""
    payload: Dict[str, Any] = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
    }
    if step is not None:
        payload["step"] = step
    if config is not None:
        payload["config"] = config
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> Dict[str, Any]:
    """Load model (and optionally optimizer/scheduler) state."""
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    return checkpoint
