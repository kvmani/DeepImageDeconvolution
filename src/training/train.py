"""Training loop for dual-output U-Net models."""
from __future__ import annotations

import json
import math
import random
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.datasets import KikuchiPairDataset, split_dataset
from src.models import build_model
from src.utils.logging import add_file_handler, get_logger
from src.utils.metrics import aggregate_metrics


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _create_optimizer(model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    lr = float(cfg.get("learning_rate", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    optimizer_name = str(cfg.get("optimizer", "adam")).lower()
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        momentum = float(cfg.get("momentum", 0.9))
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def _create_scheduler(optimizer: torch.optim.Optimizer, cfg: Dict[str, Any]) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    sched_cfg = cfg.get("scheduler", {})
    if not sched_cfg.get("enabled", False):
        return None
    sched_type = str(sched_cfg.get("type", "step")).lower()
    if sched_type == "step":
        step_size = int(sched_cfg.get("step_size", 25))
        gamma = float(sched_cfg.get("gamma", 0.5))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if sched_type == "cosine":
        t_max = int(sched_cfg.get("t_max", 50))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    raise ValueError(f"Unknown scheduler type: {sched_type}")


def _log_shapes(logger, c: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> None:
    logger.info(
        "Input/target shapes: C=%s, A=%s, B=%s",
        tuple(c.shape),
        tuple(a.shape),
        tuple(b.shape),
    )


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
) -> None:
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(payload, path)


def train_model(config: Dict[str, Any]) -> None:
    """Train the model using the provided configuration."""
    log_level_name = str(config.get("logging", {}).get("log_level", "INFO")).upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    logger = get_logger(__name__, level=log_level)

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})
    output_cfg = config.get("output", {})
    logging_cfg = config.get("logging", {})
    debug_cfg = config.get("debug", {})

    debug_enabled = bool(debug_cfg.get("enabled", False))
    seed = int(debug_cfg.get("seed", 42))

    out_dir = Path(output_cfg.get("out_dir", "outputs/train_run"))
    out_dir.mkdir(parents=True, exist_ok=True)

    log_to_file = bool(logging_cfg.get("log_to_file", True))
    if log_to_file:
        add_file_handler(out_dir / "output.log")

    _set_seed(seed)

    config_path = out_dir / "config_used.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    root_dir = Path(data_cfg.get("root_dir", "data/synthetic"))
    a_dir = str(data_cfg.get("a_dir", "A"))
    b_dir = str(data_cfg.get("b_dir", "B"))
    c_dir = str(data_cfg.get("c_dir", "C"))
    extensions = data_cfg.get("extensions", [".png", ".tif", ".tiff"])
    preprocess_cfg = data_cfg.get("preprocess", {})
    val_split = float(data_cfg.get("val_split", 0.1))
    num_workers = int(data_cfg.get("num_workers", 0))

    sample_limit = debug_cfg.get("sample_limit") if debug_enabled else None
    dataset = KikuchiPairDataset(
        root_dir=root_dir,
        a_dir=a_dir,
        b_dir=b_dir,
        c_dir=c_dir,
        extensions=extensions,
        preprocess_cfg=preprocess_cfg,
        seed=seed,
        limit=sample_limit,
    )
    train_set, val_set = split_dataset(dataset, val_split, seed)

    batch_size = int(train_cfg.get("batch_size", 4))
    epochs = int(train_cfg.get("epochs", 1))
    log_interval = int(logging_cfg.get("log_interval", 25))

    if debug_enabled:
        batch_size = int(debug_cfg.get("batch_size", batch_size))
        epochs = int(debug_cfg.get("epochs", epochs))
        log_interval = int(debug_cfg.get("log_interval", log_interval))
        num_workers = 0

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=bool(data_cfg.get("shuffle", True)),
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    model = build_model(model_cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %d", param_count)
    optimizer = _create_optimizer(model, train_cfg)
    scheduler = _create_scheduler(optimizer, train_cfg)

    loss_weights = train_cfg.get("loss", {})
    lambda_a = float(loss_weights.get("lambda_a", 1.0))
    lambda_b = float(loss_weights.get("lambda_b", 1.0))
    lambda_sum = float(loss_weights.get("lambda_sum", 0.5))

    loss_fn = nn.L1Loss()
    grad_clip = float(train_cfg.get("grad_clip", 0.0))

    best_val = math.inf
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(train_loader, start=1):
            c = batch["C"].to(device)
            a = batch["A"].to(device)
            b = batch["B"].to(device)

            if epoch == 1 and batch_idx == 1:
                _log_shapes(logger, c, a, b)

            optimizer.zero_grad()
            pred_a, pred_b = model(c)
            loss_a = loss_fn(pred_a, a)
            loss_b = loss_fn(pred_b, b)
            loss_sum = loss_fn(pred_a + pred_b, c)
            loss = lambda_a * loss_a + lambda_b * loss_b + lambda_sum * loss_sum
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_loss += float(loss.item())
            if batch_idx % log_interval == 0:
                logger.info(
                    "Epoch %d/%d | Batch %d/%d | Loss %.6f",
                    epoch,
                    epochs,
                    batch_idx,
                    len(train_loader),
                    loss.item(),
                )

        epoch_loss /= max(len(train_loader), 1)
        metrics = {"train_loss": epoch_loss}

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            psnr_a = 0.0
            ssim_a = 0.0
            psnr_b = 0.0
            ssim_b = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    c = batch["C"].to(device)
                    a = batch["A"].to(device)
                    b = batch["B"].to(device)
                    pred_a, pred_b = model(c)
                    loss_a = loss_fn(pred_a, a)
                    loss_b = loss_fn(pred_b, b)
                    loss_sum = loss_fn(pred_a + pred_b, c)
                    loss = lambda_a * loss_a + lambda_b * loss_b + lambda_sum * loss_sum
                    val_loss += float(loss.item())

                    batch_psnr_a, batch_ssim_a = aggregate_metrics(pred_a, a)
                    batch_psnr_b, batch_ssim_b = aggregate_metrics(pred_b, b)
                    psnr_a += batch_psnr_a
                    ssim_a += batch_ssim_a
                    psnr_b += batch_psnr_b
                    ssim_b += batch_ssim_b

            val_loss /= max(len(val_loader), 1)
            psnr_a /= max(len(val_loader), 1)
            ssim_a /= max(len(val_loader), 1)
            psnr_b /= max(len(val_loader), 1)
            ssim_b /= max(len(val_loader), 1)

            metrics.update(
                {
                    "val_loss": val_loss,
                    "psnr_a": psnr_a,
                    "ssim_a": ssim_a,
                    "psnr_b": psnr_b,
                    "ssim_b": ssim_b,
                }
            )
            logger.info(
                "Epoch %d/%d | Train %.6f | Val %.6f | PSNR A/B %.3f/%.3f | SSIM A/B %.3f/%.3f",
                epoch,
                epochs,
                epoch_loss,
                val_loss,
                psnr_a,
                psnr_b,
                ssim_a,
                ssim_b,
            )
        else:
            logger.info("Epoch %d/%d | Train %.6f", epoch, epochs, epoch_loss)

        history.append({"epoch": epoch, **metrics})

        if scheduler is not None:
            scheduler.step()

        last_path = out_dir / "last.pt"
        _save_checkpoint(last_path, model, optimizer, epoch, metrics)

        if output_cfg.get("save_every", 1) and epoch % int(output_cfg.get("save_every", 1)) == 0:
            checkpoint_path = out_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            _save_checkpoint(checkpoint_path, model, optimizer, epoch, metrics)

        if output_cfg.get("save_best", True):
            current_val = metrics.get("val_loss", metrics["train_loss"])
            if current_val < best_val:
                best_val = current_val
                best_path = out_dir / "best.pt"
                _save_checkpoint(best_path, model, optimizer, epoch, metrics)

    history_path = out_dir / "history.json"
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
