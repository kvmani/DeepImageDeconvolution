"""Training and evaluation loops."""
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn
import numpy as np

from src.training.metrics import compute_reconstruction_metrics
from src.preprocessing.mask import build_circular_mask


def _log_shapes(logger, c: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> None:
    logger.info(
        "Input/target shapes: C=%s, A=%s, B=%s",
        tuple(c.shape),
        tuple(a.shape),
        tuple(b.shape),
    )


def _validate_batch(logger, c: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> None:
    if c.shape != a.shape or c.shape != b.shape:
        raise ValueError(f"Shape mismatch: C={tuple(c.shape)} A={tuple(a.shape)} B={tuple(b.shape)}")
    for name, tensor in (("C", c), ("A", a), ("B", b)):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite values detected in {name} batch.")
        min_val = float(tensor.min().item())
        max_val = float(tensor.max().item())
        if min_val < -1e-3 or max_val > 1.0 + 1e-3:
            logger.warning("%s values outside [0, 1]: min=%.4f max=%.4f", name, min_val, max_val)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    weight_loss_fn: nn.Module,
    loss_weights: Dict[str, float],
    device: torch.device,
    logger,
    log_interval: int = 25,
    grad_clip: float = 0.0,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scheduler_step_per_batch: bool = False,
) -> Dict[str, float]:
    """Run a single training epoch."""
    model.train()
    epoch_loss = 0.0
    epoch_l_ab = 0.0
    epoch_l_recon = 0.0
    epoch_l_x = 0.0
    lambda_ab = loss_weights.get("lambda_ab", 1.0)
    lambda_recon = loss_weights.get("lambda_recon", 0.5)
    lambda_x = loss_weights.get("lambda_x", 0.1)
    x_min = float("inf")
    x_max = float("-inf")
    x_sum = 0.0
    x_sum_sq = 0.0
    x_count = 0

    for batch_idx, batch in enumerate(loader, start=1):
        c = batch["C"].to(device)
        a = batch["A"].to(device)
        b = batch["B"].to(device)
        x_true = batch["x"].to(device)

        if batch_idx == 1:
            _log_shapes(logger, c, a, b)
            _validate_batch(logger, c, a, b)

        optimizer.zero_grad()
        pred_a, pred_b, x_hat = model(c)
        x_weight = x_hat.view(-1, 1, 1, 1)
        recon = (x_weight * pred_a) + ((1.0 - x_weight) * pred_b)
        loss_ab = loss_fn(pred_a, a) + loss_fn(pred_b, b)
        loss_recon = loss_fn(recon, c)
        loss_x = weight_loss_fn(x_hat.squeeze(-1), x_true.squeeze(-1))
        loss = (lambda_ab * loss_ab) + (lambda_recon * loss_recon) + (lambda_x * loss_x)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None and scheduler_step_per_batch:
            scheduler.step()

        epoch_loss += float(loss.item())
        epoch_l_ab += float(loss_ab.item())
        epoch_l_recon += float(loss_recon.item())
        epoch_l_x += float(loss_x.item())
        x_vals = x_hat.detach().view(-1)
        x_min = min(x_min, float(x_vals.min().item()))
        x_max = max(x_max, float(x_vals.max().item()))
        x_sum += float(x_vals.sum().item())
        x_sum_sq += float((x_vals ** 2).sum().item())
        x_count += int(x_vals.numel())
        if batch_idx % log_interval == 0:
            current_lr = optimizer.param_groups[0].get("lr", 0.0)
            logger.info(
                "Batch %d/%d | Loss %.6f | L_ab %.6f | L_recon %.6f | L_x %.6f | x_hat [%.3f, %.3f] | LR %.6e",
                batch_idx,
                len(loader),
                loss.item(),
                loss_ab.item(),
                loss_recon.item(),
                loss_x.item(),
                float(x_vals.min().item()),
                float(x_vals.max().item()),
                current_lr,
            )

    denom = max(len(loader), 1)
    epoch_loss /= denom
    epoch_l_ab /= denom
    epoch_l_recon /= denom
    epoch_l_x /= denom
    x_mean = x_sum / max(x_count, 1)
    x_var = (x_sum_sq / max(x_count, 1)) - (x_mean**2)
    x_std = float(max(x_var, 0.0) ** 0.5)
    return {
        "train_loss": epoch_loss,
        "train_l_ab": epoch_l_ab,
        "train_l_recon": epoch_l_recon,
        "train_l_x": epoch_l_x,
        "x_hat_mean": float(x_mean),
        "x_hat_std": x_std,
        "x_hat_min": float(x_min if x_count else 0.0),
        "x_hat_max": float(x_max if x_count else 0.0),
    }


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    weight_loss_fn: nn.Module,
    loss_weights: Dict[str, float],
    device: torch.device,
    mask_enabled: bool = False,
) -> Dict[str, float]:
    """Run evaluation on a validation set."""
    model.eval()
    lambda_ab = loss_weights.get("lambda_ab", 1.0)
    lambda_recon = loss_weights.get("lambda_recon", 0.5)
    lambda_x = loss_weights.get("lambda_x", 0.1)

    metrics_accum: Dict[str, float] = {}
    val_loss = 0.0
    val_l_ab = 0.0
    val_l_recon = 0.0
    val_l_x = 0.0

    with torch.no_grad():
        for batch in loader:
            c = batch["C"].to(device)
            a = batch["A"].to(device)
            b = batch["B"].to(device)
            x_true = batch["x"].to(device)
            mask = None
            if mask_enabled:
                mask_np = build_circular_mask(c.shape[-2:])
                mask = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                mask = mask.to(device)
            pred_a, pred_b, x_hat = model(c)
            x_weight = x_hat.view(-1, 1, 1, 1)
            recon = (x_weight * pred_a) + ((1.0 - x_weight) * pred_b)
            loss_ab = loss_fn(pred_a, a) + loss_fn(pred_b, b)
            loss_recon = loss_fn(recon, c)
            loss_x = weight_loss_fn(x_hat.squeeze(-1), x_true.squeeze(-1))
            loss = (lambda_ab * loss_ab) + (lambda_recon * loss_recon) + (lambda_x * loss_x)
            val_loss += float(loss.item())
            val_l_ab += float(loss_ab.item())
            val_l_recon += float(loss_recon.item())
            val_l_x += float(loss_x.item())

            batch_metrics = compute_reconstruction_metrics(
                pred_a,
                pred_b,
                x_hat,
                a,
                b,
                c,
                x_true=x_true,
                mask=mask,
            )
            for key, value in batch_metrics.items():
                metrics_accum[key] = metrics_accum.get(key, 0.0) + float(value)

    denom = max(len(loader), 1)
    val_loss /= denom
    val_l_ab /= denom
    val_l_recon /= denom
    val_l_x /= denom
    metrics = {
        "val_loss": val_loss,
        "val_l_ab": val_l_ab,
        "val_l_recon": val_l_recon,
        "val_l_x": val_l_x,
    }
    for key, value in metrics_accum.items():
        metrics[key] = value / denom
    return metrics
