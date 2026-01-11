"""Training loop for dual-output U-Net models."""
from __future__ import annotations

import json
import math
import random
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.datasets import KikuchiPairDataset, split_dataset
from src.models import build_model
from src.utils.logging import add_file_handler, get_logger
from src.utils.io import write_image_8bit
from src.utils.metrics import aggregate_metrics, mse
from src.utils.reporting import update_image_log, write_image_log_html


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


@dataclass
class ImageLogConfig:
    enabled: bool
    interval: int
    max_samples: int
    sample_strategy: str
    sample_ids: List[str]
    split: str
    output_dir: Path
    image_format: str
    write_html: bool
    include_sum: bool
    seed: int


@dataclass
class ImageLogState:
    indices: Optional[List[int]] = None
    rng: Optional[np.random.Generator] = None


def _parse_image_log_config(logging_cfg: Dict[str, Any], out_dir: Path, seed: int) -> ImageLogConfig:
    cfg = logging_cfg.get("image_log", {})
    enabled = bool(cfg.get("enabled", False))
    interval = max(1, int(cfg.get("interval", 5)))
    max_samples = max(1, int(cfg.get("max_samples", 4)))
    sample_strategy = str(cfg.get("sample_strategy", "fixed")).lower()
    if sample_strategy not in ("fixed", "first", "random"):
        sample_strategy = "fixed"
    sample_ids = list(cfg.get("sample_ids", []))
    split = str(cfg.get("split", "val")).lower()
    if split not in ("val", "train"):
        split = "val"
    output_dir = cfg.get("output_dir")
    if not output_dir:
        output_path = out_dir / "monitoring"
    else:
        output_path = Path(output_dir)
        if not output_path.is_absolute():
            output_path = out_dir / output_path
    image_format = str(cfg.get("image_format", "png")).lower()
    if image_format not in ("png", "jpg", "jpeg"):
        image_format = "png"
    write_html = bool(cfg.get("write_html", True))
    include_sum = bool(cfg.get("include_sum", True))
    log_seed = int(cfg.get("seed", seed))
    return ImageLogConfig(
        enabled=enabled,
        interval=interval,
        max_samples=max_samples,
        sample_strategy=sample_strategy,
        sample_ids=sample_ids,
        split=split,
        output_dir=output_path,
        image_format=image_format,
        write_html=write_html,
        include_sum=include_sum,
        seed=log_seed,
    )


def _sanitize_sample_id(sample_id: str) -> str:
    return sample_id.replace("/", "_").replace("\\", "_")


def _select_log_indices(
    dataset,
    cfg: ImageLogConfig,
    state: ImageLogState,
    logger,
) -> List[int]:
    if len(dataset) == 0:
        return []
    if state.rng is None:
        state.rng = np.random.default_rng(cfg.seed)

    if cfg.sample_ids:
        if state.indices is None:
            selected: List[int] = []
            for idx in range(len(dataset)):
                sample_id = dataset[idx]["sample_id"]
                if sample_id in cfg.sample_ids:
                    selected.append(idx)
                    if len(selected) >= cfg.max_samples:
                        break
            if not selected:
                logger.warning("No matching sample_ids found for image logging.")
            state.indices = selected
        return state.indices

    strategy = cfg.sample_strategy
    if strategy in ("fixed", "first"):
        if state.indices is None:
            count = min(cfg.max_samples, len(dataset))
            if strategy == "first":
                state.indices = list(range(count))
            else:
                state.indices = state.rng.choice(len(dataset), size=count, replace=False).tolist()
        return state.indices

    count = min(cfg.max_samples, len(dataset))
    return state.rng.choice(len(dataset), size=count, replace=False).tolist()


def _tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().numpy().squeeze()
    return np.clip(array, 0.0, 1.0)


def _log_epoch_images(
    model: nn.Module,
    dataset,
    cfg: ImageLogConfig,
    state: ImageLogState,
    device: torch.device,
    epoch: int,
    logger,
    history: List[Dict[str, Any]] | None = None,
) -> None:
    indices = _select_log_indices(dataset, cfg, state, logger)
    if not indices:
        return

    epoch_dir = cfg.output_dir / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    samples_log: List[Dict[str, Any]] = []
    model.eval()

    with torch.no_grad():
        for idx in indices:
            sample = dataset[idx]
            sample_id = _sanitize_sample_id(sample["sample_id"])
            sample_dir = epoch_dir / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            c = sample["C"].unsqueeze(0).to(device)
            a = sample["A"].unsqueeze(0).to(device)
            b = sample["B"].unsqueeze(0).to(device)

            pred_a, pred_b = model(c)
            pred_a = torch.clamp(pred_a, 0.0, 1.0)
            pred_b = torch.clamp(pred_b, 0.0, 1.0)
            pred_sum = torch.clamp(pred_a + pred_b, 0.0, 1.0)

            metrics = {
                "l1_a": float(F.l1_loss(pred_a, a).item()),
                "l2_a": float(mse(pred_a, a).mean().item()),
                "psnr_a": float(aggregate_metrics(pred_a, a)[0]),
                "ssim_a": float(aggregate_metrics(pred_a, a)[1]),
                "l1_b": float(F.l1_loss(pred_b, b).item()),
                "l2_b": float(mse(pred_b, b).mean().item()),
                "psnr_b": float(aggregate_metrics(pred_b, b)[0]),
                "ssim_b": float(aggregate_metrics(pred_b, b)[1]),
            }
            if cfg.include_sum:
                metrics.update(
                    {
                        "l1_sum": float(F.l1_loss(pred_sum, c).item()),
                        "l2_sum": float(mse(pred_sum, c).mean().item()),
                        "psnr_sum": float(aggregate_metrics(pred_sum, c)[0]),
                        "ssim_sum": float(aggregate_metrics(pred_sum, c)[1]),
                    }
                )

            ext = cfg.image_format
            images = {
                "C": f"{epoch_dir.name}/{sample_id}/C.{ext}",
                "A_gt": f"{epoch_dir.name}/{sample_id}/A_gt.{ext}",
                "B_gt": f"{epoch_dir.name}/{sample_id}/B_gt.{ext}",
                "A_pred": f"{epoch_dir.name}/{sample_id}/A_pred.{ext}",
                "B_pred": f"{epoch_dir.name}/{sample_id}/B_pred.{ext}",
            }
            if cfg.include_sum:
                images["C_sum"] = f"{epoch_dir.name}/{sample_id}/C_sum.{ext}"

            write_image_8bit(sample_dir / f"C.{ext}", _tensor_to_image(c))
            write_image_8bit(sample_dir / f"A_gt.{ext}", _tensor_to_image(a))
            write_image_8bit(sample_dir / f"B_gt.{ext}", _tensor_to_image(b))
            write_image_8bit(sample_dir / f"A_pred.{ext}", _tensor_to_image(pred_a))
            write_image_8bit(sample_dir / f"B_pred.{ext}", _tensor_to_image(pred_b))
            if cfg.include_sum:
                write_image_8bit(sample_dir / f"C_sum.{ext}", _tensor_to_image(pred_sum))

            samples_log.append(
                {
                    "sample_id": sample_id,
                    "images": images,
                    "metrics": metrics,
                }
            )

    entry = {"epoch": epoch, "split": cfg.split, "samples": samples_log}
    entries = update_image_log(cfg.output_dir, entry)
    if cfg.write_html:
        write_image_log_html(cfg.output_dir, entries, history=history)


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

    if batch_size <= 0:
        raise ValueError("train.batch_size must be > 0.")
    if epochs <= 0:
        raise ValueError("train.epochs must be > 0.")
    if log_interval <= 0:
        raise ValueError("logging.log_interval must be > 0.")

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

    image_log_cfg = _parse_image_log_config(logging_cfg, out_dir, seed)
    image_log_state = ImageLogState()
    if image_log_cfg.enabled:
        image_log_cfg.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Image logging enabled: every %d epochs, %d samples (%s split)",
            image_log_cfg.interval,
            image_log_cfg.max_samples,
            image_log_cfg.split,
        )

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(train_loader, start=1):
            c = batch["C"].to(device)
            a = batch["A"].to(device)
            b = batch["B"].to(device)

            if epoch == 1 and batch_idx == 1:
                _log_shapes(logger, c, a, b)
                _validate_batch(logger, c, a, b)

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
            l2_a = 0.0
            l2_b = 0.0
            l2_sum = 0.0
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
                    l2_a += float(mse(pred_a, a).mean().item())
                    l2_b += float(mse(pred_b, b).mean().item())
                    l2_sum += float(mse(pred_a + pred_b, c).mean().item())

            val_loss /= max(len(val_loader), 1)
            psnr_a /= max(len(val_loader), 1)
            ssim_a /= max(len(val_loader), 1)
            psnr_b /= max(len(val_loader), 1)
            ssim_b /= max(len(val_loader), 1)
            l2_a /= max(len(val_loader), 1)
            l2_b /= max(len(val_loader), 1)
            l2_sum /= max(len(val_loader), 1)

            metrics.update(
                {
                    "val_loss": val_loss,
                    "psnr_a": psnr_a,
                    "ssim_a": ssim_a,
                    "psnr_b": psnr_b,
                    "ssim_b": ssim_b,
                    "l2_a": l2_a,
                    "l2_b": l2_b,
                    "l2_sum": l2_sum,
                }
            )
            logger.info(
                "Epoch %d/%d | Train %.6f | Val %.6f | PSNR A/B %.3f/%.3f | SSIM A/B %.3f/%.3f | L2 A/B %.6f/%.6f",
                epoch,
                epochs,
                epoch_loss,
                val_loss,
                psnr_a,
                psnr_b,
                ssim_a,
                ssim_b,
                l2_a,
                l2_b,
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

        if image_log_cfg.enabled and epoch % image_log_cfg.interval == 0:
            log_dataset = train_set
            if image_log_cfg.split == "val" and val_set is not None:
                log_dataset = val_set
            elif image_log_cfg.split == "val" and val_set is None:
                logger.warning("Image logging requested for val split, but no val set exists; using train.")
            try:
                _log_epoch_images(
                    model,
                    log_dataset,
                    image_log_cfg,
                    image_log_state,
                    device,
                    epoch,
                    logger,
                    history,
                )
            except Exception as exc:
                logger.error("Image logging failed: %s", exc)

    history_path = out_dir / "history.json"
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
