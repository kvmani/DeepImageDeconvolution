"""Training loop for dual-output U-Net models with weight prediction."""
from __future__ import annotations

import json
import random
import logging
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.datasets import KikuchiPairDataset, split_dataset
from src.models import build_model
from src.preprocessing.mask import build_circular_mask
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.engine import evaluate, train_one_epoch
from src.training.metrics import compute_reconstruction_metrics
from src.training.optim import build_optimizer, build_scheduler
from src.utils.logging import add_file_handler, get_logger
from src.utils.io import write_image_8bit
from src.utils.metrics import aggregate_metrics
from src.utils.reporting import update_image_log, write_image_log_html


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    include_recon: bool
    mask_metrics: bool
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
    include_recon = bool(cfg.get("include_recon", cfg.get("include_sum", True)))
    mask_metrics = bool(cfg.get("mask_metrics", True))
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
        include_recon=include_recon,
        mask_metrics=mask_metrics,
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
            mask = None
            if cfg.mask_metrics:
                mask_np = build_circular_mask(c.shape[-2:])
                mask = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                mask = mask.to(device)

            pred_a, pred_b, x_hat = model(c)
            pred_a = torch.clamp(pred_a, 0.0, 1.0)
            pred_b = torch.clamp(pred_b, 0.0, 1.0)
            x_weight = x_hat.view(-1, 1, 1, 1)
            pred_recon = torch.clamp((x_weight * pred_a) + ((1.0 - x_weight) * pred_b), 0.0, 1.0)

            metrics = compute_reconstruction_metrics(pred_a, pred_b, x_hat, a, b, c, mask=mask)
            metrics.update(
                {
                    "l1_a": float(F.l1_loss(pred_a, a).item()),
                    "l1_b": float(F.l1_loss(pred_b, b).item()),
                    "x_hat": float(x_hat.item()),
                    "y_hat": float((1.0 - x_hat).item()),
                }
            )
            if cfg.include_recon:
                psnr_recon, ssim_recon = aggregate_metrics(pred_recon, c)
                metrics.update(
                    {
                        "l1_recon": float(F.l1_loss(pred_recon, c).item()),
                        "psnr_recon": float(psnr_recon),
                        "ssim_recon": float(ssim_recon),
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
            if cfg.include_recon:
                images["C_hat"] = f"{epoch_dir.name}/{sample_id}/C_hat.{ext}"

            write_image_8bit(sample_dir / f"C.{ext}", _tensor_to_image(c))
            write_image_8bit(sample_dir / f"A_gt.{ext}", _tensor_to_image(a))
            write_image_8bit(sample_dir / f"B_gt.{ext}", _tensor_to_image(b))
            write_image_8bit(sample_dir / f"A_pred.{ext}", _tensor_to_image(pred_a))
            write_image_8bit(sample_dir / f"B_pred.{ext}", _tensor_to_image(pred_b))
            if cfg.include_recon:
                write_image_8bit(sample_dir / f"C_hat.{ext}", _tensor_to_image(pred_recon))

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
    mask_enabled = bool(preprocess_cfg.get("mask", {}).get("enabled", True))
    val_split = float(data_cfg.get("val_split", 0.1))
    num_workers = int(data_cfg.get("num_workers", 0))
    weights_csv = data_cfg.get("weights_csv")
    require_weights = bool(data_cfg.get("require_weights", True))
    weight_tolerance = float(data_cfg.get("weight_tolerance", 1e-3))

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
        weights_csv=Path(weights_csv) if weights_csv else None,
        require_weights=require_weights,
        weight_tolerance=weight_tolerance,
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
    optimizer = build_optimizer(model, train_cfg)
    scheduler, scheduler_step_per_batch = build_scheduler(
        optimizer,
        train_cfg,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
    )

    loss_weights = train_cfg.get("loss", {})
    loss_fn = nn.L1Loss()
    weight_loss_fn = nn.SmoothL1Loss()
    grad_clip = float(train_cfg.get("grad_clip", 0.0))

    best_val = float("inf")
    history = []
    start_epoch = 1
    global_step = 0

    resume_path = train_cfg.get("resume_from")
    if resume_path:
        resume_path = Path(resume_path)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        checkpoint = load_checkpoint(resume_path, model, optimizer, scheduler)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("step", 0))
        best_val = float(checkpoint.get("metrics", {}).get("val_loss", best_val))
        logger.info("Resumed training from %s (epoch %d)", resume_path, start_epoch)

    image_log_cfg = _parse_image_log_config(logging_cfg, out_dir, seed)
    if not mask_enabled and image_log_cfg.mask_metrics:
        image_log_cfg = replace(image_log_cfg, mask_metrics=False)
    image_log_state = ImageLogState()
    if image_log_cfg.enabled:
        image_log_cfg.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Image logging enabled: every %d epochs, %d samples (%s split)",
            image_log_cfg.interval,
            image_log_cfg.max_samples,
            image_log_cfg.split,
        )

    for epoch in range(start_epoch, epochs + 1):
        metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            weight_loss_fn=weight_loss_fn,
            loss_weights=loss_weights,
            device=device,
            logger=logger,
            log_interval=log_interval,
            grad_clip=grad_clip,
            scheduler=scheduler,
            scheduler_step_per_batch=scheduler_step_per_batch,
        )

        if val_loader is not None:
            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                loss_fn=loss_fn,
                weight_loss_fn=weight_loss_fn,
                loss_weights=loss_weights,
                device=device,
                mask_enabled=mask_enabled,
            )
            metrics.update(val_metrics)
            logger.info(
                "Epoch %d/%d | Train %.6f | Val %.6f | L_ab %.6f | L_recon %.6f | L_x %.6f | x_hat μ/σ %.3f/%.3f",
                epoch,
                epochs,
                metrics["train_loss"],
                metrics["val_loss"],
                metrics.get("val_l_ab", 0.0),
                metrics.get("val_l_recon", 0.0),
                metrics.get("val_l_x", 0.0),
                metrics.get("x_hat_mean", 0.0),
                metrics.get("x_hat_std", 0.0),
            )
        else:
            logger.info("Epoch %d/%d | Train %.6f", epoch, epochs, metrics["train_loss"])

        history.append({"epoch": epoch, **metrics})

        global_step += len(train_loader)

        if scheduler is not None and not scheduler_step_per_batch:
            scheduler.step()

        last_path = out_dir / "last.pt"
        save_checkpoint(
            last_path,
            model,
            optimizer,
            epoch,
            metrics,
            scheduler=scheduler,
            step=global_step,
            config=config,
        )

        if output_cfg.get("save_every", 1) and epoch % int(output_cfg.get("save_every", 1)) == 0:
            checkpoint_path = out_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                metrics,
                scheduler=scheduler,
                step=global_step,
                config=config,
            )

        if output_cfg.get("save_best", True):
            current_val = metrics.get("val_loss", metrics["train_loss"])
            if current_val < best_val:
                best_val = current_val
                best_path = out_dir / "best.pt"
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    epoch,
                    metrics,
                    scheduler=scheduler,
                    step=global_step,
                    config=config,
                )

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
