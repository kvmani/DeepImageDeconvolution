"""Inference pipeline for dual-output U-Net models with weight prediction."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from src.datasets import KikuchiMixedDataset
from src.models import build_model
from src.preprocessing.mask import build_circular_mask
from src.utils.io import write_image_16bit
from src.utils.logging import add_file_handler, get_logger


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])


def run_inference(config: Dict[str, Any]) -> None:
    """Run inference and save predicted A/B patterns."""
    log_level_name = str(config.get("logging", {}).get("log_level", "INFO")).upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    logger = get_logger(__name__, level=log_level)

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    inference_cfg = config.get("inference", {})
    output_cfg = config.get("output", {})
    post_cfg = config.get("postprocess", {})
    debug_cfg = config.get("debug", {})

    out_dir = Path(output_cfg.get("out_dir", "outputs/infer_run"))
    out_dir.mkdir(parents=True, exist_ok=True)
    if bool(config.get("logging", {}).get("log_to_file", True)):
        add_file_handler(out_dir / "output.log")

    config_path = out_dir / "config_used.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    checkpoint_path = Path(inference_cfg.get("checkpoint", ""))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    mixed_dir = Path(data_cfg.get("mixed_dir", "data/synthetic/C"))
    extensions = data_cfg.get("extensions", [".png", ".tif", ".tiff"])
    preprocess_cfg = data_cfg.get("preprocess", {})
    num_workers = int(data_cfg.get("num_workers", 0))

    sample_limit = debug_cfg.get("sample_limit") if debug_cfg.get("enabled", False) else None
    apply_mask = bool(post_cfg.get("apply_mask", True))

    dataset = KikuchiMixedDataset(
        mixed_dir=mixed_dir,
        extensions=extensions,
        preprocess_cfg=preprocess_cfg,
        seed=int(debug_cfg.get("seed", 42)),
        limit=sample_limit,
        return_mask=apply_mask,
    )

    batch_size = int(inference_cfg.get("batch_size", 4))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = build_model(model_cfg).to(device)
    _load_checkpoint(model, checkpoint_path)
    model.eval()

    output_format = str(output_cfg.get("format", "png")).lower()
    if output_format not in ("png", "tif", "tiff"):
        raise ValueError("output.format must be one of: png, tif, tiff.")
    suffix = ".tif" if output_format in ("tif", "tiff") else ".png"

    save_recon = bool(output_cfg.get("save_recon", output_cfg.get("save_sum", True)))
    save_weights = bool(output_cfg.get("save_weights", True))
    clamp_outputs = bool(inference_cfg.get("clamp_outputs", True))

    (out_dir / "A").mkdir(parents=True, exist_ok=True)
    (out_dir / "B").mkdir(parents=True, exist_ok=True)
    if save_recon:
        (out_dir / "C_hat").mkdir(parents=True, exist_ok=True)

    weight_rows = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            c = batch["C"].to(device)
            pred_a, pred_b, x_hat = model(c)
            y_hat = 1.0 - x_hat

            if clamp_outputs:
                pred_a = pred_a.clamp(0.0, 1.0)
                pred_b = pred_b.clamp(0.0, 1.0)

            if apply_mask:
                if "mask" in batch:
                    mask = batch["mask"].to(device)
                else:
                    mask = torch.from_numpy(
                        build_circular_mask(c.shape[-2:]).astype("float32")
                    ).unsqueeze(0)
                    mask = mask.to(device)
                pred_a = pred_a * mask
                pred_b = pred_b * mask

            sample_ids = batch["sample_id"]
            for i, sample_id in enumerate(sample_ids):
                a_img = pred_a[i, 0].cpu().numpy()
                b_img = pred_b[i, 0].cpu().numpy()
                write_image_16bit(out_dir / "A" / f"{sample_id}_A{suffix}", a_img)
                write_image_16bit(out_dir / "B" / f"{sample_id}_B{suffix}", b_img)
                if save_recon:
                    recon = (
                        float(x_hat[i].item()) * a_img + float(y_hat[i].item()) * b_img
                    ).clip(0.0, 1.0)
                    write_image_16bit(out_dir / "C_hat" / f"{sample_id}_C{suffix}", recon)
                if save_weights:
                    weight_rows.append(
                        {
                            "sample_id": sample_id,
                            "x_hat": float(x_hat[i].item()),
                            "y_hat": float(y_hat[i].item()),
                        }
                    )

            if batch_idx == 1:
                logger.info("Processed batch %d with C shape %s", batch_idx, tuple(c.shape))

    if save_weights and weight_rows:
        weights_path = out_dir / "weights.csv"
        weights_path.write_text(
            "sample_id,x_hat,y_hat\n"
            + "\n".join(
                f"{row['sample_id']},{row['x_hat']:.6f},{row['y_hat']:.6f}"
                for row in weight_rows
            )
        )

    logger.info("Inference outputs saved to %s", out_dir)
