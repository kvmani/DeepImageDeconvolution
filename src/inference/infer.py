"""Inference pipeline for dual-output U-Net models with weight prediction."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import torch
from torch.utils.data import DataLoader

from src.datasets import KikuchiMixedDataset
from src.models import build_model
from src.preprocessing.mask import build_circular_mask
from src.utils.io import write_image_16bit
from src.utils.logging import ProgressLogger, add_file_handler, get_git_commit, get_logger
from src.utils.reporting import (
    make_qual_grid,
    plot_weights_hist,
    safe_relpath,
    write_report_json,
)


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])


def run_inference(config: Dict[str, Any]) -> Dict[str, Any]:
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
    repo_root = Path(__file__).resolve().parents[2]
    run_id = out_dir.name
    monitoring_dir = out_dir / "monitoring"
    monitoring_dir.mkdir(parents=True, exist_ok=True)
    qual_grid_path = monitoring_dir / "qual_grid_infer.png"
    weights_plot_path = monitoring_dir / "weights_hist.png"
    if bool(config.get("logging", {}).get("log_to_file", True)):
        add_file_handler(out_dir / "output.log")

    config_path = out_dir / "config_used.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    config_relpath = safe_relpath(config_path, repo_root)
    git_commit = get_git_commit(repo_root) or ""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    checkpoint_path = Path(inference_cfg.get("checkpoint", ""))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    mixed_dir = Path(data_cfg.get("mixed_dir", "data/synthetic/C"))
    dataset_tag = str(data_cfg.get("dataset") or data_cfg.get("data_tag") or mixed_dir.name)
    dataset_path = safe_relpath(mixed_dir, repo_root)
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
    if output_format != "png":
        raise ValueError("output.format must be 'png' for canonical 16-bit outputs.")
    suffix = ".png"

    save_recon = bool(output_cfg.get("save_recon", output_cfg.get("save_sum", True)))
    save_weights = bool(output_cfg.get("save_weights", True))
    clamp_outputs = bool(inference_cfg.get("clamp_outputs", True))

    (out_dir / "A").mkdir(parents=True, exist_ok=True)
    (out_dir / "B").mkdir(parents=True, exist_ok=True)
    if save_recon:
        (out_dir / "C_hat").mkdir(parents=True, exist_ok=True)

    weight_rows = []
    x_hat_values: List[float] = []
    recon_l1_sum = 0.0
    recon_l2_sum = 0.0
    recon_count = 0
    qual_limit = 4
    qual_c: List[np.ndarray] = []
    qual_a: List[np.ndarray] = []
    qual_b: List[np.ndarray] = []
    qual_c_hat: List[np.ndarray] = []

    processed = 0
    failed = 0
    progress = ProgressLogger(total=len(dataset), logger=logger, every=max(1, len(dataset) // 10), unit="img")

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            try:
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

                x_weight = x_hat.view(-1, 1, 1, 1)
                recon = (x_weight * pred_a) + ((1.0 - x_weight) * pred_b)
                recon = recon.clamp(0.0, 1.0)
                recon_diff = recon - c
                recon_l1_sum += float(recon_diff.abs().mean().item()) * c.shape[0]
                recon_l2_sum += float((recon_diff ** 2).mean().item()) * c.shape[0]
                recon_count += int(c.shape[0])
                x_hat_values.extend([float(val) for val in x_hat.detach().cpu().view(-1).tolist()])

                sample_ids = batch["sample_id"]
                for i, sample_id in enumerate(sample_ids):
                    a_img = pred_a[i, 0].cpu().numpy()
                    b_img = pred_b[i, 0].cpu().numpy()
                    write_image_16bit(out_dir / "A" / f"{sample_id}_A{suffix}", a_img)
                    write_image_16bit(out_dir / "B" / f"{sample_id}_B{suffix}", b_img)
                    if save_recon:
                        recon_img = recon[i, 0].cpu().numpy()
                        write_image_16bit(out_dir / "C_hat" / f"{sample_id}_C{suffix}", recon_img)
                    if save_weights:
                        weight_rows.append(
                            {
                                "sample_id": sample_id,
                                "x_hat": float(x_hat[i].item()),
                                "y_hat": float(y_hat[i].item()),
                            }
                        )
                    if len(qual_c) < qual_limit:
                        qual_c.append(c[i, 0].cpu().numpy())
                        qual_a.append(a_img)
                        qual_b.append(b_img)
                        qual_c_hat.append(recon[i, 0].cpu().numpy())

                if batch_idx == 1:
                    logger.info("Processed batch %d with C shape %s", batch_idx, tuple(c.shape))
            except Exception as exc:
                failed += len(batch.get("sample_id", []))
                logger.exception("Failed during inference batch %d: %s", batch_idx, exc)
                raise
            finally:
                processed += len(batch.get("sample_id", []))
                progress.update(len(batch.get("sample_id", [])))

    if save_weights and weight_rows:
        weights_path = out_dir / "weights.csv"
        weights_path.write_text(
            "sample_id,x_hat,y_hat\n"
            + "\n".join(
                f"{row['sample_id']},{row['x_hat']:.6f},{row['y_hat']:.6f}"
                for row in weight_rows
            )
        )

    metrics: Dict[str, float] = {}
    if recon_count > 0:
        metrics["recon_l1"] = recon_l1_sum / recon_count
        metrics["recon_l2"] = recon_l2_sum / recon_count
    if x_hat_values:
        metrics["x_hat_mean"] = float(np.mean(x_hat_values))
        metrics["x_hat_std"] = float(np.std(x_hat_values))

    if qual_c:
        make_qual_grid(
            np.stack(qual_c, axis=0),
            None,
            None,
            np.stack(qual_a, axis=0),
            np.stack(qual_b, axis=0),
            np.stack(qual_c_hat, axis=0),
            qual_grid_path,
        )
        logger.info("Qual grid saved to %s", qual_grid_path)

    if plot_weights_hist(np.asarray(x_hat_values), weights_plot_path):
        logger.info("Weights histogram saved to %s", weights_plot_path)

    report_figures: Dict[str, Any] = {
        "qual_grid": safe_relpath(qual_grid_path, repo_root),
    }
    if weights_plot_path.exists():
        report_figures["weights_plot"] = safe_relpath(weights_plot_path, repo_root)

    report_payload = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit,
        "stage": "infer",
        "dataset": dataset_tag,
        "dataset_path": dataset_path,
        "config": config_relpath,
        "metrics": metrics or {"processed": float(processed)},
        "figures": report_figures,
        "notes": [],
        "artifacts": {
            "checkpoint": safe_relpath(checkpoint_path, repo_root),
        },
        "status": "complete",
        "progress": {
            "epoch": 1,
            "epochs_total": 1,
            "global_step": int(processed),
        },
    }
    if save_weights and weight_rows:
        report_payload["artifacts"]["weights_csv"] = safe_relpath(
            out_dir / "weights.csv", repo_root
        )
    report_path = write_report_json(out_dir, report_payload)
    logger.info("Report JSON written to %s", report_path)
    logger.info("Monitoring figures written to %s", monitoring_dir)

    output_counts = {
        "A": len(list((out_dir / "A").glob("*.png"))),
        "B": len(list((out_dir / "B").glob("*.png"))),
        "C_hat": len(list((out_dir / "C_hat").glob("*.png"))) if save_recon else 0,
    }
    logger.info("Inference outputs saved to %s", out_dir)
    return {
        "processed": processed,
        "failed": failed,
        "output_counts": output_counts,
        "output_dir": str(out_dir),
    }
