"""Inference pipeline for dual-output U-Net models with weight prediction."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.inference.api import InferenceRunResult, run_inference_core
from src.utils.logging import add_file_handler, get_git_commit, get_logger
from src.utils.reporting import (
    make_qual_grid,
    plot_weights_hist,
    safe_relpath,
    write_report_json,
)


def run_inference(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run inference and save predicted A/B patterns."""
    log_level_name = str(config.get("logging", {}).get("log_level", "INFO")).upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    logger = get_logger(__name__, level=log_level)

    output_cfg = config.get("output", {})

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

    result: InferenceRunResult = run_inference_core(config)

    weight_rows = result.weight_rows
    x_hat_values = result.x_hat_values
    metrics = result.recon_metrics

    if output_cfg.get("save_weights", True) and weight_rows:
        weights_path = out_dir / "weights.csv"
        weights_path.write_text(
            "sample_id,x_hat,y_hat\n"
            + "\n".join(
                f"{row['sample_id']},{row['x_hat']:.6f},{row['y_hat']:.6f}"
                for row in weight_rows
            )
        )

    if result.qual_samples["C"]:
        make_qual_grid(
            np.stack(result.qual_samples["C"], axis=0),
            None,
            None,
            np.stack(result.qual_samples["A"], axis=0),
            np.stack(result.qual_samples["B"], axis=0),
            np.stack(result.qual_samples["C_hat"], axis=0)
            if result.qual_samples["C_hat"]
            else np.stack(result.qual_samples["C"], axis=0),
            qual_grid_path,
        )
        logger.info("Qual grid saved to %s", qual_grid_path)

    if plot_weights_hist(np.asarray(x_hat_values), weights_plot_path):
        logger.info("Weights histogram saved to %s", weights_plot_path)

    data_cfg = config.get("data", {})
    mixed_dir = Path(data_cfg.get("mixed_dir", "data/synthetic/C"))
    dataset_tag = str(data_cfg.get("dataset") or data_cfg.get("data_tag") or mixed_dir.name)
    dataset_path = safe_relpath(mixed_dir, repo_root)
    inference_cfg = config.get("inference", {})
    checkpoint_path = Path(inference_cfg.get("checkpoint", ""))

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
        "metrics": metrics or {"processed": float(result.processed)},
        "figures": report_figures,
        "notes": [],
        "artifacts": {
            "checkpoint": safe_relpath(checkpoint_path, repo_root),
        },
        "status": "complete",
        "progress": {
            "epoch": 1,
            "epochs_total": 1,
            "global_step": int(result.processed),
        },
    }
    if output_cfg.get("save_weights", True) and weight_rows:
        report_payload["artifacts"]["weights_csv"] = safe_relpath(
            out_dir / "weights.csv", repo_root
        )
    report_path = write_report_json(out_dir, report_payload)
    logger.info("Report JSON written to %s", report_path)
    logger.info("Monitoring figures written to %s", monitoring_dir)

    logger.info("Inference outputs saved to %s", out_dir)
    return {
        "processed": result.processed,
        "failed": result.failed,
        "output_counts": result.output_counts,
        "output_dir": str(out_dir),
    }
