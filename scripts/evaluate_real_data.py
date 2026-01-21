"""Evaluate a trained model on real mixed patterns with masked metrics."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models import build_model
from src.preprocessing.mask import build_circular_mask
from src.training.metrics import physics_consistency
from src.utils.config import load_config
from src.utils.io import collect_image_paths, read_image_float01, write_image_16bit
from src.utils.logging import ProgressLogger, add_file_handler, get_git_commit, get_logger
from src.utils.metrics import aggregate_metrics, masked_l1, masked_mse, mse
from src.utils.reporting import make_qual_grid, plot_weights_hist, safe_relpath, write_report_json


DEFAULT_CONFIG = REPO_ROOT / "configs/eval_real_default.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a model on real mixed patterns.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    parser.add_argument("--out_dir", type=str, default=None, help="Override output directory.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Override checkpoint path.")
    parser.add_argument("--mixed-dir", type=str, default=None, help="Override mixed input dir.")
    parser.add_argument("--reference-a", type=str, default=None, help="Override reference A path.")
    parser.add_argument("--reference-b", type=str, default=None, help="Override reference B path.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level.",
    )
    return parser.parse_args()


def _apply_override(config: Dict[str, Any], key: str, value: Optional[str]) -> None:
    if value is None:
        return
    config.setdefault("data", {})
    if key == "mixed_dir":
        config["data"]["mixed_dir"] = value
    elif key == "reference_a":
        config["data"]["reference_a"] = value
    elif key == "reference_b":
        config["data"]["reference_b"] = value
    elif key == "checkpoint":
        config.setdefault("inference", {})["checkpoint"] = value
    elif key == "out_dir":
        config.setdefault("output", {})["out_dir"] = value


def _to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array.astype(np.float32)).unsqueeze(0).unsqueeze(0)


def _center_crop(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    height, width = image.shape
    target_h, target_w = target_shape
    if height == target_h and width == target_w:
        return image
    top = max((height - target_h) // 2, 0)
    left = max((width - target_w) // 2, 0)
    return image[top : top + target_h, left : left + target_w]


def _accumulate(metrics_sum: Dict[str, float], metrics: Dict[str, float]) -> None:
    for key, value in metrics.items():
        metrics_sum[key] = metrics_sum.get(key, 0.0) + float(value)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config) if args.config else DEFAULT_CONFIG
    config = load_config(config_path)

    if args.debug:
        config.setdefault("debug", {})["enabled"] = True

    _apply_override(config, "out_dir", args.out_dir)
    _apply_override(config, "checkpoint", args.checkpoint)
    _apply_override(config, "mixed_dir", args.mixed_dir)
    _apply_override(config, "reference_a", args.reference_a)
    _apply_override(config, "reference_b", args.reference_b)

    log_level_name = str(config.get("logging", {}).get("log_level", args.log_level)).upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    logger = get_logger(__name__, level=log_level)

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    inference_cfg = config.get("inference", {})
    output_cfg = config.get("output", {})
    eval_cfg = config.get("evaluation", {})
    debug_cfg = config.get("debug", {})

    out_dir = Path(output_cfg.get("out_dir", "outputs/eval_real"))
    out_dir.mkdir(parents=True, exist_ok=True)
    if bool(config.get("logging", {}).get("log_to_file", True)):
        add_file_handler(out_dir / "output.log")

    repo_root = REPO_ROOT
    run_id = out_dir.name
    config_out = out_dir / "config_used.json"
    config_out.write_text(json.dumps(config, indent=2))
    config_relpath = safe_relpath(config_out, repo_root)
    git_commit = get_git_commit(repo_root) or ""

    output_format = str(output_cfg.get("format", "png")).lower()
    if output_format != "png":
        raise ValueError("output.format must be 'png' for canonical 16-bit outputs.")

    checkpoint_path = Path(inference_cfg.get("checkpoint", ""))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    mixed_dir = Path(data_cfg.get("mixed_dir", "data/raw/Double Pattern Data"))
    allow_non_16bit = bool(data_cfg.get("allow_non_16bit", True))
    extensions = data_cfg.get("extensions", [".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"])
    recursive = bool(data_cfg.get("recursive", False))
    sample_limit = debug_cfg.get("sample_limit") if debug_cfg.get("enabled", False) else None
    auto_crop = bool(eval_cfg.get("auto_crop", False))
    apply_mask = bool(eval_cfg.get("apply_mask", True))
    qual_limit = int(eval_cfg.get("qual_limit", 4))
    save_weights = bool(eval_cfg.get("save_weights", True))

    mixed_paths = collect_image_paths(mixed_dir, recursive=recursive)
    if sample_limit:
        mixed_paths = mixed_paths[: int(sample_limit)]
    if not mixed_paths:
        raise ValueError(f"No mixed images found under {mixed_dir}")

    ref_a_path = Path(data_cfg.get("reference_a")) if data_cfg.get("reference_a") else None
    ref_b_path = Path(data_cfg.get("reference_b")) if data_cfg.get("reference_b") else None
    ref_a = read_image_float01(ref_a_path, allow_non_16bit=allow_non_16bit) if ref_a_path else None
    ref_b = read_image_float01(ref_b_path, allow_non_16bit=allow_non_16bit) if ref_b_path else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    monitoring_dir = out_dir / "monitoring"
    monitoring_dir.mkdir(parents=True, exist_ok=True)
    qual_grid_path = monitoring_dir / "qual_grid_real.png"
    weights_plot_path = monitoring_dir / "weights_hist_real.png"

    (out_dir / "A_pred").mkdir(parents=True, exist_ok=True)
    (out_dir / "B_pred").mkdir(parents=True, exist_ok=True)
    (out_dir / "C").mkdir(parents=True, exist_ok=True)
    if bool(output_cfg.get("save_recon", True)):
        (out_dir / "C_hat").mkdir(parents=True, exist_ok=True)

    metrics_sum: Dict[str, float] = {}
    count = 0
    qual_c: List[np.ndarray] = []
    qual_a: List[np.ndarray] = []
    qual_b: List[np.ndarray] = []
    qual_c_hat: List[np.ndarray] = []
    qual_ref_a: List[np.ndarray] = []
    qual_ref_b: List[np.ndarray] = []
    x_hat_values: List[float] = []
    weight_rows = []

    progress = ProgressLogger(total=len(mixed_paths), logger=logger, every=max(1, len(mixed_paths) // 10), unit="img")

    with torch.no_grad():
        for path in mixed_paths:
            c_np = read_image_float01(path, allow_non_16bit=allow_non_16bit)

            target_shape = c_np.shape
            ref_a_np = ref_a
            ref_b_np = ref_b
            if ref_a_np is not None and ref_a_np.shape != target_shape:
                if not auto_crop:
                    raise ValueError(f"Reference A shape {ref_a_np.shape} != mixed {target_shape}")
                ref_a_np = _center_crop(ref_a_np, target_shape)
            if ref_b_np is not None and ref_b_np.shape != target_shape:
                if not auto_crop:
                    raise ValueError(f"Reference B shape {ref_b_np.shape} != mixed {target_shape}")
                ref_b_np = _center_crop(ref_b_np, target_shape)

            mask = None
            if apply_mask:
                mask_np = build_circular_mask(target_shape).astype(np.float32)
                mask = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)
                c_np = c_np * mask_np
                if ref_a_np is not None:
                    ref_a_np = ref_a_np * mask_np
                if ref_b_np is not None:
                    ref_b_np = ref_b_np * mask_np

            c = _to_tensor(c_np).to(device)
            pred_a, pred_b, x_hat = model(c)
            pred_a = pred_a.clamp(0.0, 1.0)
            pred_b = pred_b.clamp(0.0, 1.0)
            if mask is not None:
                pred_a = pred_a * mask
                pred_b = pred_b * mask

            x_weight = x_hat.view(-1, 1, 1, 1)
            recon = (x_weight * pred_a) + ((1.0 - x_weight) * pred_b)
            recon = recon.clamp(0.0, 1.0)

            metrics = physics_consistency(pred_a, pred_b, x_hat, c, mask=mask)

            if ref_a_np is not None and ref_b_np is not None:
                ref_a_t = _to_tensor(ref_a_np).to(device)
                ref_b_t = _to_tensor(ref_b_np).to(device)
                psnr_a, ssim_a = aggregate_metrics(pred_a, ref_a_t, mask=mask)
                psnr_b, ssim_b = aggregate_metrics(pred_b, ref_b_t, mask=mask)
                metrics.update(
                    {
                        "psnr_a": psnr_a,
                        "ssim_a": ssim_a,
                        "psnr_b": psnr_b,
                        "ssim_b": ssim_b,
                        "l2_a": float(mse(pred_a, ref_a_t).mean().item()),
                        "l2_b": float(mse(pred_b, ref_b_t).mean().item()),
                    }
                )
                if mask is not None:
                    metrics.update(
                        {
                            "l1_a_masked": float(masked_l1(pred_a, ref_a_t, mask).mean().item()),
                            "l1_b_masked": float(masked_l1(pred_b, ref_b_t, mask).mean().item()),
                            "l2_a_masked": float(masked_mse(pred_a, ref_a_t, mask).mean().item()),
                            "l2_b_masked": float(masked_mse(pred_b, ref_b_t, mask).mean().item()),
                        }
                    )

            _accumulate(metrics_sum, metrics)
            count += 1

            sample_id = path.stem
            write_image_16bit(out_dir / "C" / f"{sample_id}_C.png", c_np)
            write_image_16bit(out_dir / "A_pred" / f"{sample_id}_A.png", pred_a[0, 0].cpu().numpy())
            write_image_16bit(out_dir / "B_pred" / f"{sample_id}_B.png", pred_b[0, 0].cpu().numpy())
            if bool(output_cfg.get("save_recon", True)):
                write_image_16bit(out_dir / "C_hat" / f"{sample_id}_C.png", recon[0, 0].cpu().numpy())

            x_hat_val = float(x_hat.item())
            x_hat_values.append(x_hat_val)
            if save_weights:
                weight_rows.append({"sample_id": sample_id, "x_hat": x_hat_val, "y_hat": 1.0 - x_hat_val})

            if len(qual_c) < qual_limit:
                qual_c.append(c_np)
                qual_a.append(pred_a[0, 0].cpu().numpy())
                qual_b.append(pred_b[0, 0].cpu().numpy())
                qual_c_hat.append(recon[0, 0].cpu().numpy())
                if ref_a_np is not None:
                    qual_ref_a.append(ref_a_np)
                if ref_b_np is not None:
                    qual_ref_b.append(ref_b_np)

            progress.update(1)

    averaged_metrics = {key: value / max(count, 1) for key, value in metrics_sum.items()}
    if x_hat_values:
        averaged_metrics["x_hat_mean"] = float(np.mean(x_hat_values))
        averaged_metrics["x_hat_std"] = float(np.std(x_hat_values))

    if qual_c:
        a_gt = np.stack(qual_ref_a, axis=0) if qual_ref_a else None
        b_gt = np.stack(qual_ref_b, axis=0) if qual_ref_b else None
        make_qual_grid(
            np.stack(qual_c, axis=0),
            a_gt,
            b_gt,
            np.stack(qual_a, axis=0),
            np.stack(qual_b, axis=0),
            np.stack(qual_c_hat, axis=0),
            qual_grid_path,
        )

    if plot_weights_hist(np.asarray(x_hat_values), weights_plot_path):
        logger.info("Weights histogram saved to %s", weights_plot_path)

    if save_weights and weight_rows:
        weights_path = out_dir / "weights.csv"
        weights_path.write_text(
            "sample_id,x_hat,y_hat\n"
            + "\n".join(
                f"{row['sample_id']},{row['x_hat']:.6f},{row['y_hat']:.6f}"
                for row in weight_rows
            )
        )

    report_figures: Dict[str, Any] = {
        "qual_grid": safe_relpath(qual_grid_path, repo_root),
    }
    if weights_plot_path.exists():
        report_figures["weights_plot"] = safe_relpath(weights_plot_path, repo_root)

    artifacts: Dict[str, str] = {
        "checkpoint": safe_relpath(checkpoint_path, repo_root),
        "mixed_dir": safe_relpath(mixed_dir, repo_root),
    }
    if save_weights and weight_rows:
        artifacts["weights_csv"] = safe_relpath(out_dir / "weights.csv", repo_root)

    report_payload = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit,
        "stage": "infer",
        "dataset": str(mixed_dir.name),
        "dataset_path": safe_relpath(mixed_dir, repo_root),
        "config": config_relpath,
        "metrics": averaged_metrics or {"processed": float(count)},
        "figures": report_figures,
        "notes": ["real_data_eval"],
        "artifacts": artifacts,
        "status": "complete",
        "progress": {"epoch": 1, "epochs_total": 1, "global_step": int(count)},
    }
    report_path = write_report_json(out_dir, report_payload)
    logger.info("Report JSON written to %s", report_path)


if __name__ == "__main__":
    main()
