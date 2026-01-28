"""Reusable inference API for CLI and GUI."""
from __future__ import annotations

from dataclasses import dataclass
import copy
import logging
from pathlib import Path
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.datasets import KikuchiMixedDataset, KikuchiMixedListDataset
from src.inference.metrics import compute_image_metrics, summarize_metrics
from src.inference.model_manager import ModelManager
from src.preprocessing.mask import build_circular_mask
from src.preprocessing.pipeline import PreprocessConfig, apply_preprocess, parse_preprocess_cfg
from src.preprocessing.transforms import apply_crop_indices
from src.utils.config import deep_update
from src.utils.io import read_image_float01, write_image_16bit
from src.utils.logging import ProgressLogger, ProgressSnapshot, get_logger


@dataclass
class InferenceSampleResult:
    """Per-sample inference outputs and metrics."""

    sample_id: str
    input_path: Path
    output_a: Path
    output_b: Path
    output_c_hat: Optional[Path]
    metrics: Dict[str, float]


@dataclass
class InferenceRunResult:
    """Summary and artifacts from a single inference run."""

    processed: int
    failed: int
    output_counts: Dict[str, int]
    output_dir: Path
    weight_rows: List[Dict[str, float]]
    x_hat_values: List[float]
    recon_metrics: Dict[str, float]
    sample_results: List[InferenceSampleResult]
    sample_metrics_summary: Dict[str, float]
    qual_samples: Dict[str, List[np.ndarray]]
    cancelled: bool


def merge_inference_config(
    base: Dict[str, Any],
    gui_overrides: Optional[Dict[str, Any]] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Merge inference configuration with precedence: CLI > GUI > YAML."""
    merged: Dict[str, Any] = copy.deepcopy(base)
    if gui_overrides:
        merged = deep_update(merged, gui_overrides)
    if cli_overrides:
        merged = deep_update(merged, cli_overrides)
    return merged


def _resolve_device(inference_cfg: Dict[str, Any]) -> torch.device:
    preferred = str(inference_cfg.get("device", "auto")).lower()
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "cuda":
        if not torch.cuda.is_available():
            logger = get_logger(__name__)
            logger.warning("CUDA requested but not available; falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_dataset(
    config: Dict[str, Any],
    input_paths: Optional[Sequence[Path]],
) -> Dataset:
    data_cfg = config.get("data", {})
    debug_cfg = config.get("debug", {})
    post_cfg = config.get("postprocess", {})
    preprocess_cfg = data_cfg.get("preprocess", {})

    sample_limit = debug_cfg.get("sample_limit") if debug_cfg.get("enabled", False) else None
    mask_cfg = preprocess_cfg.get("mask", {})
    apply_mask = bool(post_cfg.get("apply_mask", True))
    return_mask = apply_mask or bool(mask_cfg.get("enabled", True))
    seed = int(debug_cfg.get("seed", 42))

    if input_paths is None:
        mixed_dir = Path(data_cfg.get("mixed_dir", "data/synthetic/C"))
        extensions = data_cfg.get("extensions", [".png", ".tif", ".tiff"])
        return KikuchiMixedDataset(
            mixed_dir=mixed_dir,
            extensions=extensions,
            preprocess_cfg=preprocess_cfg,
            seed=seed,
            limit=sample_limit,
            return_mask=return_mask,
        )
    return KikuchiMixedListDataset(
        paths=input_paths,
        preprocess_cfg=preprocess_cfg,
        seed=seed,
        limit=sample_limit,
        return_mask=return_mask,
    )


def _extract_crop_meta(
    batch: Dict[str, Any],
    index: int,
) -> Optional[Tuple[int, int, Tuple[int, int]]]:
    crop = batch.get("crop")
    if not isinstance(crop, dict):
        return None
    enabled = crop.get("enabled")
    if enabled is None:
        return None
    if isinstance(enabled, torch.Tensor):
        enabled_value = bool(enabled[index].item())
    else:
        enabled_value = bool(enabled[index])
    if not enabled_value:
        return None
    top = crop.get("top")
    left = crop.get("left")
    height = crop.get("height")
    width = crop.get("width")
    if top is None or left is None or height is None or width is None:
        return None
    if isinstance(top, torch.Tensor):
        top_value = int(top[index].item())
        left_value = int(left[index].item())
        height_value = int(height[index].item())
        width_value = int(width[index].item())
    else:
        top_value = int(top[index])
        left_value = int(left[index])
        height_value = int(height[index])
        width_value = int(width[index])
    return top_value, left_value, (height_value, width_value)


def _preprocess_gt(
    gt_image: np.ndarray,
    preprocess_cfg: PreprocessConfig,
    mask_np: Optional[np.ndarray],
    crop_meta: Optional[Tuple[int, int, Tuple[int, int]]],
    logger: logging.Logger,
    sample_id: str,
) -> Optional[np.ndarray]:
    if preprocess_cfg.crop_enabled:
        if crop_meta is None:
            logger.warning(
                "Skipping GT metrics for %s because crop metadata is missing.", sample_id
            )
            return None
        top, left, crop_size = crop_meta
        gt_image = apply_crop_indices(gt_image, top, left, crop_size)
    return apply_preprocess(gt_image, preprocess_cfg, mask_np)


def _normalize_sample_id(path: Path, tag: str) -> str:
    stem = path.stem
    suffix = f"_{tag}"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def build_gt_lookup(paths: Iterable[Path], tag: str) -> Dict[str, Path]:
    """Build a mapping from sample id to ground truth path."""
    mapping: Dict[str, Path] = {}
    for path in paths:
        sample_id = _normalize_sample_id(path, tag)
        if sample_id in mapping:
            raise ValueError(f"Duplicate GT sample id {sample_id} from {path}")
        mapping[sample_id] = path
    return mapping


def run_inference_core(
    config: Dict[str, Any],
    input_paths: Optional[Sequence[Path]] = None,
    output_dir: Optional[Path] = None,
    model_manager: Optional[ModelManager] = None,
    progress_callback: Optional[Callable[[ProgressSnapshot], None]] = None,
    stop_event: Optional[threading.Event] = None,
    gt_lookup: Optional[Dict[str, Dict[str, Path]]] = None,
) -> InferenceRunResult:
    """Run inference over provided inputs and return structured results."""
    log_level_name = str(config.get("logging", {}).get("log_level", "INFO")).upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    logger = get_logger(__name__, level=log_level)

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    inference_cfg = config.get("inference", {})
    output_cfg = config.get("output", {})
    post_cfg = config.get("postprocess", {})
    preprocess_cfg = parse_preprocess_cfg(data_cfg.get("preprocess", {}))

    out_dir = Path(output_dir or output_cfg.get("out_dir", "outputs/infer_run"))
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(inference_cfg.get("checkpoint", ""))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    dataset = _build_dataset(config, input_paths)
    num_workers = int(data_cfg.get("num_workers", 0))
    batch_size = int(inference_cfg.get("batch_size", 4))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    device = _resolve_device(inference_cfg)
    logger.info("Using device: %s", device)

    if model_manager is None:
        model_manager = ModelManager()
    model = model_manager.get_model(model_cfg, checkpoint_path, device)

    output_format = str(output_cfg.get("format", "png")).lower()
    if output_format != "png":
        raise ValueError("output.format must be 'png' for canonical 16-bit outputs.")
    suffix = ".png"

    save_recon = bool(output_cfg.get("save_recon", output_cfg.get("save_sum", True)))
    save_weights = bool(output_cfg.get("save_weights", True))
    clamp_outputs = bool(inference_cfg.get("clamp_outputs", True))
    apply_mask = bool(post_cfg.get("apply_mask", True))

    (out_dir / "A").mkdir(parents=True, exist_ok=True)
    (out_dir / "B").mkdir(parents=True, exist_ok=True)
    if save_recon:
        (out_dir / "C_hat").mkdir(parents=True, exist_ok=True)

    weight_rows: List[Dict[str, float]] = []
    x_hat_values: List[float] = []
    recon_l1_sum = 0.0
    recon_l2_sum = 0.0
    recon_count = 0
    qual_limit = 4
    qual_samples: Dict[str, List[np.ndarray]] = {
        "C": [],
        "A": [],
        "B": [],
        "C_hat": [],
    }
    sample_results: List[InferenceSampleResult] = []
    sample_metrics: List[Dict[str, float]] = []

    processed = 0
    failed = 0
    cancelled = False
    progress = ProgressLogger(
        total=len(dataset),
        logger=logger,
        every=max(1, len(dataset) // 10),
        unit="img",
    )

    start_time = time.perf_counter()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            if stop_event is not None and stop_event.is_set():
                cancelled = True
                logger.warning("Inference cancelled by user.")
                break
            try:
                c = batch["C"].to(device)
                pred_a, pred_b, x_hat = model(c)
                y_hat = 1.0 - x_hat

                if clamp_outputs:
                    pred_a = pred_a.clamp(0.0, 1.0)
                    pred_b = pred_b.clamp(0.0, 1.0)

                mask = None
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
                recon_l2_sum += float((recon_diff**2).mean().item()) * c.shape[0]
                recon_count += int(c.shape[0])
                x_hat_values.extend([float(val) for val in x_hat.detach().cpu().view(-1).tolist()])

                sample_ids = batch["sample_id"]
                for i, sample_id in enumerate(sample_ids):
                    a_img = pred_a[i, 0].cpu().numpy()
                    b_img = pred_b[i, 0].cpu().numpy()
                    out_a = out_dir / "A" / f"{sample_id}_A{suffix}"
                    out_b = out_dir / "B" / f"{sample_id}_B{suffix}"
                    write_image_16bit(out_a, a_img)
                    write_image_16bit(out_b, b_img)

                    out_c_hat = None
                    if save_recon:
                        recon_img = recon[i, 0].cpu().numpy()
                        out_c_hat = out_dir / "C_hat" / f"{sample_id}_C{suffix}"
                        write_image_16bit(out_c_hat, recon_img)

                    if save_weights:
                        weight_rows.append(
                            {
                                "sample_id": sample_id,
                                "x_hat": float(x_hat[i].item()),
                                "y_hat": float(y_hat[i].item()),
                            }
                        )

                    if len(qual_samples["C"]) < qual_limit:
                        qual_samples["C"].append(c[i, 0].cpu().numpy())
                        qual_samples["A"].append(a_img)
                        qual_samples["B"].append(b_img)
                        if save_recon and out_c_hat is not None:
                            qual_samples["C_hat"].append(recon[i, 0].cpu().numpy())

                    metrics: Dict[str, float] = {}
                    if gt_lookup:
                        gt_entry = gt_lookup.get(sample_id, {})
                        gt_a_path = gt_entry.get("A")
                        gt_b_path = gt_entry.get("B")
                        mask_np = None
                        if mask is not None:
                            mask_np = mask[i, 0].cpu().numpy()
                        crop_meta = _extract_crop_meta(batch, i)
                        if gt_a_path is not None:
                            gt_a = read_image_float01(gt_a_path, allow_non_16bit=True)
                            gt_a = _preprocess_gt(
                                gt_a, preprocess_cfg, mask_np, crop_meta, logger, str(sample_id)
                            )
                            if gt_a is None:
                                gt_a_path = None
                        if gt_a_path is not None:
                            metrics.update(
                                {
                                    f"a_{key}": value
                                    for key, value in compute_image_metrics(
                                        a_img, gt_a, mask=mask_np
                                    ).items()
                                }
                            )
                        if gt_b_path is not None:
                            gt_b = read_image_float01(gt_b_path, allow_non_16bit=True)
                            gt_b = _preprocess_gt(
                                gt_b, preprocess_cfg, mask_np, crop_meta, logger, str(sample_id)
                            )
                            if gt_b is None:
                                gt_b_path = None
                        if gt_b_path is not None:
                            metrics.update(
                                {
                                    f"b_{key}": value
                                    for key, value in compute_image_metrics(
                                        b_img, gt_b, mask=mask_np
                                    ).items()
                                }
                            )
                        if metrics:
                            sample_metrics.append(metrics)

                    sample_results.append(
                        InferenceSampleResult(
                            sample_id=str(sample_id),
                            input_path=Path(batch["path"][i]),
                            output_a=out_a,
                            output_b=out_b,
                            output_c_hat=out_c_hat,
                            metrics=metrics,
                        )
                    )

                if batch_idx == 1:
                    logger.info("Processed batch %d with C shape %s", batch_idx, tuple(c.shape))
            except Exception as exc:
                failed += len(batch.get("sample_id", []))
                logger.exception("Failed during inference batch %d: %s", batch_idx, exc)
                raise
            finally:
                processed += len(batch.get("sample_id", []))
                snapshot = progress.update(len(batch.get("sample_id", [])))
                if progress_callback is not None:
                    progress_callback(snapshot)

    elapsed = time.perf_counter() - start_time
    logger.info("Inference elapsed time: %.2fs", elapsed)

    recon_metrics: Dict[str, float] = {}
    if recon_count > 0:
        recon_metrics["recon_l1"] = recon_l1_sum / recon_count
        recon_metrics["recon_l2"] = recon_l2_sum / recon_count
    if x_hat_values:
        recon_metrics["x_hat_mean"] = float(np.mean(x_hat_values))
        recon_metrics["x_hat_std"] = float(np.std(x_hat_values))

    output_counts = {
        "A": len(list((out_dir / "A").glob("*.png"))),
        "B": len(list((out_dir / "B").glob("*.png"))),
        "C_hat": len(list((out_dir / "C_hat").glob("*.png"))) if save_recon else 0,
    }

    return InferenceRunResult(
        processed=processed,
        failed=failed,
        output_counts=output_counts,
        output_dir=out_dir,
        weight_rows=weight_rows,
        x_hat_values=x_hat_values,
        recon_metrics=recon_metrics,
        sample_results=sample_results,
        sample_metrics_summary=summarize_metrics(sample_metrics),
        qual_samples=qual_samples,
        cancelled=cancelled,
    )
