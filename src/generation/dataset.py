"""Dataset generation for synthetic Kikuchi pattern mixtures."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import csv
import json

import numpy as np

from src.generation.mix import MixPipeline, mix_normalize_then_mix, mix_then_normalize
from src.preprocessing.mask import apply_circular_mask, build_circular_mask, detect_circular_mask
from src.preprocessing.normalise import normalize_image
from src.preprocessing.transforms import (
    apply_crop,
    apply_flip,
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_rotation_90,
)
from src.utils.io import collect_image_paths, read_image_16bit, to_float01, write_image_16bit
from src.utils.logging import get_logger


@dataclass
class DebugSettings:
    """Debug settings for dataset generation."""

    enabled: bool
    seed: int
    visualize: bool
    max_visualizations: int
    sample_limit: Optional[int]


def _ensure_same_shape(
    image_a: np.ndarray,
    image_b: np.ndarray,
    auto_crop: bool,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if image_a.shape == image_b.shape:
        return image_a, image_b
    if not auto_crop:
        raise ValueError(
            f"Image shapes do not match and auto-crop is disabled: {image_a.shape} vs {image_b.shape}."
        )
    target_h = min(image_a.shape[0], image_b.shape[0])
    target_w = min(image_a.shape[1], image_b.shape[1])
    size = (target_h, target_w)
    image_a = apply_crop(image_a, size=size, mode="center", rng=rng)
    image_b = apply_crop(image_b, size=size, mode="center", rng=rng)
    return image_a, image_b


def _apply_augmentations(
    image: np.ndarray,
    rng: np.random.Generator,
    augment_cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    flip_h = False
    flip_v = False
    rot_k = 0

    if augment_cfg.get("flip_horizontal", False):
        flip_h = bool(rng.random() < 0.5)
    if augment_cfg.get("flip_vertical", False):
        flip_v = bool(rng.random() < 0.5)
    if augment_cfg.get("rotate90", False):
        rot_k = int(rng.integers(0, 4))

    image = apply_flip(image, flip_horizontal=flip_h, flip_vertical=flip_v)
    image = apply_rotation_90(image, k=rot_k)

    meta = {"flip_horizontal": flip_h, "flip_vertical": flip_v, "rotate90_k": rot_k}
    return image, meta


def _preprocess_image(
    image: np.ndarray,
    rng: np.random.Generator,
    preprocess_cfg: Dict[str, Any],
    apply_augment: bool,
) -> Tuple[np.ndarray, Dict[str, Any], Optional[np.ndarray]]:
    meta: Dict[str, Any] = {}
    mask: Optional[np.ndarray] = None

    crop_cfg = preprocess_cfg.get("crop", {})
    crop_enabled = crop_cfg.get("enabled", False)
    if crop_enabled:
        size_list = crop_cfg.get("size")
        if not size_list or len(size_list) != 2:
            raise ValueError("Crop size must be a 2-element list when cropping is enabled.")
        size = tuple(size_list)
        mode = crop_cfg.get("mode", "center")
        image = apply_crop(image, size=size, mode=mode, rng=rng)
        meta["crop"] = {"size": size, "mode": mode}

    denoise_cfg = preprocess_cfg.get("denoise", {})
    if denoise_cfg.get("enabled", False):
        sigma = float(denoise_cfg.get("sigma", 0.0))
        image = apply_gaussian_blur(image, sigma=sigma)
        meta["denoise_sigma"] = sigma

    if apply_augment:
        augment_cfg = preprocess_cfg.get("augment", {})
        image, aug_meta = _apply_augmentations(image, rng, augment_cfg)
        meta["augment"] = aug_meta

    mask_cfg = preprocess_cfg.get("mask", {})
    mask_enabled = bool(mask_cfg.get("enabled", True))
    if mask_enabled:
        mask = build_circular_mask(image.shape)
        detect_existing = bool(mask_cfg.get("detect_existing", True))
        outside_zero_fraction = float(mask_cfg.get("outside_zero_fraction", 0.98))
        zero_tolerance = float(mask_cfg.get("zero_tolerance", 1e-6))
        already_masked = None
        measured_fraction = None
        if detect_existing:
            already_masked, measured_fraction = detect_circular_mask(
                image,
                mask,
                zero_tolerance=zero_tolerance,
                outside_zero_fraction=outside_zero_fraction,
            )
        image = apply_circular_mask(image, mask)
        meta["mask"] = {
            "enabled": True,
            "already_masked": already_masked,
            "outside_zero_fraction": measured_fraction,
            "zero_tolerance": zero_tolerance,
            "outside_zero_threshold": outside_zero_fraction,
        }
    else:
        meta["mask"] = {"enabled": False}

    normalize_cfg = preprocess_cfg.get("normalize", {})
    if normalize_cfg.get("enabled", True):
        method = normalize_cfg.get("method", "min_max")
        histogram_bins = int(normalize_cfg.get("histogram_bins", 4096))
        percentile = tuple(normalize_cfg.get("percentile", (1.0, 99.0)))
        smart = bool(normalize_cfg.get("smart", True))
        image = normalize_image(
            image,
            method=method,
            histogram_bins=histogram_bins,
            percentile=percentile,
            mask=mask,
            smart_minmax=smart,
        )
        meta["normalize_method"] = method
        meta["normalize_smart"] = smart

    return image.astype(np.float32), meta, mask


def _mix_images(
    image_a: np.ndarray,
    image_b: np.ndarray,
    weight_a: float,
    mix_cfg: Dict[str, Any],
    rng: np.random.Generator,
    mask: Optional[np.ndarray],
) -> np.ndarray:
    pipeline: MixPipeline = mix_cfg.get("pipeline", "normalize_then_mix")
    normalize_after_mix = bool(mix_cfg.get("normalize_after_mix", False))
    normalize_method = mix_cfg.get("normalize_method", "min_max")
    normalize_smart = bool(mix_cfg.get("normalize_smart", True))

    if pipeline == "normalize_then_mix":
        mixed = mix_normalize_then_mix(
            image_a,
            image_b,
            weight_a,
            normalize_after_mix=normalize_after_mix,
            normalize_method=normalize_method,
            mask=mask,
            normalize_smart=normalize_smart,
        )
    elif pipeline == "mix_then_normalize":
        mixed = mix_then_normalize(
            image_a,
            image_b,
            weight_a,
            normalize_method=normalize_method,
            mask=mask,
            normalize_smart=normalize_smart,
        )
    else:
        raise ValueError(f"Unknown mix pipeline: {pipeline}")

    blur_cfg = mix_cfg.get("blur", {})
    if blur_cfg.get("enabled", False):
        sigma = float(blur_cfg.get("sigma", 0.0))
        mixed = apply_gaussian_blur(mixed, sigma=sigma)

    noise_cfg = mix_cfg.get("noise", {})
    if noise_cfg.get("enabled", False):
        std = float(noise_cfg.get("std", 0.0))
        mixed = apply_gaussian_noise(mixed, std=std, rng=rng)

    if mask is not None:
        mixed = apply_circular_mask(mixed, mask)

    return mixed.astype(np.float32)


def generate_synthetic_dataset(config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate synthetic mixed Kikuchi patterns.

    Parameters
    ----------
    config:
        Configuration dictionary.

    Returns
    -------
    dict
        Summary statistics.
    """
    logger = get_logger(__name__)

    data_cfg = config.get("data", {})
    debug_cfg = config.get("debug", {})

    sample_limit = debug_cfg.get("sample_limit")
    if sample_limit is not None:
        sample_limit = int(sample_limit)

    debug = DebugSettings(
        enabled=bool(debug_cfg.get("enabled", False)),
        seed=int(debug_cfg.get("seed", 42)),
        visualize=bool(debug_cfg.get("visualize", False)),
        max_visualizations=int(debug_cfg.get("max_visualizations", 10)),
        sample_limit=sample_limit,
    )

    input_dir = Path(data_cfg.get("input_dir", "data/raw"))
    output_dir = Path(data_cfg.get("output_dir", "data/synthetic"))
    num_samples = int(data_cfg.get("num_samples", 0))
    allow_same = bool(data_cfg.get("allow_same", False))

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "A").mkdir(parents=True, exist_ok=True)
    (output_dir / "B").mkdir(parents=True, exist_ok=True)
    (output_dir / "C").mkdir(parents=True, exist_ok=True)

    if debug.visualize:
        (output_dir / "debug").mkdir(parents=True, exist_ok=True)

    if debug.enabled:
        num_samples = min(num_samples, int(debug.sample_limit or num_samples))

    if num_samples <= 0:
        raise ValueError("num_samples must be > 0.")

    rng = np.random.default_rng(debug.seed)

    paths = collect_image_paths(input_dir)
    logger.info("Found %d input images in %s", len(paths), input_dir)
    if len(paths) < 2 and not allow_same:
        raise ValueError("Need at least two input images when allow_same is False.")

    preprocess_cfg = data_cfg.get("preprocess", {})
    mix_cfg = data_cfg.get("mix", {})
    weight_cfg = mix_cfg.get("weight", {})
    weight_min = float(weight_cfg.get("min", 0.1))
    weight_max = float(weight_cfg.get("max", 0.9))
    if not (0.0 <= weight_min <= weight_max <= 1.0):
        raise ValueError("Weight bounds must satisfy 0 <= min <= max <= 1.")
    auto_crop = bool(preprocess_cfg.get("auto_crop_to_min", True))
    augment_apply_to = str(preprocess_cfg.get("augment", {}).get("apply_to", "B"))
    sum_tolerance = float(debug_cfg.get("sum_tolerance", 1e-3))
    sum_check_enabled = bool(
        debug.enabled
        and mix_cfg.get("pipeline", "normalize_then_mix") == "normalize_then_mix"
        and not mix_cfg.get("normalize_after_mix", False)
        and not mix_cfg.get("blur", {}).get("enabled", False)
        and not mix_cfg.get("noise", {}).get("enabled", False)
    )

    metadata_path = output_dir / "metadata.csv"
    output_cfg = data_cfg.get("output", {})
    output_format = str(output_cfg.get("format", "png")).lower()
    if output_format not in ("png", "tif", "tiff"):
        raise ValueError("output.format must be one of: png, tif, tiff.")
    suffix = ".tif" if output_format in ("tif", "tiff") else ".png"

    metadata_fields = [
        "sample_id",
        "source_a",
        "source_b",
        "weight_a",
        "weight_b",
        "pipeline",
        "normalize_after_mix",
        "mix_normalize_method",
        "mix_normalize_smart",
        "preprocess_normalize_method",
        "preprocess_normalize_smart",
        "mask_enabled",
        "mask_detected_a",
        "mask_detected_b",
        "mask_outside_zero_fraction_a",
        "mask_outside_zero_fraction_b",
        "augment_on",
        "augment_meta",
        "output_format",
    ]

    summary: Dict[str, Any] = {"samples": 0, "output_dir": str(output_dir)}

    config_path = output_dir / "config_used.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metadata_fields)
        writer.writeheader()

        for idx in range(num_samples):
            path_a = paths[int(rng.integers(0, len(paths)))]
            path_b = paths[int(rng.integers(0, len(paths)))]
            if not allow_same:
                while path_b == path_a:
                    path_b = paths[int(rng.integers(0, len(paths)))]

            image_a = to_float01(read_image_16bit(path_a))
            image_b = to_float01(read_image_16bit(path_b))

            image_a, image_b = _ensure_same_shape(image_a, image_b, auto_crop, rng)

            augment_a = augment_apply_to.lower() in ("a", "both")
            augment_b = augment_apply_to.lower() in ("b", "both")

            image_a, meta_a, mask_a = _preprocess_image(
                image_a, rng, preprocess_cfg, augment_a
            )
            image_b, meta_b, mask_b = _preprocess_image(
                image_b, rng, preprocess_cfg, augment_b
            )

            mask = mask_a if mask_a is not None else mask_b

            weight_a = float(rng.uniform(weight_min, weight_max))
            weight_b = 1.0 - weight_a

            mixed = _mix_images(image_a, image_b, weight_a, mix_cfg, rng, mask)

            sample_id = f"sample_{idx:06d}"
            if mixed.min() < 0 or mixed.max() > 1:
                mixed = np.clip(mixed, 0.0, 1.0)

            if sum_check_enabled:
                mix_check = (weight_a * image_a) + (weight_b * image_b)
                diff = float(np.mean(np.abs(mixed - mix_check)))
                if diff > sum_tolerance:
                    logger.warning(
                        "Sum check deviation %.6f exceeds tolerance %.6f for %s",
                        diff,
                        sum_tolerance,
                        sample_id,
                    )
            write_image_16bit(output_dir / "A" / f"{sample_id}_A{suffix}", image_a)
            write_image_16bit(output_dir / "B" / f"{sample_id}_B{suffix}", image_b)
            write_image_16bit(output_dir / "C" / f"{sample_id}_C{suffix}", mixed)

            if debug.visualize and idx < debug.max_visualizations:
                from src.generation.visualize import save_debug_panel

                debug_path = output_dir / "debug" / f"{sample_id}_panel.png"
                save_debug_panel(
                    debug_path,
                    image_a,
                    image_b,
                    mixed,
                    weight_a=weight_a,
                    weight_b=weight_b,
                    pipeline=mix_cfg.get("pipeline", "normalize_then_mix"),
                    source_a=path_a.name,
                    source_b=path_b.name,
                    mask_enabled=bool(meta_a.get("mask", {}).get("enabled", False)),
                    mask_detected_a=meta_a.get("mask", {}).get("already_masked"),
                    mask_detected_b=meta_b.get("mask", {}).get("already_masked"),
                    outside_zero_fraction_a=meta_a.get("mask", {}).get("outside_zero_fraction"),
                    outside_zero_fraction_b=meta_b.get("mask", {}).get("outside_zero_fraction"),
                    preprocess_smart=meta_a.get("normalize_smart", None),
                )

            writer.writerow(
                {
                    "sample_id": sample_id,
                    "source_a": path_a.name,
                    "source_b": path_b.name,
                    "weight_a": f"{weight_a:.6f}",
                    "weight_b": f"{weight_b:.6f}",
                    "pipeline": mix_cfg.get("pipeline", "normalize_then_mix"),
                    "normalize_after_mix": mix_cfg.get("normalize_after_mix", False),
                    "mix_normalize_method": mix_cfg.get("normalize_method", "min_max"),
                    "mix_normalize_smart": mix_cfg.get("normalize_smart", True),
                    "preprocess_normalize_method": meta_a.get("normalize_method", "none"),
                    "preprocess_normalize_smart": meta_a.get("normalize_smart", None),
                    "mask_enabled": meta_a.get("mask", {}).get("enabled", False),
                    "mask_detected_a": meta_a.get("mask", {}).get("already_masked"),
                    "mask_detected_b": meta_b.get("mask", {}).get("already_masked"),
                    "mask_outside_zero_fraction_a": meta_a.get("mask", {}).get(
                        "outside_zero_fraction"
                    ),
                    "mask_outside_zero_fraction_b": meta_b.get("mask", {}).get(
                        "outside_zero_fraction"
                    ),
                    "augment_on": augment_apply_to,
                    "augment_meta": json.dumps({"A": meta_a.get("augment"), "B": meta_b.get("augment")}),
                    "output_format": output_format,
                }
            )

            summary["samples"] += 1
            if debug.enabled and idx % 10 == 0:
                logger.debug("Generated %s", sample_id)

    logger.info("Generated %s samples in %s", summary["samples"], output_dir)
    return summary
