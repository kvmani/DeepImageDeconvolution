"""Utilities for the mixing experiments notebook.

This module keeps `notebooks/mixing_experiments.ipynb` short by providing
helpers to load/preprocess patterns, apply mixing pipelines consistent with the
synthetic data generator, and visualize results.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

from src.generation.mix import MixPipeline, mix_normalize_then_mix, mix_then_normalize
from src.preprocessing.mask import apply_circular_mask, build_mask_with_metadata
from src.preprocessing.normalise import NormalizeMethod, normalize_image
from src.preprocessing.transforms import (
    apply_gaussian_blur,
    apply_gaussian_noise,
    center_crop_to_min,
)
from src.utils.io import read_image_float01_with_meta


@dataclass(frozen=True)
class MixingExperimentConfig:
    """Configuration for an interactive A/B -> C mixing experiment."""

    a_path: Path
    b_path: Path

    allow_non_16bit: bool = True
    auto_crop_to_min: bool = True

    mask_enabled: bool = True
    detect_existing_mask: bool = True
    zero_tolerance: float = 5e-4
    outside_zero_fraction: float = 0.98

    preprocess_normalize_enabled: bool = True
    preprocess_normalize_method: NormalizeMethod = "min_max"
    preprocess_smart_normalize: bool = True
    preprocess_histogram_bins: int = 1024
    preprocess_percentile: Tuple[float, float] = (1.0, 99.0)

    pipeline: MixPipeline = "normalize_then_mix"
    weight_a: float = 0.6
    mix_normalize_after_mix: bool = False
    mix_normalize_method: NormalizeMethod = "max"
    mix_normalize_smart: bool = True

    blur_sigma: float = 0.0
    noise_std: float = 0.0
    seed: int = 0


@dataclass(frozen=True)
class MixingExperimentResult:
    """Output of a mixing experiment run."""

    image_a: np.ndarray
    image_b: np.ndarray
    image_c: np.ndarray
    diff: np.ndarray
    mask: np.ndarray | None
    meta: Dict[str, object]


def _compute_mask_and_meta(
    cfg: MixingExperimentConfig, image_a: np.ndarray, image_b: np.ndarray
) -> tuple[np.ndarray, Dict[str, object]]:
    meta: Dict[str, object] = {}
    mask, _ = build_mask_with_metadata(
        image_a,
        detect_existing=False,
        zero_tolerance=cfg.zero_tolerance,
        outside_zero_fraction=cfg.outside_zero_fraction,
    )
    if not cfg.detect_existing_mask:
        return mask, meta

    _, meta_a = build_mask_with_metadata(
        image_a,
        mask=mask,
        detect_existing=True,
        zero_tolerance=cfg.zero_tolerance,
        outside_zero_fraction=cfg.outside_zero_fraction,
    )
    _, meta_b = build_mask_with_metadata(
        image_b,
        mask=mask,
        detect_existing=True,
        zero_tolerance=cfg.zero_tolerance,
        outside_zero_fraction=cfg.outside_zero_fraction,
    )
    meta.update(
        {
            "mask_detected_a": meta_a.get("already_masked"),
            "mask_detected_b": meta_b.get("already_masked"),
            "mask_outside_zero_fraction_a": meta_a.get("outside_zero_fraction"),
            "mask_outside_zero_fraction_b": meta_b.get("outside_zero_fraction"),
        }
    )
    return mask, meta


def _preprocess(
    image: np.ndarray, cfg: MixingExperimentConfig, mask: np.ndarray | None
) -> np.ndarray:
    if cfg.mask_enabled and mask is not None:
        image = apply_circular_mask(image, mask)
    if cfg.preprocess_normalize_enabled:
        image = normalize_image(
            image,
            method=cfg.preprocess_normalize_method,
            histogram_bins=cfg.preprocess_histogram_bins,
            percentile=cfg.preprocess_percentile,
            mask=mask,
            smart_minmax=cfg.preprocess_smart_normalize,
        )
    return image.astype(np.float32)


def run_mixing_experiment(cfg: MixingExperimentConfig) -> MixingExperimentResult:
    """Run a single A/B mixing experiment and return images for plotting."""
    rng = np.random.default_rng(cfg.seed)

    image_a, meta_a = read_image_float01_with_meta(
        cfg.a_path, allow_non_16bit=cfg.allow_non_16bit
    )
    image_b, meta_b = read_image_float01_with_meta(
        cfg.b_path, allow_non_16bit=cfg.allow_non_16bit
    )

    if cfg.auto_crop_to_min:
        image_a, image_b = center_crop_to_min(image_a, image_b)
    elif image_a.shape != image_b.shape:
        raise ValueError(f"A/B shape mismatch: {image_a.shape} vs {image_b.shape}")

    mask = None
    mask_meta: Dict[str, object] = {}
    if cfg.mask_enabled:
        mask, mask_meta = _compute_mask_and_meta(cfg, image_a, image_b)

    image_a = _preprocess(image_a, cfg, mask)
    image_b = _preprocess(image_b, cfg, mask)

    weight_a = float(cfg.weight_a)
    if not 0.0 <= weight_a <= 1.0:
        raise ValueError("weight_a must be in [0, 1].")
    weight_b = 1.0 - weight_a

    if cfg.pipeline == "normalize_then_mix":
        image_c = mix_normalize_then_mix(
            image_a,
            image_b,
            weight_a=weight_a,
            normalize_after_mix=cfg.mix_normalize_after_mix,
            normalize_method=cfg.mix_normalize_method,
            mask=mask,
            normalize_smart=cfg.mix_normalize_smart,
        )
    elif cfg.pipeline == "mix_then_normalize":
        image_c = mix_then_normalize(
            image_a,
            image_b,
            weight_a=weight_a,
            normalize_method=cfg.mix_normalize_method,
            mask=mask,
            normalize_smart=cfg.mix_normalize_smart,
        )
    else:
        raise ValueError(f"Unknown pipeline: {cfg.pipeline}")

    if cfg.blur_sigma > 0:
        image_c = apply_gaussian_blur(image_c, sigma=float(cfg.blur_sigma))
    if cfg.noise_std > 0:
        image_c = apply_gaussian_noise(image_c, std=float(cfg.noise_std), rng=rng)

    if cfg.mask_enabled and mask is not None:
        image_c = apply_circular_mask(image_c, mask)

    image_c = np.clip(image_c, 0.0, 1.0).astype(np.float32)
    diff = image_c - (weight_a * image_a + weight_b * image_b)

    meta: Dict[str, object] = {}
    meta.update({"a": meta_a, "b": meta_b})
    meta.update(mask_meta)
    meta.update(
        {
            "weight_a": weight_a,
            "weight_b": weight_b,
            "pipeline": cfg.pipeline,
            "mix_normalize_after_mix": cfg.mix_normalize_after_mix,
            "mix_normalize_method": cfg.mix_normalize_method,
            "mix_normalize_smart": cfg.mix_normalize_smart,
        }
    )
    return MixingExperimentResult(
        image_a=image_a,
        image_b=image_b,
        image_c=image_c,
        diff=diff.astype(np.float32),
        mask=mask,
        meta=meta,
    )


def plot_mixing_panel(result: MixingExperimentResult, cfg: MixingExperimentConfig) -> None:
    """Plot a compact 2x2 panel: A, B, C, and the sum-rule diff map."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    a = result.image_a
    b = result.image_b
    c = result.image_c
    diff = result.diff

    weight_a = float(result.meta["weight_a"])
    weight_b = float(result.meta["weight_b"])
    pipeline = str(result.meta["pipeline"])

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    axes[0].imshow(a, cmap="gray")
    axes[0].set_title(f"A\nmin={a.min():.3f} max={a.max():.3f}")
    axes[0].axis("off")

    axes[1].imshow(b, cmap="gray")
    axes[1].set_title(f"B\nmin={b.min():.3f} max={b.max():.3f}")
    axes[1].axis("off")

    axes[2].imshow(c, cmap="gray")
    axes[2].set_title(
        "C (synthetic)\n" f"wA={weight_a:.3f} wB={weight_b:.3f} | {pipeline}"
    )
    axes[2].axis("off")

    im = axes[3].imshow(diff, cmap="coolwarm")
    axes[3].set_title("C - (wA*A + wB*B)")
    axes[3].axis("off")
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    if cfg.mask_enabled and result.mask is not None:
        height, width = a.shape
        center = ((width - 1) / 2.0, (height - 1) / 2.0)
        radius = min(height, width) / 2.0
        for ax in (axes[0], axes[1], axes[2]):
            ax.add_patch(Circle(center, radius, fill=False, color="yellow", linewidth=1.2))

    fig.suptitle("Synthetic Mixing Preview", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    plt.close(fig)


def _iter_image_paths(directory: Path, extensions: Iterable[str]) -> list[Path]:
    exts = {ext.lower() for ext in extensions}
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in exts)


def show_interactive_mixer(cfg: MixingExperimentConfig) -> None:
    """Launch an interactive widget UI for exploring mixing configurations."""
    import ipywidgets as widgets
    from IPython.display import clear_output, display

    normalize_methods: tuple[NormalizeMethod, ...] = (
        "min_max",
        "max",
        "percentile",
        "histogram_equalization",
    )

    candidate_dir = cfg.a_path.parent
    candidates = _iter_image_paths(
        candidate_dir, extensions=(".bmp", ".png", ".tif", ".tiff")
    )
    if cfg.a_path not in candidates:
        candidates = [cfg.a_path, *candidates]
    if cfg.b_path not in candidates:
        candidates = [*candidates, cfg.b_path]

    a_path = widgets.Dropdown(options=candidates, value=cfg.a_path, description="A")
    b_path = widgets.Dropdown(options=candidates, value=cfg.b_path, description="B")

    pipeline = widgets.Dropdown(
        options=("normalize_then_mix", "mix_then_normalize"),
        value=cfg.pipeline,
        description="pipeline",
    )
    weight_a = widgets.FloatSlider(
        min=0.0, max=1.0, step=0.02, value=cfg.weight_a, description="wA"
    )

    mask_enabled = widgets.Checkbox(value=cfg.mask_enabled, description="mask")
    detect_existing_mask = widgets.Checkbox(
        value=cfg.detect_existing_mask, description="detect mask"
    )
    auto_crop_to_min = widgets.Checkbox(value=cfg.auto_crop_to_min, description="auto-crop")
    allow_non_16bit = widgets.Checkbox(
        value=cfg.allow_non_16bit, description="allow non-16bit"
    )

    pre_norm_enabled = widgets.Checkbox(
        value=cfg.preprocess_normalize_enabled, description="pre-normalize"
    )
    pre_norm_method = widgets.Dropdown(
        options=normalize_methods,
        value=cfg.preprocess_normalize_method,
        description="pre norm",
    )
    pre_norm_smart = widgets.Checkbox(
        value=cfg.preprocess_smart_normalize, description="pre smart"
    )

    mix_norm_after = widgets.Checkbox(
        value=cfg.mix_normalize_after_mix, description="norm after mix"
    )
    mix_norm_method = widgets.Dropdown(
        options=normalize_methods, value=cfg.mix_normalize_method, description="mix norm"
    )
    mix_norm_smart = widgets.Checkbox(value=cfg.mix_normalize_smart, description="mix smart")

    blur_sigma = widgets.FloatSlider(
        min=0.0, max=3.0, step=0.2, value=cfg.blur_sigma, description="blur"
    )
    noise_std = widgets.FloatSlider(
        min=0.0, max=0.1, step=0.005, value=cfg.noise_std, description="noise"
    )
    seed = widgets.IntText(value=cfg.seed, description="seed")

    output = widgets.Output()

    def _render(_: object = None) -> None:
        with output:
            clear_output(wait=True)
            cfg_run = replace(
                cfg,
                a_path=Path(a_path.value),
                b_path=Path(b_path.value),
                pipeline=pipeline.value,
                weight_a=float(weight_a.value),
                mask_enabled=bool(mask_enabled.value),
                detect_existing_mask=bool(detect_existing_mask.value),
                auto_crop_to_min=bool(auto_crop_to_min.value),
                allow_non_16bit=bool(allow_non_16bit.value),
                preprocess_normalize_enabled=bool(pre_norm_enabled.value),
                preprocess_normalize_method=pre_norm_method.value,
                preprocess_smart_normalize=bool(pre_norm_smart.value),
                mix_normalize_after_mix=bool(mix_norm_after.value),
                mix_normalize_method=mix_norm_method.value,
                mix_normalize_smart=bool(mix_norm_smart.value),
                blur_sigma=float(blur_sigma.value),
                noise_std=float(noise_std.value),
                seed=int(seed.value),
            )
            result = run_mixing_experiment(cfg_run)
            plot_mixing_panel(result, cfg_run)
            a_dtype = str(result.meta.get("a", {}).get("dtype"))
            b_dtype = str(result.meta.get("b", {}).get("dtype"))
            print(f"A dtype={a_dtype} | B dtype={b_dtype}")

    controls = widgets.VBox(
        [
            widgets.HTML("<b>Inputs</b>"),
            widgets.HBox([a_path, b_path]),
            widgets.HTML("<b>Preprocess</b>"),
            widgets.HBox([mask_enabled, detect_existing_mask, auto_crop_to_min, allow_non_16bit]),
            widgets.HBox([pre_norm_enabled, pre_norm_method, pre_norm_smart]),
            widgets.HTML("<b>Mix</b>"),
            widgets.HBox([pipeline, weight_a]),
            widgets.HBox([mix_norm_after, mix_norm_method, mix_norm_smart]),
            widgets.HTML("<b>Effects</b>"),
            widgets.HBox([blur_sigma, noise_std, seed]),
        ]
    )

    for widget in (
        a_path,
        b_path,
        pipeline,
        weight_a,
        mask_enabled,
        detect_existing_mask,
        auto_crop_to_min,
        allow_non_16bit,
        pre_norm_enabled,
        pre_norm_method,
        pre_norm_smart,
        mix_norm_after,
        mix_norm_method,
        mix_norm_smart,
        blur_sigma,
        noise_std,
        seed,
    ):
        widget.observe(_render, names="value")

    display(widgets.VBox([controls, output]))
    _render()
