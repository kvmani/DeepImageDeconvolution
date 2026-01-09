"""Visualization helpers for debug mode."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def save_debug_panel(
    path: Path,
    image_a: np.ndarray,
    image_b: np.ndarray,
    image_c: np.ndarray,
    weight_a: float,
    weight_b: float,
    pipeline: str,
    source_a: str,
    source_b: str,
    mask_enabled: bool = False,
    mask_detected_a: bool | None = None,
    mask_detected_b: bool | None = None,
    outside_zero_fraction_a: float | None = None,
    outside_zero_fraction_b: float | None = None,
    preprocess_smart: bool | None = None,
) -> None:
    """Save a debug panel with annotations.

    Parameters
    ----------
    path:
        Output path for the figure.
    image_a:
        Preprocessed image A.
    image_b:
        Preprocessed image B.
    image_c:
        Mixed image C.
    weight_a:
        Weight for image A.
    weight_b:
        Weight for image B.
    pipeline:
        Mixing pipeline name.
    source_a:
        Source filename for A.
    source_b:
        Source filename for B.
    mask_enabled:
        Whether circular masking is enabled.
    mask_detected_a:
        Whether A was detected as already masked.
    mask_detected_b:
        Whether B was detected as already masked.
    outside_zero_fraction_a:
        Measured outside-zero fraction for A.
    outside_zero_fraction_b:
        Measured outside-zero fraction for B.
    preprocess_smart:
        Whether smart normalization was enabled during preprocessing.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for debug visualization. Install matplotlib or disable visualize."
        ) from exc

    diff = image_c - (weight_a * image_a + weight_b * image_b)

    mask_line_a = "mask: off"
    mask_line_b = "mask: off"
    if mask_enabled:
        mask_line_a = "mask: on"
        mask_line_b = "mask: on"
        if mask_detected_a is not None:
            mask_line_a += f" | detected={mask_detected_a}"
        if mask_detected_b is not None:
            mask_line_b += f" | detected={mask_detected_b}"
        if outside_zero_fraction_a is not None:
            mask_line_a += f" | outside_zero={outside_zero_fraction_a:.3f}"
        if outside_zero_fraction_b is not None:
            mask_line_b += f" | outside_zero={outside_zero_fraction_b:.3f}"

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    axes[0].imshow(image_a, cmap="gray")
    axes[0].set_title(
        f"A: {source_a}\nmin={image_a.min():.3f} max={image_a.max():.3f}\n{mask_line_a}"
    )
    axes[0].axis("off")

    axes[1].imshow(image_b, cmap="gray")
    axes[1].set_title(
        f"B: {source_b}\nmin={image_b.min():.3f} max={image_b.max():.3f}\n{mask_line_b}"
    )
    axes[1].axis("off")

    axes[2].imshow(image_c, cmap="gray")
    c_title = (
        "C (mixed)\n"
        f"wA={weight_a:.3f} wB={weight_b:.3f} | {pipeline}\n"
        f"min={image_c.min():.3f} max={image_c.max():.3f}"
    )
    if preprocess_smart is not None:
        c_title += f"\npre_norm_smart={preprocess_smart}"
    axes[2].set_title(c_title)
    axes[2].axis("off")

    im = axes[3].imshow(diff, cmap="coolwarm")
    axes[3].set_title("C - (wA*A + wB*B)")
    axes[3].axis("off")
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    if mask_enabled:
        try:
            from matplotlib.patches import Circle
        except ImportError:
            Circle = None
        if Circle is not None:
            height, width = image_a.shape
            center = ((width - 1) / 2.0, (height - 1) / 2.0)
            radius = min(height, width) / 2.0
            for ax in axes[:3]:
                circle = Circle(center, radius, fill=False, color="yellow", linewidth=1.2)
                ax.add_patch(circle)

    fig.suptitle("Synthetic Mix Debug Panel", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(path, dpi=150)
    plt.close(fig)
