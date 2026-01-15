"""Generate slide-ready figures for the lab meeting demo."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.io import read_image_16bit, to_float01
from src.utils.logging import collect_environment, resolve_log_level, setup_logging, write_manifest
from src.utils.reporting import safe_relpath


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Generate figures for lab meeting demo.")
    parser.add_argument("--run-id", required=True, help="Run ID under outputs/.")
    parser.add_argument(
        "--out-root",
        default="outputs",
        help="Root output directory (relative to repo root by default).",
    )
    parser.add_argument("--strict", action="store_true", help="Fail on missing assets.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level.",
    )
    parser.add_argument("--log-file", default=None, help="Optional log file path.")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging to WARNING.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def _resolve_path(path_str: str) -> Path:
    """Resolve a report path to an absolute path.

    Parameters
    ----------
    path_str:
        Path string from metadata/report.

    Returns
    -------
    pathlib.Path
        Absolute path.
    """
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _load_image(path: Path) -> np.ndarray:
    """Load a 16-bit image as float32 in [0, 1].

    Parameters
    ----------
    path:
        Image path.

    Returns
    -------
    numpy.ndarray
        Float image in [0, 1].
    """
    image = read_image_16bit(path)
    return to_float01(image)


def _strategy_label(strategy: str) -> str:
    """Return a display label for a strategy.

    Parameters
    ----------
    strategy:
        Strategy name.

    Returns
    -------
    str
        Display label.
    """
    if strategy == "mix_then_normalize":
        return "mix->normalize"
    return "normalize->mix"


def _save_grid(
    panels: List[Dict[str, object]],
    title: str,
    output_path: Path,
    nrows: int,
    ncols: int,
) -> None:
    """Save a grid of images with titles.

    Parameters
    ----------
    panels:
        List of dicts with image/title fields.
    title:
        Figure title.
    output_path:
        Output path for the figure.
    nrows:
        Number of rows.
    ncols:
        Number of columns.
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 8))
    axes = np.array(axes).reshape(-1)
    for idx, axis in enumerate(axes):
        if idx >= len(panels):
            axis.axis("off")
            continue
        panel = panels[idx]
        axis.imshow(panel["image"], cmap="gray")
        axis.set_title(panel["title"], fontsize=16)
        axis.axis("off")
    fig.suptitle(title, fontsize=18)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    """CLI entry point.

    Returns
    -------
    None
    """
    args = parse_args()
    start_time = time.perf_counter()

    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root
    run_dir = out_root / args.run_id
    log_level = resolve_log_level(args.log_level, debug=args.debug, quiet=args.quiet)
    log_file = Path(args.log_file) if args.log_file else run_dir / "logs" / "make_figures.log"
    logger = setup_logging("lab_meeting_demo_figures", level=log_level, log_file=log_file, run_id=args.run_id)

    meta_path = run_dir / "demo_metadata.json"
    if not meta_path.exists():
        logger.error("demo_metadata.json not found at %s", meta_path)
        raise SystemExit(1)

    metadata = json.loads(meta_path.read_text())
    inputs = metadata.get("inputs", {})
    weight_sweep = metadata.get("weight_sweep", {})
    strategy_compare = metadata.get("strategy_compare", {})

    required_input_keys = ("A_demo", "B_demo")
    for key in required_input_keys:
        if key not in inputs:
            logger.error("Missing inputs.%s in demo_metadata.json", key)
            raise SystemExit(1)

    image_a = _load_image(_resolve_path(inputs["A_demo"]))
    image_b = _load_image(_resolve_path(inputs["B_demo"]))

    sweep_strategy = str(weight_sweep.get("strategy", "mix_then_normalize"))
    sweep_weights = weight_sweep.get("weights", [])
    if not isinstance(sweep_weights, list) or not sweep_weights:
        message = "weight_sweep.weights is missing or empty."
        if args.strict:
            logger.error(message)
            raise SystemExit(1)
        logger.warning(message)
        sweep_weights = []

    sweep_panels: List[Dict[str, object]] = [
        {"title": "A", "image": image_a},
        {"title": "B", "image": image_b},
    ]
    for item in sweep_weights:
        c_path = item.get("path")
        if not isinstance(c_path, str):
            continue
        c_img = _load_image(_resolve_path(c_path))
        weight_a = float(item.get("x", 0.0))
        weight_b = float(item.get("y", 0.0))
        sweep_panels.append(
            {
                "title": f"C (x/y={weight_a:.2f}/{weight_b:.2f})",
                "image": c_img,
            }
        )

    title = f"Weight sweep ({_strategy_label(sweep_strategy)})"
    weight_grid_path = run_dir / "figures" / "weight_sweep_grid.png"
    _save_grid(sweep_panels, title, weight_grid_path, nrows=2, ncols=3)
    logger.info("Saved weight sweep grid: %s", weight_grid_path)

    compare_weight = strategy_compare.get("weight", {})
    compare_a = float(compare_weight.get("x", 0.5))
    compare_b = float(compare_weight.get("y", 0.5))
    compare_panels = [
        {"title": "A", "image": image_a},
        {"title": "B", "image": image_b},
    ]

    for strategy_name in ("normalize_then_mix", "mix_then_normalize"):
        c_path = strategy_compare.get(strategy_name)
        if not isinstance(c_path, str):
            continue
        c_img = _load_image(_resolve_path(c_path))
        compare_panels.append(
            {
                "title": f"{_strategy_label(strategy_name)}\n(x/y={compare_a:.2f}/{compare_b:.2f})",
                "image": c_img,
            }
        )

    compare_title = "Strategy comparison"
    compare_path = run_dir / "figures" / "strategy_compare_grid.png"
    _save_grid(compare_panels, compare_title, compare_path, nrows=1, ncols=4)
    logger.info("Saved strategy comparison grid: %s", compare_path)

    elapsed = time.perf_counter() - start_time
    manifest = {
        "run_id": args.run_id,
        "stage": "demo",
        "script": "make_figures",
        "status": "success",
        "outputs": {
            "weight_sweep_grid": safe_relpath(weight_grid_path, REPO_ROOT),
            "strategy_compare_grid": safe_relpath(compare_path, REPO_ROOT),
        },
        "environment": collect_environment(),
        "timing": {"elapsed_s": round(elapsed, 3)},
    }
    manifest_path = write_manifest(run_dir, manifest, filename="manifest.make_figures.json")
    logger.info("Wrote manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
