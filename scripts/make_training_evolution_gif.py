"""Build an animated GIF showing training evolution over epochs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoMinorLocator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import deep_update, load_config
from src.utils.logging import resolve_log_level, setup_logging


DEFAULT_CONFIG = REPO_ROOT / "configs/training_evolution_gif.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an animated GIF showing per-epoch prediction evolution plus metric curves."
        )
    )
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config.")
    parser.add_argument("--run_dir", type=str, default=None, help="Run directory override.")
    parser.add_argument("--out_gif", type=str, default=None, help="Output GIF path override.")
    parser.add_argument("--sample-id", type=str, default=None, help="Sample ID to track.")
    parser.add_argument("--sample-index", type=int, default=None, help="Sample index fallback.")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging to WARNING and above.",
    )
    return parser.parse_args()


def _as_path(value: Any, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path


def _coerce_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.floating, np.integer)):
        value_f = float(value)
        if math.isfinite(value_f):
            return value_f
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            value_f = float(text)
        except ValueError:
            return None
        if math.isfinite(value_f):
            return value_f
    return None


def load_history(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"History file not found: {path}")
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = []
            for row in reader:
                converted: Dict[str, Any] = {}
                for key, value in row.items():
                    if key == "epoch":
                        try:
                            converted[key] = int(float(value)) if value else None
                        except (ValueError, TypeError):
                            converted[key] = None
                    else:
                        converted[key] = _coerce_number(value)
                rows.append(converted)
            return rows
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"History JSON should be a list: {path}")
    normalized: List[Dict[str, Any]] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        converted = dict(row)
        epoch_val = _coerce_number(row.get("epoch"))
        converted["epoch"] = int(epoch_val) if epoch_val is not None else None
        normalized.append(converted)
    return normalized


def load_image_log(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Image log not found: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Image log JSON should be a list: {path}")
    return data


def _resolve_sample_id(entries: List[Dict[str, Any]], sample_id: Optional[str], sample_index: int) -> str:
    if sample_id:
        return sample_id
    for entry in entries:
        samples = entry.get("samples") or []
        if not samples:
            continue
        index = max(sample_index, 0)
        if index < len(samples):
            candidate = samples[index].get("sample_id")
            if candidate:
                return str(candidate)
    raise ValueError("No samples found in image_log entries to infer a sample_id.")


def _select_sample(
    entry: Dict[str, Any],
    sample_id: str,
) -> Optional[Dict[str, Any]]:
    for sample in entry.get("samples") or []:
        if str(sample.get("sample_id")) == sample_id:
            return sample
    return None


def _format_header(
    fields: List[Dict[str, Any]],
    data: Dict[str, Any],
    separator: str,
    skip_missing: bool,
) -> str:
    parts: List[str] = []
    for field in fields:
        key = field.get("key")
        if not key:
            continue
        label = field.get("label", key)
        raw_value = data.get(key)
        value = raw_value
        if isinstance(raw_value, float) and not math.isfinite(raw_value):
            value = None
        fmt = field.get("format")
        if value is None:
            if skip_missing:
                continue
            rendered = "NA"
        else:
            if fmt:
                try:
                    rendered = format(value, fmt)
                except (ValueError, TypeError):
                    rendered = str(value)
            else:
                rendered = str(value)
        parts.append(f"{label} {rendered}")
    return separator.join(parts)


def _series_from_history(
    history: Iterable[Dict[str, Any]],
    key: str,
    max_epoch: Optional[int] = None,
) -> Tuple[List[int], List[float]]:
    xs: List[int] = []
    ys: List[float] = []
    for row in history:
        epoch = row.get("epoch")
        if epoch is None:
            continue
        if max_epoch is not None and epoch > max_epoch:
            continue
        value = _coerce_number(row.get(key))
        if value is None:
            continue
        xs.append(int(epoch))
        ys.append(float(value))
    return xs, ys


def _apply_style(style_cfg: Dict[str, Any]) -> None:
    base_font = style_cfg.get("base_font_size", 10)
    plt.rcParams.update(
        {
            "font.family": style_cfg.get("font_family", "DejaVu Sans"),
            "font.size": base_font,
            "axes.titlesize": style_cfg.get("title_size", base_font + 1),
            "axes.labelsize": style_cfg.get("label_size", base_font - 1),
            "xtick.labelsize": style_cfg.get("tick_size", base_font - 2),
            "ytick.labelsize": style_cfg.get("tick_size", base_font - 2),
            "legend.fontsize": style_cfg.get("legend_size", base_font - 2),
            "axes.linewidth": style_cfg.get("axis_line_width", 0.8),
            "figure.facecolor": style_cfg.get("figure_facecolor", "#ffffff"),
            "axes.facecolor": style_cfg.get("axes_facecolor", "#ffffff"),
        }
    )


def _draw_plot(
    ax: plt.Axes,
    plot_cfg: Dict[str, Any],
    history: List[Dict[str, Any]],
    history_by_epoch: Dict[int, Dict[str, Any]],
    current_epoch: int,
    plots_cfg: Dict[str, Any],
    style_cfg: Dict[str, Any],
) -> None:
    show_history = plots_cfg.get("show_history", "up_to_epoch")
    max_epoch = None if show_history == "full" else current_epoch
    line_width = style_cfg.get("line_width", 1.6)
    series_cfgs = plot_cfg.get("series", [])
    any_data = False

    for series in series_cfgs:
        key = series.get("key")
        if not key:
            continue
        xs, ys = _series_from_history(history, key, max_epoch=max_epoch)
        if not xs:
            continue
        any_data = True
        ax.plot(
            xs,
            ys,
            label=series.get("label", key),
            color=series.get("color"),
            linestyle=series.get("linestyle", "-"),
            linewidth=series.get("linewidth", line_width),
            alpha=series.get("alpha", 1.0),
        )
        if plots_cfg.get("highlight_epoch", True):
            current_row = history_by_epoch.get(current_epoch, {})
            current_val = _coerce_number(current_row.get(key))
            if current_val is not None:
                marker_cfg = plots_cfg.get("marker", {})
                ax.scatter(
                    [current_epoch],
                    [current_val],
                    s=marker_cfg.get("size", 24),
                    color=series.get("color"),
                    alpha=marker_cfg.get("alpha", 0.9),
                    zorder=4,
                )

    if not any_data:
        ax.text(
            0.5,
            0.5,
            "No data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=style_cfg.get("label_size", 9),
            color="#666666",
        )

    if plots_cfg.get("highlight_epoch", True):
        highlight_cfg = plots_cfg.get("highlight", {})
        ax.axvline(
            current_epoch,
            color=highlight_cfg.get("color", "#111111"),
            linestyle=highlight_cfg.get("linestyle", ":"),
            linewidth=highlight_cfg.get("linewidth", 1.0),
            alpha=highlight_cfg.get("alpha", 0.6),
        )

    ax.set_title(plot_cfg.get("title", ""))
    ax.set_xlabel(plots_cfg.get("x_label", "Epoch"))
    ax.set_ylabel(plot_cfg.get("y_label", ""))
    ax.grid(
        True,
        which="major",
        alpha=style_cfg.get("grid", {}).get("major_alpha", 0.35),
        linewidth=style_cfg.get("grid", {}).get("major_linewidth", 0.7),
    )
    ax.grid(
        True,
        which="minor",
        alpha=style_cfg.get("grid", {}).get("minor_alpha", 0.15),
        linewidth=style_cfg.get("grid", {}).get("minor_linewidth", 0.5),
    )
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", direction="out")
    if any_data:
        ax.legend(frameon=False, loc="best")


def _load_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        if img.mode not in ("L", "RGB"):
            img = img.convert("L")
        return np.array(img)


def build_frame(
    entry: Dict[str, Any],
    sample: Dict[str, Any],
    history: List[Dict[str, Any]],
    history_by_epoch: Dict[int, Dict[str, Any]],
    config: Dict[str, Any],
    image_log_dir: Path,
) -> Image.Image:
    layout_cfg = config.get("layout", {})
    style_cfg = config.get("style", {})
    plots_cfg = config.get("plots", {})

    _apply_style(style_cfg)

    image_order = layout_cfg.get("image_order", [])
    fig_size = layout_cfg.get("figure_size", [13.5, 8.0])
    image_row_ratio = layout_cfg.get("image_row_ratio", 1.0)
    plot_row_ratio = layout_cfg.get("plot_row_ratio", 1.0)
    plot_rows, plot_cols = plots_cfg.get("grid", [2, 2])
    plot_rows = int(plot_rows)
    plot_cols = int(plot_cols)

    fig = plt.figure(figsize=(fig_size[0], fig_size[1]), dpi=config.get("output", {}).get("dpi", 150))

    outer = GridSpec(
        nrows=2,
        ncols=1,
        height_ratios=[image_row_ratio, plot_row_ratio * plot_rows],
        hspace=layout_cfg.get("plot_hspace", 0.28),
        figure=fig,
    )
    image_gs = outer[0].subgridspec(1, max(len(image_order), 1), wspace=layout_cfg.get("image_wspace", 0.06))
    plots_gs = outer[1].subgridspec(
        plot_rows,
        plot_cols,
        hspace=layout_cfg.get("plot_hspace", 0.28),
        wspace=layout_cfg.get("plot_wspace", 0.22),
    )

    images = sample.get("images", {})
    for idx, label in enumerate(image_order):
        ax = fig.add_subplot(image_gs[0, idx])
        rel_path = images.get(label)
        if rel_path:
            path = image_log_dir / rel_path
            if path.exists():
                img_arr = _load_image(path)
                if img_arr.ndim == 2:
                    ax.imshow(img_arr, cmap="gray", vmin=0, vmax=255)
                else:
                    ax.imshow(img_arr)
            else:
                ax.text(0.5, 0.5, "Missing", ha="center", va="center", fontsize=9, color="#666666")
        else:
            ax.text(0.5, 0.5, "Missing", ha="center", va="center", fontsize=9, color="#666666")
        ax.set_title(label, fontsize=layout_cfg.get("image_title_size", 10), pad=layout_cfg.get("image_title_pad", 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

    history_epoch = entry.get("epoch")
    for plot_idx, plot_cfg in enumerate(plots_cfg.get("items", [])):
        if plot_idx >= plot_rows * plot_cols:
            break
        row = plot_idx // plot_cols
        col = plot_idx % plot_cols
        ax = fig.add_subplot(plots_gs[row, col])
        _draw_plot(
            ax=ax,
            plot_cfg=plot_cfg,
            history=history,
            history_by_epoch=history_by_epoch,
            current_epoch=int(history_epoch) if history_epoch is not None else 0,
            plots_cfg=plots_cfg,
            style_cfg=style_cfg,
        )

    margins = layout_cfg.get("margins", {})
    fig.subplots_adjust(
        left=margins.get("left", 0.04),
        right=margins.get("right", 0.985),
        bottom=margins.get("bottom", 0.05),
        top=margins.get("top", 0.92),
    )

    header_cfg = layout_cfg.get("header", {})
    if header_cfg.get("enabled", True):
        header_fields = header_cfg.get("fields", [])
        context: Dict[str, Any] = {
            "epoch": entry.get("epoch"),
            "split": entry.get("split"),
            "sample_id": sample.get("sample_id"),
        }
        context.update(history_by_epoch.get(int(history_epoch), {}) if history_epoch is not None else {})
        context.update(sample.get("metrics") or {})
        header_text = _format_header(
            fields=header_fields,
            data=context,
            separator=header_cfg.get("separator", " | "),
            skip_missing=header_cfg.get("skip_missing", True),
        )
        if header_text:
            fig.suptitle(
                header_text,
                fontsize=header_cfg.get("font_size", 12),
                fontweight=header_cfg.get("weight", "semibold"),
                y=0.98,
            )

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buffer = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buffer = buffer.reshape(height, width, 3)
    frame = Image.fromarray(buffer)
    plt.close(fig)
    return frame


def main() -> None:
    args = parse_args()
    log_level = resolve_log_level(args.log_level, quiet=args.quiet)
    logger = setup_logging("make_training_evolution_gif", level=log_level)

    config_path = Path(args.config) if args.config else DEFAULT_CONFIG
    config = load_config(config_path) if config_path.exists() else {}
    overrides: Dict[str, Any] = {}
    if args.run_dir:
        overrides["run_dir"] = args.run_dir
    if args.out_gif:
        overrides.setdefault("output", {})["gif"] = args.out_gif
    if args.sample_id:
        overrides.setdefault("sample", {})["id"] = args.sample_id
    if args.sample_index is not None:
        overrides.setdefault("sample", {})["index"] = args.sample_index
    if overrides:
        config = deep_update(config, overrides)

    run_dir = Path(config.get("run_dir", "."))
    input_cfg = config.get("input", {})
    output_cfg = config.get("output", {})
    sample_cfg = config.get("sample", {})

    image_log_path = _as_path(input_cfg.get("image_log", "monitoring/image_log.json"), run_dir)
    history_path = _as_path(input_cfg.get("history", "history.json"), run_dir)
    output_gif_path = _as_path(output_cfg.get("gif", "monitoring/training_evolution.gif"), run_dir)
    output_gif_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Using run_dir: %s", run_dir.resolve())
    logger.info("Image log: %s", image_log_path.resolve())
    logger.info("History: %s", history_path.resolve())
    logger.info("Output GIF: %s", output_gif_path.resolve())

    entries = load_image_log(image_log_path)
    if not entries:
        raise ValueError("Image log has no entries.")
    def _epoch_key(item: Dict[str, Any]) -> int:
        raw = item.get("epoch")
        value = _coerce_number(raw)
        return int(value) if value is not None else 0

    entries_sorted = sorted(entries, key=_epoch_key)

    history = load_history(history_path)
    history_sorted = sorted(
        [row for row in history if row.get("epoch") is not None],
        key=lambda item: item.get("epoch", 0),
    )
    history_by_epoch = {int(row["epoch"]): row for row in history_sorted}

    sample_id = _resolve_sample_id(
        entries_sorted,
        sample_cfg.get("id"),
        int(sample_cfg.get("index", 0)),
    )
    missing_policy = str(sample_cfg.get("missing_policy", "skip")).lower()
    logger.info("Tracking sample_id: %s", sample_id)

    frames: List[Image.Image] = []
    image_log_dir = image_log_path.parent

    for entry in entries_sorted:
        epoch = entry.get("epoch")
        if epoch is None:
            continue
        sample = _select_sample(entry, sample_id)
        if sample is None:
            message = f"Sample {sample_id} not found in epoch {epoch}."
            if missing_policy == "error":
                raise ValueError(message)
            logger.warning("%s Skipping epoch.", message)
            continue
        frame = build_frame(
            entry=entry,
            sample=sample,
            history=history_sorted,
            history_by_epoch=history_by_epoch,
            config=config,
            image_log_dir=image_log_dir,
        )
        if output_cfg.get("save_frames", False):
            frame_dir = _as_path(output_cfg.get("frame_dir", "monitoring/frames"), run_dir)
            frame_dir.mkdir(parents=True, exist_ok=True)
            frame_path = frame_dir / f"frame_{int(epoch):04d}.png"
            frame.save(frame_path)
        frames.append(frame)

    if not frames:
        raise ValueError("No frames generated; check sample_id and image_log contents.")

    fps = float(output_cfg.get("fps", 2))
    if fps <= 0:
        fps = 2.0
    duration_ms = int(1000 / fps)
    loop = int(output_cfg.get("loop", 0))
    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=loop,
    )
    logger.info("GIF saved: %s (frames=%d, fps=%.2f)", output_gif_path, len(frames), fps)


if __name__ == "__main__":
    main()
