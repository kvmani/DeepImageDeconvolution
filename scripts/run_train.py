"""CLI wrapper for training."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.train import train_model
from src.utils.config import deep_update, load_config
from src.utils.logging import (
    collect_environment,
    get_git_commit,
    resolve_log_level,
    setup_logging,
    summarize_images,
    write_manifest,
)
from src.utils.run import resolve_run_dir


DEFAULT_CONFIG = REPO_ROOT / "configs/train_default.yaml"
DEBUG_CONFIG = REPO_ROOT / "configs/train_debug.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a dual-output U-Net model.")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config.")
    parser.add_argument("--out_dir", type=str, default=None, help="Override output directory.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override.")
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Append a timestamped run tag to the output directory.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional log file path.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging to WARNING and above.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run identifier to include in logs.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="key=value",
        help=(
            "Repeatable YAML override (dot path) applied last. "
            "Example: --set train.lr=2e-4 --set data.root_dir=./datasets "
            "CLI overrides always take precedence over YAML and built-in overrides."
        ),
    )
    return parser.parse_args()


def build_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if args.out_dir:
        overrides.setdefault("output", {})["out_dir"] = args.out_dir

    debug_overrides: Dict[str, Any] = {}
    if args.debug:
        debug_overrides["enabled"] = True
    if args.seed is not None:
        debug_overrides["seed"] = args.seed
    if debug_overrides:
        overrides["debug"] = debug_overrides

    return overrides


def load_base_config(args: argparse.Namespace) -> Dict[str, Any]:
    if args.config:
        return load_config(Path(args.config))
    if args.debug and DEBUG_CONFIG.exists():
        return load_config(DEBUG_CONFIG)
    return load_config(DEFAULT_CONFIG)


def set_by_path(target: Dict[str, Any], path: str, value: Any) -> None:
    """Set a nested dict value by dot path, creating intermediate dicts."""
    parts = path.split(".")
    cursor = target
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor.get(part), dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def parse_set_overrides(values: list[str]) -> Dict[str, Any]:
    """Parse --set CLI overrides into a nested dictionary."""
    overrides: Dict[str, Any] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"Invalid --set override '{raw}'. Expected key=value.")
        key, value_str = raw.split("=", 1)
        if not key or not value_str:
            raise ValueError(f"Invalid --set override '{raw}'. Expected key=value.")
        if any(not part for part in key.split(".")):
            raise ValueError(f"Invalid --set key '{key}'. Use dot notation like train.lr.")
        try:
            value = yaml.safe_load(value_str)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML value for --set {key}: {exc}") from exc
        set_by_path(overrides, key, value)
    return overrides


def _collect_paths(directory: Path, extensions: list[str]) -> list[Path]:
    return sorted([path for path in directory.iterdir() if path.suffix.lower() in extensions])


def main() -> None:
    args = parse_args()
    log_level = resolve_log_level(args.log_level, debug=args.debug, quiet=args.quiet)
    log_file = Path(args.log_file) if args.log_file else None
    run_id = args.run_id or args.run_tag
    logger = setup_logging("run_train", level=log_level, log_file=log_file, run_id=run_id)

    base_config = load_base_config(args)
    overrides = build_overrides(args)
    config = deep_update(base_config, overrides)
    if args.run_tag:
        out_dir = Path(config.get("output", {}).get("out_dir", "outputs/train_run"))
        config.setdefault("output", {})["out_dir"] = str(
            resolve_run_dir(out_dir, args.run_tag)
        )
        config.setdefault("output", {})["run_tag"] = args.run_tag

    try:
        set_overrides = parse_set_overrides(args.set)
    except ValueError as exc:
        logger.error("Failed to parse --set overrides: %s", exc)
        raise

    if args.set:
        logger.info("CLI overrides (--set) raw: %s", ", ".join(args.set))
        logger.info(
            "CLI overrides (--set) parsed:\n%s",
            yaml.safe_dump(set_overrides, sort_keys=False).strip(),
        )
    if set_overrides:
        config = deep_update(config, set_overrides)

    data_cfg = config.get("data", {})
    debug_cfg = config.get("debug", {})
    train_cfg = config.get("train", {})
    preprocess_cfg = data_cfg.get("preprocess", {})
    output_cfg = config.get("output", {})

    root_dir = Path(data_cfg.get("root_dir", "data/synthetic"))
    extensions = [ext.lower() for ext in data_cfg.get("extensions", [".png", ".tif", ".tiff"])]
    a_dir = root_dir / str(data_cfg.get("a_dir", "A"))
    b_dir = root_dir / str(data_cfg.get("b_dir", "B"))
    c_dir = root_dir / str(data_cfg.get("c_dir", "C"))
    out_dir = Path(output_cfg.get("out_dir", "outputs/train_run"))

    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = out_dir / "resolved_config.yaml"
    with resolved_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    logger.info("Resolved configuration saved to %s", resolved_config_path.resolve())

    manifest: Dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": get_git_commit(REPO_ROOT),
        "args": vars(args),
        "environment": collect_environment(),
        "output_dir": str(out_dir),
        "config": config,
        "failures": [],
    }

    start_time = time.perf_counter()
    summary: Dict[str, Any] = {}
    try:
        if not root_dir.exists():
            logger.error("Data root directory does not exist: %s", root_dir)
            raise FileNotFoundError(f"Data root directory not found: {root_dir}")

        a_paths = _collect_paths(a_dir, extensions) if a_dir.exists() else []
        b_paths = _collect_paths(b_dir, extensions) if b_dir.exists() else []
        c_paths = _collect_paths(c_dir, extensions) if c_dir.exists() else []
        if not a_paths or not b_paths or not c_paths:
            logger.error("Missing training inputs: A=%d B=%d C=%d", len(a_paths), len(b_paths), len(c_paths))
            raise ValueError("Missing paired training inputs under root_dir.")

        a_summary = summarize_images(a_paths)
        c_summary = summarize_images(c_paths)
        logger.info("Resolved root dir: %s", root_dir.resolve())
        logger.info("Output dir: %s", out_dir.resolve())
        logger.info(
            "Pre-flight: A=%d B=%d C=%d | A sizes=%s->%s | C sizes=%s->%s | A dtypes=%s | C dtypes=%s",
            len(a_paths),
            len(b_paths),
            len(c_paths),
            a_summary.get("min_size"),
            a_summary.get("max_size"),
            c_summary.get("min_size"),
            c_summary.get("max_size"),
            a_summary.get("sample_dtypes"),
            c_summary.get("sample_dtypes"),
        )
        logger.info(
            "Config: batch_size=%s | epochs=%s | val_split=%.3f | seed=%s | debug=%s",
            train_cfg.get("batch_size", 0),
            train_cfg.get("epochs", 0),
            float(data_cfg.get("val_split", 0.0)),
            debug_cfg.get("seed", 42),
            debug_cfg.get("enabled", False),
        )
        logger.info(
            "Preprocess: mask=%s | normalize=%s | augment=%s | crop=%s",
            preprocess_cfg.get("mask", {}).get("enabled", True),
            preprocess_cfg.get("normalize", {}).get("enabled", False),
            preprocess_cfg.get("augment", {}).get("enabled", False),
            preprocess_cfg.get("crop", {}).get("enabled", False),
        )

        logger.info("Starting training")
        summary = train_model(config)
        logger.info("Training complete")
    except Exception as exc:
        manifest["failures"].append({"error": str(exc)})
        logger.exception("Training failed")
        raise
    finally:
        duration = time.perf_counter() - start_time
        manifest.update(
            {
                "summary": summary,
                "timings": {"wall_time_s": duration},
            }
        )
        write_manifest(out_dir, manifest)
        if manifest["failures"]:
            error_report = out_dir / "error_report.json"
            error_report.write_text(json.dumps(manifest["failures"], indent=2))
            logger.warning("Failures recorded in %s", error_report)
        logger.info(
            "Summary: train_samples=%s | val_samples=%s | epochs=%s | runtime=%.2fs | out_dir=%s",
            summary.get("train_samples"),
            summary.get("val_samples"),
            summary.get("epochs"),
            duration,
            out_dir,
        )


if __name__ == "__main__":
    main()
