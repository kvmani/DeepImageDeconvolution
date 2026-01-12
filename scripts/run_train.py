"""CLI wrapper for training."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.train import train_model
from src.utils.config import deep_update, load_config
from src.utils.logging import get_logger
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


def main() -> None:
    args = parse_args()
    logger = get_logger(__name__)

    base_config = load_base_config(args)
    overrides = build_overrides(args)
    config = deep_update(base_config, overrides)
    if args.run_tag:
        out_dir = Path(config.get("output", {}).get("out_dir", "outputs/train_run"))
        config.setdefault("output", {})["out_dir"] = str(
            resolve_run_dir(out_dir, args.run_tag)
        )
        config.setdefault("output", {})["run_tag"] = args.run_tag

    logger.info("Starting training")
    train_model(config)
    logger.info("Training complete")


if __name__ == "__main__":
    main()
