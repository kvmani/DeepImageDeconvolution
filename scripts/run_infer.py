"""CLI wrapper for inference."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from src.inference.infer import run_inference
from src.utils.config import deep_update, load_config
from src.utils.logging import get_logger


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs/infer_default.yaml"
DEBUG_CONFIG = REPO_ROOT / "configs/infer_debug.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a trained model.")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--out_dir", type=str, default=None, help="Override output directory.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    return parser.parse_args()


def build_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    overrides.setdefault("inference", {})["checkpoint"] = args.checkpoint

    if args.out_dir:
        overrides.setdefault("output", {})["out_dir"] = args.out_dir

    if args.debug:
        overrides.setdefault("debug", {})["enabled"] = True

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

    logger.info("Starting inference")
    run_inference(config)
    logger.info("Inference complete")


if __name__ == "__main__":
    main()
