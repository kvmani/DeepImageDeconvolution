"""CLI for synthetic Kikuchi pattern generation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.generation.dataset import generate_synthetic_dataset
from src.utils.config import deep_update, load_config
from src.utils.logging import get_logger


DEFAULT_CONFIG = REPO_ROOT / "configs/default.yaml"
DEBUG_CONFIG = REPO_ROOT / "configs/debug.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic mixed Kikuchi patterns from pure inputs."
    )
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config.")
    parser.add_argument("--input-dir", type=str, default=None, help="Override input directory.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory.")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples.")
    parser.add_argument("--pipeline", type=str, default=None, help="Mixing pipeline name.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override.")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save debug visualization panels.",
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Disable circular masking for inputs.",
    )
    parser.add_argument(
        "--no-smart-normalize",
        action="store_true",
        help="Disable smart (mask-aware) normalization.",
    )
    return parser.parse_args()


def build_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    data_overrides: Dict[str, Any] = {}
    mix_overrides: Dict[str, Any] = {}

    if args.input_dir:
        data_overrides["input_dir"] = args.input_dir
    if args.output_dir:
        data_overrides["output_dir"] = args.output_dir
    if args.num_samples is not None:
        data_overrides["num_samples"] = args.num_samples
    if args.pipeline:
        mix_overrides["pipeline"] = args.pipeline

    if mix_overrides:
        data_overrides["mix"] = mix_overrides
    if data_overrides:
        overrides["data"] = data_overrides

    debug_overrides: Dict[str, Any] = {}
    if args.debug:
        debug_overrides["enabled"] = True
    if args.seed is not None:
        debug_overrides["seed"] = args.seed
    if args.visualize:
        debug_overrides["visualize"] = True
    if debug_overrides:
        overrides["debug"] = debug_overrides

    if args.no_mask:
        overrides.setdefault("data", {}).setdefault("preprocess", {}).setdefault(
            "mask", {}
        )["enabled"] = False
    if args.no_smart_normalize:
        overrides.setdefault("data", {}).setdefault("preprocess", {}).setdefault(
            "normalize", {}
        )["smart"] = False

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

    logger.info("Starting synthetic data generation")
    generate_synthetic_dataset(config)
    logger.info("Synthetic data generation complete")


if __name__ == "__main__":
    main()
