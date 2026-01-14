"""CLI for synthetic Kikuchi pattern generation."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.generation.dataset import generate_synthetic_dataset
from src.utils.config import deep_update, load_config
from src.utils.io import collect_image_paths
from src.utils.logging import (
    collect_environment,
    get_git_commit,
    resolve_log_level,
    setup_logging,
    summarize_images,
    write_manifest,
)
from src.utils.run import resolve_run_dir


DEFAULT_CONFIG = REPO_ROOT / "configs/default.yaml"
DEBUG_CONFIG = REPO_ROOT / "configs/debug.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic mixed Kikuchi patterns from pure inputs."
    )
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config.")
    parser.add_argument("--input-dir", type=str, default=None, help="Override input directory.")
    parser.add_argument(
        "--recursive-input",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Recursively scan input-dir for images (overrides data.input_recursive).",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory.")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples.")
    parser.add_argument("--pipeline", type=str, default=None, help="Mixing pipeline name.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override.")
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Append a timestamped run tag to the output directory.",
    )
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
    return parser.parse_args()


def build_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    data_overrides: Dict[str, Any] = {}
    mix_overrides: Dict[str, Any] = {}

    if args.input_dir:
        data_overrides["input_dir"] = args.input_dir
    if args.recursive_input is not None:
        data_overrides["input_recursive"] = bool(args.recursive_input)
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
    log_level = resolve_log_level(args.log_level, debug=args.debug, quiet=args.quiet)
    log_file = Path(args.log_file) if args.log_file else None
    run_id = args.run_id or args.run_tag
    logger = setup_logging("generate_data", level=log_level, log_file=log_file, run_id=run_id)

    base_config = load_base_config(args)
    overrides = build_overrides(args)
    config = deep_update(base_config, overrides)
    if args.run_tag:
        output_dir = Path(config.get("data", {}).get("output_dir", "data/synthetic"))
        config.setdefault("data", {})["output_dir"] = str(
            resolve_run_dir(output_dir, args.run_tag)
        )
        config.setdefault("output", {})["run_tag"] = args.run_tag

    data_cfg = config.get("data", {})
    mix_cfg = data_cfg.get("mix", {})
    preprocess_cfg = data_cfg.get("preprocess", {})
    debug_cfg = config.get("debug", {})
    input_dir = Path(data_cfg.get("input_dir", "data/raw"))
    output_dir = Path(data_cfg.get("output_dir", "data/synthetic"))

    manifest: Dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": get_git_commit(REPO_ROOT),
        "args": vars(args),
        "environment": collect_environment(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "config": config,
        "failures": [],
    }

    start_time = time.perf_counter()
    summary: Dict[str, Any] = {}
    try:
        if not input_dir.exists():
            logger.error("Input directory does not exist: %s", input_dir)
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        input_recursive = bool(data_cfg.get("input_recursive", False))
        input_paths = collect_image_paths(input_dir, recursive=input_recursive)
        if not input_paths:
            logger.error("No input images found in %s", input_dir)
            raise ValueError("No input images found.")

        input_summary = summarize_images(input_paths)
        logger.info("Resolved input dir: %s", input_dir.resolve())
        logger.info("Resolved output dir: %s", output_dir.resolve())
        logger.info(
            "Pre-flight: %d inputs | extensions=%s | size range=%s -> %s | dtypes=%s | recursive=%s",
            len(input_paths),
            input_summary.get("extensions"),
            input_summary.get("min_size"),
            input_summary.get("max_size"),
            input_summary.get("sample_dtypes"),
            input_recursive,
        )
        logger.info(
            "Config: num_samples=%s | pipeline=%s | weight_range=[%.3f, %.3f] | allow_same=%s",
            data_cfg.get("num_samples", 0),
            mix_cfg.get("pipeline", "normalize_then_mix"),
            float(mix_cfg.get("weight", {}).get("min", 0.1)),
            float(mix_cfg.get("weight", {}).get("max", 0.9)),
            data_cfg.get("allow_same", False),
        )
        logger.info(
            "Preprocess: mask=%s | normalize=%s | crop=%s | augment=%s",
            preprocess_cfg.get("mask", {}).get("enabled", True),
            preprocess_cfg.get("normalize", {}).get("enabled", True),
            preprocess_cfg.get("crop", {}).get("enabled", False),
            preprocess_cfg.get("augment", {}).get("enabled", False),
        )
        logger.info(
            "Debug: enabled=%s | seed=%s | visualize=%s",
            debug_cfg.get("enabled", False),
            debug_cfg.get("seed", 42),
            debug_cfg.get("visualize", False),
        )

        logger.info("Starting synthetic data generation")
        summary = generate_synthetic_dataset(config)
        logger.info("Synthetic data generation complete")
    except Exception as exc:
        manifest["failures"].append({"error": str(exc)})
        logger.exception("Synthetic data generation failed")
        raise
    finally:
        duration = time.perf_counter() - start_time
        output_counts = {
            "A": len(list((output_dir / "A").glob("*.png"))) if (output_dir / "A").exists() else 0,
            "B": len(list((output_dir / "B").glob("*.png"))) if (output_dir / "B").exists() else 0,
            "C": len(list((output_dir / "C").glob("*.png"))) if (output_dir / "C").exists() else 0,
        }
        manifest.update(
            {
                "summary": summary,
                "timings": {"wall_time_s": duration},
                "output_counts": output_counts,
            }
        )
        write_manifest(output_dir, manifest)
        if manifest["failures"]:
            error_report = output_dir / "error_report.json"
            error_report.write_text(json.dumps(manifest["failures"], indent=2))
            logger.warning("Failures recorded in %s", error_report)
        logger.info(
            "Summary: samples=%s | outputs=%s | runtime=%.2fs | output_dir=%s",
            summary.get("samples", 0),
            output_counts,
            duration,
            output_dir,
        )


if __name__ == "__main__":
    main()
