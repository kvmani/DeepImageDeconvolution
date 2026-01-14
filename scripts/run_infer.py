"""CLI wrapper for inference."""
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

from src.inference.infer import run_inference
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


DEFAULT_CONFIG = REPO_ROOT / "configs/infer_default.yaml"
DEBUG_CONFIG = REPO_ROOT / "configs/infer_debug.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a trained model.")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--out_dir", type=str, default=None, help="Override output directory.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
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
    log_level = resolve_log_level(args.log_level, debug=args.debug, quiet=args.quiet)
    log_file = Path(args.log_file) if args.log_file else None
    run_id = args.run_id or args.run_tag
    logger = setup_logging("run_infer", level=log_level, log_file=log_file, run_id=run_id)

    base_config = load_base_config(args)
    overrides = build_overrides(args)
    config = deep_update(base_config, overrides)
    if args.run_tag:
        out_dir = Path(config.get("output", {}).get("out_dir", "outputs/infer_run"))
        config.setdefault("output", {})["out_dir"] = str(
            resolve_run_dir(out_dir, args.run_tag)
        )
        config.setdefault("output", {})["run_tag"] = args.run_tag

    data_cfg = config.get("data", {})
    output_cfg = config.get("output", {})
    debug_cfg = config.get("debug", {})

    mixed_dir = Path(data_cfg.get("mixed_dir", "data/synthetic/C"))
    out_dir = Path(output_cfg.get("out_dir", "outputs/infer_run"))

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
        if not mixed_dir.exists():
            logger.error("Mixed input directory does not exist: %s", mixed_dir)
            raise FileNotFoundError(f"Mixed input directory not found: {mixed_dir}")

        extensions = [ext.lower() for ext in data_cfg.get("extensions", [".png", ".tif", ".tiff"])]
        mixed_paths = sorted([path for path in mixed_dir.iterdir() if path.suffix.lower() in extensions])
        if not mixed_paths:
            logger.error("No mixed inputs found in %s", mixed_dir)
            raise ValueError("No mixed inputs found.")

        mixed_summary = summarize_images(mixed_paths)
        logger.info("Resolved mixed dir: %s", mixed_dir.resolve())
        logger.info("Output dir: %s", out_dir.resolve())
        logger.info(
            "Pre-flight: mixed=%d | extensions=%s | size range=%s->%s | dtypes=%s",
            len(mixed_paths),
            mixed_summary.get("extensions"),
            mixed_summary.get("min_size"),
            mixed_summary.get("max_size"),
            mixed_summary.get("sample_dtypes"),
        )
        logger.info(
            "Debug: enabled=%s | seed=%s | sample_limit=%s",
            debug_cfg.get("enabled", False),
            debug_cfg.get("seed", 42),
            debug_cfg.get("sample_limit"),
        )

        logger.info("Starting inference")
        summary = run_inference(config)
        logger.info("Inference complete")
    except Exception as exc:
        manifest["failures"].append({"error": str(exc)})
        logger.exception("Inference failed")
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
            "Summary: processed=%s | failed=%s | runtime=%.2fs | out_dir=%s",
            summary.get("processed"),
            summary.get("failed"),
            duration,
            out_dir,
        )


if __name__ == "__main__":
    main()
