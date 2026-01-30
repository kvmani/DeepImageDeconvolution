"""Prepare experimental images into a canonical training/eval format."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from src.utils.logging import (
    resolve_log_level,
    setup_logging,
)
from src.preprocessing.prepare import prepare_experimental_dataset

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare experimental images into a canonical training/eval format."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw/Double Pattern Data",
        help="Root folder with experimental images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/Double Pattern Data",
        help="Destination root for prepared outputs.",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="png",
        choices=("png",),
        help="Output image format for prepared data (16-bit PNG only).",
    )
    parser.add_argument(
        "--output-bit-depth",
        type=str,
        default="16",
        choices=("16",),
        help="Output bit depth (always scales to 16-bit).",
    )
    parser.add_argument(
        "--recursive-input",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recursively scan input-dir for images (default: true).",
    )
    parser.add_argument(
        "--grayscale-method",
        type=str,
        default="luma",
        choices=("luma", "average"),
        help="Grayscale conversion for multi-channel inputs.",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help="Optional path for manifest JSON (defaults to output-dir/manifest.json).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing prepared files.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Limit the number of input files processed.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (limits samples and enables verbose logging).",
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


def main() -> None:
    args = parse_args()
    log_level = resolve_log_level(args.log_level, debug=args.debug, quiet=args.quiet)
    log_file = Path(args.log_file) if args.log_file else None
    logger = setup_logging("prepare_experimental_data", level=log_level, log_file=log_file, run_id=args.run_id)
    config: Dict[str, Any] = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "output_format": args.output_format,
        "output_bit_depth": args.output_bit_depth,
        "recursive_input": args.recursive_input,
        "grayscale_method": args.grayscale_method,
        "manifest_path": args.manifest_path,
        "overwrite": args.overwrite,
        "sample_limit": args.sample_limit,
        "debug": args.debug,
    }

    prepare_experimental_dataset(config, logger)


if __name__ == "__main__":
    main()
