"""CLI wrapper for deterministic mixing-fraction inversion."""
from __future__ import annotations

import argparse
import logging
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.deterministic_mixing_inversion import run_deterministic_inversion
from src.utils.config import deep_update, load_config
from src.utils.logging import (
    collect_environment,
    get_git_commit,
    resolve_log_level,
    setup_logging,
    write_manifest,
)
from src.utils.run import resolve_run_dir


DEFAULT_CONFIG = REPO_ROOT / "configs/deterministic_inversion/inversion_default.yaml"
DEBUG_CONFIG = REPO_ROOT / "configs/deterministic_inversion/inversion_debug.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic EBSD mixing-fraction inversion.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    parser.add_argument("--mixed-dir", type=str, default=None, help="Override mixed pattern directory.")
    parser.add_argument(
        "--candidate-root",
        type=str,
        default=None,
        help="Override candidate pool root directory.",
    )
    parser.add_argument("--a-pattern", type=str, default=None, help="Regex for A-type candidate files.")
    parser.add_argument("--b-pattern", type=str, default=None, help="Regex for B-type candidate files.")
    parser.add_argument("--out_dir", type=str, default=None, help="Override output directory.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Override debug sample limit.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Override debug max candidate pairs.",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Append timestamped run tag to output directory.",
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
        help="Optional run identifier included in logs.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="key=value",
        help="Repeatable dot-path override, e.g. --set deterministic_inversion.search.grid_steps=[0.1,0.02].",
    )
    return parser.parse_args()


def _set_by_path(target: Dict[str, Any], key_path: str, value: Any) -> None:
    parts = key_path.split(".")
    cursor = target
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def _parse_set_overrides(overrides: list[str]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for raw_override in overrides:
        if "=" not in raw_override:
            raise ValueError(f"Invalid --set override '{raw_override}'. Expected key=value.")
        key_path, value_text = raw_override.split("=", 1)
        if not key_path:
            raise ValueError(f"Invalid --set override '{raw_override}'. Empty key.")
        value = yaml.safe_load(value_text)
        _set_by_path(parsed, key_path, value)
    return parsed


def load_base_config(args: argparse.Namespace) -> Dict[str, Any]:
    if args.config:
        return load_config(Path(args.config))
    if args.debug and DEBUG_CONFIG.exists():
        return load_config(DEBUG_CONFIG)
    return load_config(DEFAULT_CONFIG)


def build_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}

    if args.mixed_dir:
        overrides.setdefault("data", {})["mixed_dir"] = args.mixed_dir
    if args.candidate_root:
        overrides.setdefault("data", {}).setdefault("candidate_pool", {})["root_dir"] = args.candidate_root
    if args.a_pattern:
        overrides.setdefault("data", {}).setdefault("candidate_pool", {})["a_pattern"] = args.a_pattern
    if args.b_pattern:
        overrides.setdefault("data", {}).setdefault("candidate_pool", {})["b_pattern"] = args.b_pattern
    if args.out_dir:
        overrides.setdefault("output", {})["out_dir"] = args.out_dir
    if args.debug:
        overrides.setdefault("debug", {})["enabled"] = True
    if args.sample_limit is not None:
        overrides.setdefault("debug", {})["sample_limit"] = args.sample_limit
    if args.max_pairs is not None:
        overrides.setdefault("debug", {})["max_pairs"] = args.max_pairs

    return overrides


def main() -> None:
    args = parse_args()
    log_level = resolve_log_level(args.log_level, debug=args.debug, quiet=args.quiet)
    log_file = Path(args.log_file) if args.log_file else None
    run_id = args.run_id or args.run_tag
    logger = setup_logging("invert_mixing_fraction", level=log_level, log_file=log_file, run_id=run_id)
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    logging.getLogger("PIL").setLevel(logging.INFO)

    base_config = load_base_config(args)
    overrides = build_overrides(args)
    config = deep_update(base_config, overrides)

    try:
        set_overrides = _parse_set_overrides(args.set)
    except ValueError as exc:
        logger.error("Failed to parse --set overrides: %s", exc)
        raise
    if set_overrides:
        config = deep_update(config, set_overrides)

    if args.run_tag:
        out_dir = Path(config.get("output", {}).get("out_dir", "outputs/deterministic_inversion"))
        config.setdefault("output", {})["out_dir"] = str(resolve_run_dir(out_dir, args.run_tag))
        config.setdefault("output", {})["run_tag"] = args.run_tag

    output_dir = Path(config.get("output", {}).get("out_dir", "outputs/deterministic_inversion"))
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = output_dir / "resolved_config.yaml"
    with resolved_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    logger.info("Resolved configuration saved to %s", resolved_config_path.resolve())

    manifest: Dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": get_git_commit(REPO_ROOT),
        "args": vars(args),
        "environment": collect_environment(),
        "output_dir": str(output_dir),
        "config": config,
        "failures": [],
    }

    start_time = time.perf_counter()
    summary: Dict[str, Any] = {}
    try:
        logger.info("Starting deterministic inversion")
        summary = run_deterministic_inversion(config=config, logger=logger)
        logger.info("Deterministic inversion complete")
    except Exception as exc:
        manifest["failures"].append({"error": str(exc)})
        logger.exception("Deterministic inversion failed")
        raise
    finally:
        wall_time = time.perf_counter() - start_time
        manifest.update({"summary": summary, "timings": {"wall_time_s": wall_time}})
        write_manifest(output_dir, manifest)
        if manifest["failures"]:
            error_report = output_dir / "error_report.json"
            error_report.write_text(json.dumps(manifest["failures"], indent=2))
            logger.warning("Failures recorded in %s", error_report)
        logger.info(
            "Summary: processed=%s/%s | candidate_pairs=%s | runtime=%.2fs | out_dir=%s",
            summary.get("processed"),
            summary.get("total_mixed"),
            summary.get("candidate_pairs"),
            wall_time,
            output_dir,
        )


if __name__ == "__main__":
    main()
