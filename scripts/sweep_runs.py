"""Run a grid of training configs and record a runs_index.json summary."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import itertools
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import deep_update, load_config
from src.utils.logging import resolve_log_level, setup_logging


DEFAULT_CONFIG = REPO_ROOT / "configs/train_default.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a training sweep over config grids.")
    parser.add_argument("--config", type=str, default=None, help="Base training config path.")
    parser.add_argument(
        "--grid",
        action="append",
        default=[],
        metavar="key=val1,val2",
        help="Repeatable grid override (dot path). Example: --grid train.lr=1e-4,2e-4",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="key=value",
        help="Repeatable override applied to all runs.",
    )
    parser.add_argument("--out-root", type=str, default="outputs/sweeps", help="Sweep output root.")
    parser.add_argument("--run-tag", type=str, default="sweep", help="Sweep tag prefix.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for all runs.")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level.",
    )
    parser.add_argument("--log-file", type=str, default=None, help="Optional log file path.")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging to WARNING.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without executing.")
    return parser.parse_args()


def set_by_path(target: Dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cursor = target
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor.get(part), dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def parse_overrides(values: list[str]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"Invalid override '{raw}'. Expected key=value.")
        key, value_str = raw.split("=", 1)
        if not key:
            raise ValueError(f"Invalid override '{raw}'. Expected key=value.")
        value = yaml.safe_load(value_str)
        set_by_path(overrides, key, value)
    return overrides


def parse_grid(values: list[str]) -> Dict[str, List[Any]]:
    grid: Dict[str, List[Any]] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"Invalid grid '{raw}'. Expected key=val1,val2.")
        key, value_str = raw.split("=", 1)
        parts = [part.strip() for part in value_str.split(",") if part.strip()]
        if not parts:
            raise ValueError(f"Invalid grid '{raw}'. Provide at least one value.")
        grid[key] = [yaml.safe_load(part) for part in parts]
    return grid


def iter_grid(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid.keys())
    combos = []
    for values in itertools.product(*(grid[key] for key in keys)):
        combo: Dict[str, Any] = {}
        for key, value in zip(keys, values):
            set_by_path(combo, key, value)
        combos.append(combo)
    return combos


def _hash_overrides(overrides: Dict[str, Any]) -> str:
    payload = json.dumps(overrides, sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:6]


def main() -> None:
    args = parse_args()
    log_level = resolve_log_level(args.log_level, debug=False, quiet=args.quiet)
    log_file = Path(args.log_file) if args.log_file else None
    logger = setup_logging("sweep_runs", level=log_level, log_file=log_file)

    base_config_path = Path(args.config) if args.config else DEFAULT_CONFIG
    base_config = load_config(base_config_path)
    if args.debug:
        base_config.setdefault("debug", {})["enabled"] = True

    common_overrides = parse_overrides(args.set)
    grid = parse_grid(args.grid)
    combos = iter_grid(grid)

    out_root = Path(args.out_root)
    sweep_root = out_root / args.run_tag
    sweep_root.mkdir(parents=True, exist_ok=True)
    configs_dir = sweep_root / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    runs_index_path = sweep_root / "runs_index.json"

    runs_index: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_config": str(base_config_path),
        "run_tag": args.run_tag,
        "grid": grid,
        "common_overrides": common_overrides,
        "runs": [],
    }

    for idx, grid_override in enumerate(combos, start=1):
        overrides = copy.deepcopy(common_overrides)
        overrides = deep_update(overrides, grid_override)
        run_hash = _hash_overrides(overrides)
        run_id = f"{args.run_tag}_{idx:03d}_{run_hash}"
        out_dir = sweep_root / run_id

        config = copy.deepcopy(base_config)
        config = deep_update(config, overrides)
        config.setdefault("output", {})["out_dir"] = str(out_dir)

        config_path = configs_dir / f"{run_id}.yaml"
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)

        entry: Dict[str, Any] = {
            "run_id": run_id,
            "out_dir": str(out_dir),
            "config_path": str(config_path),
            "overrides": overrides,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "status": "pending",
        }

        logger.info("Sweep run %d/%d: %s", idx, len(combos), run_id)
        if args.dry_run:
            entry["status"] = "dry_run"
        else:
            cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "run_train.py"),
                "--config",
                str(config_path),
                "--out_dir",
                str(out_dir),
                "--run-id",
                run_id,
                "--log-level",
                args.log_level,
            ]
            result = subprocess.run(cmd, check=False)
            entry["return_code"] = result.returncode
            entry["status"] = "complete" if result.returncode == 0 else "failed"

            report_path = out_dir / "report.json"
            if report_path.exists():
                entry["report_path"] = str(report_path)
                try:
                    report = json.loads(report_path.read_text())
                    entry["metrics"] = report.get("metrics")
                    entry["status"] = report.get("status", entry["status"])
                except json.JSONDecodeError:
                    entry["metrics"] = {}

        entry["finished_at"] = datetime.now().isoformat(timespec="seconds")
        runs_index["runs"].append(entry)
        runs_index_path.write_text(json.dumps(runs_index, indent=2))

    logger.info("Sweep complete: %d runs recorded in %s", len(runs_index["runs"]), runs_index_path)


if __name__ == "__main__":
    main()
