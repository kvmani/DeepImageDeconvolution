"""Write report.json for the lab meeting demo."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.logging import collect_environment, resolve_log_level, setup_logging, write_manifest
from src.utils.reporting import safe_relpath, write_report_json


REQUIRED_FIGURES = ("weight_sweep_grid", "strategy_compare_grid")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Write lab meeting report.json.")
    parser.add_argument("--run-id", required=True, help="Run ID under outputs/.")
    parser.add_argument(
        "--out-root",
        default="outputs",
        help="Root output directory (relative to repo root by default).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level.",
    )
    parser.add_argument("--log-file", default=None, help="Optional log file path.")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging to WARNING.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def _resolve_run_dir(run_id: str, out_root: str) -> Path:
    """Resolve the run directory under outputs/.

    Parameters
    ----------
    run_id:
        Run identifier.
    out_root:
        Output root directory.

    Returns
    -------
    pathlib.Path
        Resolved run directory.
    """
    root = Path(out_root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    return root / run_id


def _coerce_weights(weights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Coerce weight entries into the report schema format.

    Parameters
    ----------
    weights:
        Weight metadata list from demo_metadata.json.

    Returns
    -------
    list[dict]
        Cleaned weight entries.
    """
    cleaned: List[Dict[str, Any]] = []
    for item in weights:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        if not isinstance(path, str):
            continue
        cleaned.append(
            {
                "x": float(item.get("x", 0.0)),
                "y": float(item.get("y", 0.0)),
                "C_path": safe_relpath(path, REPO_ROOT),
            }
        )
    return cleaned


def _build_strategies(metadata: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Build strategies payload and notes from metadata.

    Parameters
    ----------
    metadata:
        Parsed demo metadata.

    Returns
    -------
    tuple[dict, list[str]]
        Strategies payload and notes.
    """
    weight_sweep = metadata.get("weight_sweep", {})
    sweep_strategy = str(weight_sweep.get("strategy", "mix_then_normalize"))
    sweep_weights = weight_sweep.get("weights", [])
    mix_weights: List[Dict[str, Any]] = []
    norm_weights: List[Dict[str, Any]] = []
    if sweep_strategy == "mix_then_normalize":
        mix_weights = _coerce_weights(sweep_weights)
    elif sweep_strategy == "normalize_then_mix":
        norm_weights = _coerce_weights(sweep_weights)

    strategy_compare = metadata.get("strategy_compare", {})
    compare_norm = strategy_compare.get("normalize_then_mix")
    compare_mix = strategy_compare.get("mix_then_normalize")

    normalize_example = None
    if isinstance(compare_norm, str):
        normalize_example = safe_relpath(compare_norm, REPO_ROOT)
    elif norm_weights:
        normalize_example = norm_weights[0]["C_path"]

    mix_enabled = bool(mix_weights)
    norm_enabled = bool(norm_weights or normalize_example)

    strategies: Dict[str, Any] = {
        "mix_then_normalize": {
            "enabled": mix_enabled,
            "weights": mix_weights,
        },
        "normalize_then_mix": {
            "enabled": norm_enabled,
            "example_C_path": normalize_example or "",
        },
    }
    if norm_weights:
        strategies["normalize_then_mix"]["weights"] = norm_weights

    notes = metadata.get("notes", [])
    return strategies, notes if isinstance(notes, list) else []


def _build_figures(run_dir: Path) -> Dict[str, str]:
    """Build figure path payload.

    Parameters
    ----------
    run_dir:
        Output run directory.

    Returns
    -------
    dict
        Figure paths keyed by figure name.
    """
    figures: Dict[str, str] = {}
    for key in REQUIRED_FIGURES:
        figure_path = run_dir / "figures" / f"{key}.png"
        figures[key] = safe_relpath(figure_path, REPO_ROOT)
    return figures


def main() -> None:
    """CLI entry point.

    Returns
    -------
    None
    """
    args = parse_args()
    start_time = time.perf_counter()

    run_dir = _resolve_run_dir(args.run_id, args.out_root)
    log_level = resolve_log_level(args.log_level, debug=args.debug, quiet=args.quiet)
    log_file = Path(args.log_file) if args.log_file else run_dir / "logs" / "write_report.log"
    logger = setup_logging("lab_meeting_demo_report", level=log_level, log_file=log_file, run_id=args.run_id)

    meta_path = run_dir / "demo_metadata.json"
    if not meta_path.exists():
        logger.error("demo_metadata.json not found at %s", meta_path)
        raise SystemExit(1)

    metadata = json.loads(meta_path.read_text())
    inputs = metadata.get("inputs", {})
    if not isinstance(inputs, dict):
        logger.error("inputs missing from demo_metadata.json")
        raise SystemExit(1)
    if not inputs.get("A_source") or not inputs.get("B_source"):
        logger.error("inputs.A_source and inputs.B_source are required.")
        raise SystemExit(1)

    strategies, notes = _build_strategies(metadata)
    if len(notes) < 2:
        notes = notes + ["Generated demo report.", "Figures assembled from outputs/"]
        notes = notes[:2]

    figures = _build_figures(run_dir)

    report = {
        "run_id": metadata.get("run_id", args.run_id),
        "timestamp": metadata.get("timestamp", "unknown"),
        "git_commit": metadata.get("git_commit", "unknown"),
        "stage": "demo",
        "inputs": {
            "A_path": safe_relpath(inputs.get("A_source"), REPO_ROOT),
            "B_path": safe_relpath(inputs.get("B_source"), REPO_ROOT),
        },
        "strategies": strategies,
        "figures": figures,
        "notes": notes,
    }

    report_path = write_report_json(run_dir, report)
    logger.info("Wrote report.json: %s", report_path)

    elapsed = time.perf_counter() - start_time
    manifest = {
        "run_id": args.run_id,
        "stage": "demo",
        "script": "write_report",
        "status": "success",
        "outputs": {"report": safe_relpath(report_path, REPO_ROOT)},
        "environment": collect_environment(),
        "timing": {"elapsed_s": round(elapsed, 3)},
    }
    manifest_path = write_manifest(run_dir, manifest, filename="manifest.write_report.json")
    logger.info("Wrote manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
