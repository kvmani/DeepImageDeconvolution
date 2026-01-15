"""Validate report.json schema for the lab meeting demo."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.logging import collect_environment, resolve_log_level, setup_logging, write_manifest
from src.utils.reporting import safe_relpath


REQUIRED_KEYS = (
    "run_id",
    "timestamp",
    "git_commit",
    "stage",
    "inputs",
    "strategies",
    "figures",
    "notes",
)

REQUIRED_FIGURES = ("weight_sweep_grid", "strategy_compare_grid")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Validate lab meeting report.json.")
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


def _resolve_report_path(path_str: str, run_dir: Path, repo_root: Path) -> Path:
    """Resolve a report path against repo root or run directory.

    Parameters
    ----------
    path_str:
        Path string from report.json.
    run_dir:
        Run directory under outputs/.
    repo_root:
        Repository root directory.

    Returns
    -------
    pathlib.Path
        Resolved path.
    """
    path = Path(path_str)
    if path.is_absolute():
        return path
    repo_candidate = repo_root / path
    if repo_candidate.exists():
        return repo_candidate
    return run_dir / path


def _validate_path(
    errors: List[str],
    label: str,
    value: Any,
    run_dir: Path,
    repo_root: Path,
) -> None:
    """Validate a report path and record errors.

    Parameters
    ----------
    errors:
        Error list to append to.
    label:
        Label for error messages.
    value:
        Path value to validate.
    run_dir:
        Run directory under outputs/.
    repo_root:
        Repository root directory.
    """
    if not isinstance(value, str):
        errors.append(f"{label} must be a string path.")
        return
    if Path(value).is_absolute():
        errors.append(f"{label} must be a relative path, not {value}.")
        return
    resolved = _resolve_report_path(value, run_dir, repo_root)
    if not resolved.exists():
        errors.append(f"Missing file for {label}: {resolved}")


def validate_report_payload(
    report: Dict[str, Any],
    run_id: str,
    run_dir: Path,
    repo_root: Path,
) -> List[str]:
    """Validate report.json payload.

    Parameters
    ----------
    report:
        Parsed report.json payload.
    run_id:
        Expected run ID.
    run_dir:
        Run directory under outputs/.
    repo_root:
        Repository root path.

    Returns
    -------
    list[str]
        Validation errors.
    """
    errors: List[str] = []

    if not isinstance(report, dict):
        return ["report.json must be a JSON object."]

    for key in REQUIRED_KEYS:
        if key not in report:
            errors.append(f"Missing required key: {key}")

    if report.get("run_id") != run_id:
        errors.append(f"run_id mismatch: expected {run_id}, got {report.get('run_id')}")

    if report.get("stage") != "demo":
        errors.append("stage must be 'demo'.")

    inputs = report.get("inputs")
    if not isinstance(inputs, dict):
        errors.append("inputs must be a JSON object.")
    else:
        _validate_path(errors, "inputs.A_path", inputs.get("A_path"), run_dir, repo_root)
        _validate_path(errors, "inputs.B_path", inputs.get("B_path"), run_dir, repo_root)

    strategies = report.get("strategies")
    if not isinstance(strategies, dict):
        errors.append("strategies must be a JSON object.")
    else:
        mix_strategy = strategies.get("mix_then_normalize")
        if not isinstance(mix_strategy, dict):
            errors.append("strategies.mix_then_normalize must be an object.")
        else:
            if not isinstance(mix_strategy.get("enabled"), bool):
                errors.append("mix_then_normalize.enabled must be a boolean.")
            weights = mix_strategy.get("weights")
            if not isinstance(weights, list):
                errors.append("strategies.mix_then_normalize.weights must be a list.")
            else:
                if mix_strategy.get("enabled") and not weights:
                    errors.append("mix_then_normalize enabled but weights list is empty.")
                for idx, item in enumerate(weights):
                    if not isinstance(item, dict):
                        errors.append(f"mix_then_normalize.weights[{idx}] must be an object.")
                        continue
                    if not isinstance(item.get("x"), (int, float)):
                        errors.append(f"mix_then_normalize.weights[{idx}].x must be numeric.")
                    if not isinstance(item.get("y"), (int, float)):
                        errors.append(f"mix_then_normalize.weights[{idx}].y must be numeric.")
                    _validate_path(
                        errors,
                        f"mix_then_normalize.weights[{idx}].C_path",
                        item.get("C_path"),
                        run_dir,
                        repo_root,
                    )

        norm_strategy = strategies.get("normalize_then_mix")
        if not isinstance(norm_strategy, dict):
            errors.append("strategies.normalize_then_mix must be an object.")
        else:
            if not isinstance(norm_strategy.get("enabled"), bool):
                errors.append("normalize_then_mix.enabled must be a boolean.")
            if norm_strategy.get("enabled"):
                _validate_path(
                    errors,
                    "normalize_then_mix.example_C_path",
                    norm_strategy.get("example_C_path"),
                    run_dir,
                    repo_root,
                )
            if "weights" in norm_strategy:
                weights = norm_strategy.get("weights")
                if not isinstance(weights, list):
                    errors.append("normalize_then_mix.weights must be a list if provided.")
                else:
                    for idx, item in enumerate(weights):
                        if not isinstance(item, dict):
                            errors.append(
                                f"normalize_then_mix.weights[{idx}] must be an object."
                            )
                            continue
                        _validate_path(
                            errors,
                            f"normalize_then_mix.weights[{idx}].C_path",
                            item.get("C_path"),
                            run_dir,
                            repo_root,
                        )

    figures = report.get("figures")
    if not isinstance(figures, dict):
        errors.append("figures must be a JSON object.")
    else:
        for key in REQUIRED_FIGURES:
            _validate_path(errors, f"figures.{key}", figures.get(key), run_dir, repo_root)

    notes = report.get("notes")
    if not isinstance(notes, list) or not all(isinstance(item, str) for item in notes):
        errors.append("notes must be a list of strings.")
    elif len(notes) < 2:
        errors.append("notes must contain at least two entries.")

    return errors


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
    log_file = Path(args.log_file) if args.log_file else run_dir / "logs" / "validate_report.log"
    logger = setup_logging(
        "lab_meeting_demo_validate",
        level=log_level,
        log_file=log_file,
        run_id=args.run_id,
    )

    report_path = run_dir / "report.json"
    if not report_path.exists():
        logger.error("report.json not found at %s", report_path)
        raise SystemExit(1)

    report = json.loads(report_path.read_text())
    errors = validate_report_payload(report, args.run_id, run_dir, REPO_ROOT)
    if errors:
        for error in errors:
            logger.error(error)
        raise SystemExit(1)

    logger.info("report.json validated: %s", report_path)

    elapsed = time.perf_counter() - start_time
    manifest = {
        "run_id": args.run_id,
        "stage": "demo",
        "script": "validate_report",
        "status": "success",
        "outputs": {"report": safe_relpath(report_path, REPO_ROOT)},
        "environment": collect_environment(),
        "timing": {"elapsed_s": round(elapsed, 3)},
    }
    manifest_path = write_manifest(run_dir, manifest, filename="manifest.validate_report.json")
    logger.info("Wrote manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
