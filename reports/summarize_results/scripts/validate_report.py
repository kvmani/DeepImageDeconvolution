"""Validate report.json schema and figure paths."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.logging import resolve_log_level, setup_logging


REQUIRED_KEYS = (
    "run_id",
    "timestamp",
    "git_commit",
    "stage",
    "dataset",
    "dataset_path",
    "config",
    "metrics",
    "figures",
    "status",
    "progress",
)


def _resolve_report_path(path_str: str, run_dir: Path, repo_root: Path) -> Path:
    """Resolve a report.json path against repo root or run directory.

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
        Resolved path candidate.
    """
    path = Path(path_str)
    if path.is_absolute():
        return path
    repo_candidate = repo_root / path
    if repo_candidate.exists():
        return repo_candidate
    return run_dir / path


def validate_report_payload(
    report: Dict[str, Any],
    run_id: str,
    run_dir: Path,
    repo_root: Path,
    strict_figures: bool = True,
) -> Tuple[List[str], List[str]]:
    """Validate report.json schema and figure references.

    Parameters
    ----------
    report:
        Parsed report.json payload.
    run_id:
        Expected run identifier.
    run_dir:
        Run directory under outputs/.
    repo_root:
        Repository root directory.
    strict_figures:
        When True, missing figure files are treated as errors.

    Returns
    -------
    tuple of list[str], list[str]
        Validation errors and missing figure keys.
    """
    errors: List[str] = []
    missing_figures: List[str] = []

    if not isinstance(report, dict):
        return ["report.json must be a JSON object."], []

    for key in REQUIRED_KEYS:
        if key not in report:
            errors.append(f"Missing required key: {key}")

    if report.get("run_id") != run_id:
        errors.append(
            f"run_id mismatch: expected {run_id}, got {report.get('run_id')}"
        )

    stage = report.get("stage")
    if stage not in {"train", "infer"}:
        errors.append("stage must be 'train' or 'infer'.")

    status = report.get("status")
    if status not in {"running", "complete", "interrupted", "failed"}:
        errors.append("status must be running, complete, interrupted, or failed.")

    progress = report.get("progress")
    if not isinstance(progress, dict):
        errors.append("progress must be an object with epoch/epochs_total/global_step.")
    else:
        for key in ("epoch", "epochs_total", "global_step"):
            if key not in progress:
                errors.append(f"progress.{key} is required.")
            elif not isinstance(progress.get(key), (int, float)):
                errors.append(f"progress.{key} must be numeric.")

    for key in ("dataset", "dataset_path", "config", "timestamp", "git_commit"):
        if key in report and not isinstance(report.get(key), str):
            errors.append(f"{key} must be a string.")
    for key in ("dataset_path", "config"):
        value = report.get(key)
        if isinstance(value, str) and Path(value).is_absolute():
            errors.append(f"{key} must be a relative path, not {value}.")

    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        errors.append("metrics must be a JSON object.")
    else:
        numeric = [
            value for value in metrics.values() if isinstance(value, (int, float))
        ]
        if not numeric:
            errors.append("metrics must contain at least one numeric value.")

    figures = report.get("figures")
    if not isinstance(figures, dict):
        errors.append("figures must be a JSON object.")
        figures = {}

    def _check_figure(key: str) -> None:
        value = figures.get(key)
        if not isinstance(value, str):
            errors.append(f"figures.{key} must be a string path.")
            return
        if Path(value).is_absolute():
            errors.append(f"figures.{key} must be a relative path, not {value}.")
            return
        resolved = _resolve_report_path(value, run_dir, repo_root)
        if not resolved.exists():
            missing_figures.append(key)
            if strict_figures:
                errors.append(f"Missing figure for {key}: {resolved}")

    if "qual_grid" not in figures:
        errors.append("figures.qual_grid is required.")
    else:
        _check_figure("qual_grid")

    if stage == "train":
        if "loss_curve" not in figures:
            errors.append("figures.loss_curve is required for training runs.")
        else:
            _check_figure("loss_curve")
    elif stage == "infer" and "loss_curve" in figures:
        _check_figure("loss_curve")

    for optional_key in ("weights_plot", "metrics_curve"):
        if optional_key in figures:
            _check_figure(optional_key)

    failure_modes = figures.get("failure_modes")
    if failure_modes is not None:
        if not isinstance(failure_modes, list) or not all(
            isinstance(item, str) for item in failure_modes
        ):
            errors.append("figures.failure_modes must be a list of strings.")

    notes = report.get("notes")
    if notes is not None:
        if not isinstance(notes, list) or not all(
            isinstance(item, str) for item in notes
        ):
            errors.append("notes must be a list of strings if provided.")

    artifacts = report.get("artifacts")
    if artifacts is not None and not isinstance(artifacts, dict):
        errors.append("artifacts must be an object if provided.")
    elif isinstance(artifacts, dict):
        for key, value in artifacts.items():
            if isinstance(value, str) and Path(value).is_absolute():
                errors.append(f"artifacts.{key} must be a relative path, not {value}.")
        if stage == "train":
            if "history" not in artifacts:
                errors.append("artifacts.history is required for training runs.")
            if "metrics_csv" not in artifacts:
                errors.append("artifacts.metrics_csv is required for training runs.")

    tracking_sample = report.get("tracking_sample")
    if tracking_sample is not None:
        if not isinstance(tracking_sample, dict):
            errors.append("tracking_sample must be an object if provided.")
        else:
            for key in ("sample_id", "epoch", "images"):
                if key not in tracking_sample:
                    errors.append(f"tracking_sample.{key} is required when tracking_sample is set.")
            images = tracking_sample.get("images")
            if images is not None and not isinstance(images, dict):
                errors.append("tracking_sample.images must be an object.")
            elif isinstance(images, dict):
                for key, value in images.items():
                    if not isinstance(value, str):
                        errors.append(f"tracking_sample.images.{key} must be a string path.")
                    elif Path(value).is_absolute():
                        errors.append(
                            f"tracking_sample.images.{key} must be a relative path, not {value}."
                        )

    return errors, missing_figures


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Validate report.json schema and figures.")
    parser.add_argument("--run-id", required=True, help="Run ID under outputs/.")
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


def main() -> None:
    """CLI entry point.

    Returns
    -------
    None
        The function exits on validation failure.
    """
    args = parse_args()
    log_level = resolve_log_level(args.log_level, debug=args.debug, quiet=args.quiet)
    log_file = Path(args.log_file) if args.log_file else None
    logger = setup_logging("validate_report", level=log_level, log_file=log_file, run_id=args.run_id)

    run_dir = REPO_ROOT / "outputs" / args.run_id
    report_path = run_dir / "report.json"
    if not report_path.exists():
        logger.error("report.json not found at %s", report_path)
        raise SystemExit(1)

    report = json.loads(report_path.read_text())
    errors, _ = validate_report_payload(
        report, args.run_id, run_dir, REPO_ROOT, strict_figures=True
    )
    if errors:
        for error in errors:
            logger.error(error)
        raise SystemExit(1)

    logger.info("report.json validated: %s", report_path)


if __name__ == "__main__":
    main()
