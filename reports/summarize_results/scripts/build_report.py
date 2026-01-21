"""Build latest.json and local figures for the summarize_results deck."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reports.summarize_results.scripts.validate_report import validate_report_payload
from scripts.compare_runs import build_comparison_rows, write_comparison_table
from src.utils.logging import resolve_log_level, setup_logging
from src.utils.reporting import normalize_path, safe_relpath


REPORT_ROOT = REPO_ROOT / "reports" / "summarize_results"


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


def _create_placeholder(path: Path, label: str) -> None:
    """Create a simple placeholder image for missing figures.

    Parameters
    ----------
    path:
        Output PNG path.
    label:
        Label to embed in the placeholder.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (640, 360), color=(240, 240, 240))
    draw = ImageDraw.Draw(image)
    text = f"MISSING FIGURE\n{label}"
    draw.multiline_text((20, 20), text, fill=(80, 80, 80))
    image.save(path)


def _extract_next_steps(todo_path: Path, limit: int = 5) -> List[str]:
    """Extract top incomplete tasks from the todo list.

    Parameters
    ----------
    todo_path:
        Path to todo_list.md.
    limit:
        Maximum number of tasks to return.

    Returns
    -------
    list of str
        Extracted task summaries.
    """
    if not todo_path.exists():
        return []
    steps: List[str] = []
    for line in todo_path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("- [ ] "):
            steps.append(stripped[6:].strip())
            if len(steps) >= limit:
                break
    return steps


def _copy_figure(
    key: str,
    source: str,
    run_dir: Path,
    figures_dir: Path,
    strict: bool,
) -> Tuple[str, str]:
    """Copy a figure into the local figures directory.

    Parameters
    ----------
    key:
        Figure key (loss_curve, qual_grid, etc.).
    source:
        Source path string from report.json.
    run_dir:
        Run directory under outputs/.
    figures_dir:
        Destination directory for copied figures.
    strict:
        When True, missing figures raise an error.

    Returns
    -------
    tuple of str
        Source path and relative destination path.
    """
    source_path = _resolve_report_path(source, run_dir, REPO_ROOT)
    suffix = Path(source).suffix or ".png"
    dest_path = figures_dir / f"{key}{suffix}"
    if source_path.exists():
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest_path)
    elif strict:
        raise FileNotFoundError(f"Missing figure for {key}: {source_path}")
    else:
        _create_placeholder(dest_path, key)
    rel_dest = normalize_path(dest_path.relative_to(REPORT_ROOT))
    return str(source_path), rel_dest


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Build summarize_results deck inputs.")
    parser.add_argument("--run-id", required=True, help="Run ID under outputs/.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any required figure is missing.",
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


def main() -> None:
    """CLI entry point.

    Returns
    -------
    None
        The function exits on build failure.
    """
    args = parse_args()
    log_level = resolve_log_level(args.log_level, debug=args.debug, quiet=args.quiet)
    log_file = Path(args.log_file) if args.log_file else None
    logger = setup_logging("build_report", level=log_level, log_file=log_file, run_id=args.run_id)

    run_dir = REPO_ROOT / "outputs" / args.run_id
    report_path = run_dir / "report.json"
    if not report_path.exists():
        logger.error("report.json not found at %s", report_path)
        raise SystemExit(1)

    report = json.loads(report_path.read_text())
    errors, missing_figures = validate_report_payload(
        report, args.run_id, run_dir, REPO_ROOT, strict_figures=args.strict
    )
    if errors:
        for error in errors:
            logger.error(error)
        raise SystemExit(1)
    if missing_figures and not args.strict:
        logger.warning("Missing figures will be replaced with placeholders: %s", missing_figures)

    figures_dir = REPORT_ROOT / "figures"
    data_dir = REPORT_ROOT / "_data"
    build_dir = REPORT_ROOT / "build"
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)

    report_figures = report.get("figures", {})
    updated_figures: Dict[str, Any] = {}
    manifest_lines = [
        f"run_id: {report.get('run_id')}",
        f"stage: {report.get('stage')}",
        f"git_commit: {report.get('git_commit')}",
        f"built_at: {datetime.now().isoformat(timespec='seconds')}",
        f"source_report: {safe_relpath(report_path, REPO_ROOT)}",
        "figures:",
    ]

    for key, value in report_figures.items():
        if key == "failure_modes":
            continue
        if not isinstance(value, str):
            continue
        try:
            source_path, rel_dest = _copy_figure(key, value, run_dir, figures_dir, args.strict)
        except FileNotFoundError as exc:
            logger.error(str(exc))
            raise SystemExit(1)
        updated_figures[key] = rel_dest
        manifest_lines.append(f"  - {key}: {rel_dest} (from {source_path})")

    failure_modes = report_figures.get("failure_modes")
    if isinstance(failure_modes, list):
        updated_figures["failure_modes"] = failure_modes

    comparison_table = None
    comparison_metrics = []
    if metrics:
        comparison_metrics = sorted([key for key in metrics.keys()])
        all_reports = sorted((REPO_ROOT / "outputs").glob("*/report.json"))
        rows, _ = build_comparison_rows(all_reports, metrics_keys=comparison_metrics)
        rows = [row for row in rows if row.get("stage") == report.get("stage")]
        if rows:
            def _ts(row: Dict[str, Any]) -> str:
                return str(row.get("timestamp") or "")

            rows = sorted(rows, key=_ts, reverse=True)[:10]
            comparison_path = data_dir / "run_comparison.csv"
            write_comparison_table(
                rows,
                comparison_metrics,
                comparison_path,
                output_format="csv",
                source_glob="outputs/*/report.json",
            )
            comparison_table = normalize_path(comparison_path.relative_to(REPORT_ROOT))

    next_steps = _extract_next_steps(REPO_ROOT / "todo_list.md")
    notes = report.get("notes") or []
    artifacts = report.get("artifacts") or {}
    metrics = report.get("metrics") or {}
    latest_payload = {
        "run_id": report.get("run_id"),
        "timestamp": report.get("timestamp"),
        "git_commit": report.get("git_commit"),
        "stage": report.get("stage"),
        "status": report.get("status"),
        "progress": report.get("progress"),
        "dataset": report.get("dataset"),
        "dataset_path": report.get("dataset_path"),
        "config": report.get("config"),
        "metrics": metrics,
        "figures": updated_figures,
        "notes": notes,
        "artifacts": artifacts,
        "tracking_sample": report.get("tracking_sample"),
        "comparison_table": comparison_table,
        "comparison_metrics": comparison_metrics,
        "next_steps": next_steps,
        "source_report": safe_relpath(report_path, REPO_ROOT),
    }

    latest_path = data_dir / "latest.json"
    latest_path.write_text(json.dumps(latest_payload, indent=2))
    logger.info("latest.json written to %s", latest_path)

    manifest_path = build_dir / "manifest.txt"
    manifest_path.write_text("\n".join(manifest_lines) + "\n")
    logger.info("Build manifest written to %s", manifest_path)


if __name__ == "__main__":
    main()
