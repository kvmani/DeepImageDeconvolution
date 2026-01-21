"""Aggregate report.json metrics across multiple runs for quick comparison."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.logging import resolve_log_level, setup_logging


def _parse_list(values: list[str]) -> list[str]:
    items: list[str] = []
    for raw in values:
        for part in raw.split(","):
            cleaned = part.strip()
            if cleaned:
                items.append(cleaned)
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare metrics across training/inference runs.")
    parser.add_argument(
        "--run-id",
        action="append",
        default=[],
        help="Run ID under outputs/ (repeatable).",
    )
    parser.add_argument(
        "--report",
        action="append",
        default=[],
        help="Explicit report.json path (repeatable).",
    )
    parser.add_argument(
        "--glob",
        default="outputs/*/report.json",
        help="Glob for report.json files when no --run-id/--report is provided.",
    )
    parser.add_argument(
        "--metrics",
        action="append",
        default=[],
        help="Metric keys to include (repeatable or comma-separated). Defaults to union of all metrics.",
    )
    parser.add_argument(
        "--format",
        choices=("csv", "json"),
        default="csv",
        help="Output format.",
    )
    parser.add_argument("--out", default=None, help="Output path for the comparison table.")
    parser.add_argument("--sort-by", default=None, help="Sort rows by a metric key.")
    parser.add_argument("--descending", action="store_true", help="Sort descending.")
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


def _resolve_report_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    run_ids = _parse_list(args.run_id)
    report_args = _parse_list(args.report)

    for run_id in run_ids:
        paths.append(REPO_ROOT / "outputs" / run_id / "report.json")
    for report in report_args:
        path = Path(report)
        if not path.is_absolute():
            path = REPO_ROOT / path
        paths.append(path)

    if not run_ids and not report_args:
        paths = sorted(REPO_ROOT.glob(args.glob))

    return paths


def _load_report(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _sort_key(value: Any) -> tuple[bool, Any]:
    if value is None:
        return (True, None)
    return (False, value)


def _coerce_cell(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return value


def build_comparison_rows(
    report_paths: list[Path],
    metrics_keys: list[str] | None = None,
    sort_by: str | None = None,
    descending: bool = False,
) -> tuple[list[Dict[str, Any]], list[str]]:
    rows: list[Dict[str, Any]] = []
    metrics_union: list[str] = []
    metrics_set = set()

    for path in report_paths:
        if not path.exists():
            continue
        report = _load_report(path)
        metrics = report.get("metrics") or {}
        if isinstance(metrics, dict):
            for key in metrics:
                if key not in metrics_set:
                    metrics_set.add(key)
                    metrics_union.append(key)

        progress = report.get("progress") or {}
        row: Dict[str, Any] = {
            "run_id": report.get("run_id") or path.parent.name,
            "stage": report.get("stage"),
            "status": report.get("status"),
            "timestamp": report.get("timestamp"),
            "dataset": report.get("dataset"),
            "epoch": progress.get("epoch"),
            "epochs_total": progress.get("epochs_total"),
        }
        if isinstance(metrics, dict):
            row.update(metrics)
        rows.append(row)

    if metrics_keys is None:
        metrics_keys = metrics_union

    if sort_by:
        rows.sort(
            key=lambda row: _sort_key(row.get(sort_by)),
            reverse=descending,
        )

    return rows, metrics_keys


def write_comparison_table(
    rows: list[Dict[str, Any]],
    metrics_keys: list[str],
    out_path: Path,
    output_format: str = "csv",
    source_glob: str | None = None,
) -> Path:
    base_fields = ["run_id", "stage", "status", "timestamp", "dataset", "epoch", "epochs_total"]
    fieldnames = base_fields + metrics_keys
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "json":
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source_glob": source_glob,
            "runs": rows,
            "metrics": metrics_keys,
        }
        out_path.write_text(json.dumps(payload, indent=2))
    else:
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _coerce_cell(row.get(key)) for key in fieldnames})

    return out_path


def main() -> None:
    args = parse_args()
    log_level = resolve_log_level(args.log_level, debug=args.debug, quiet=args.quiet)
    log_file = Path(args.log_file) if args.log_file else None
    logger = setup_logging("compare_runs", level=log_level, log_file=log_file)

    report_paths = _resolve_report_paths(args)
    if not report_paths:
        logger.error("No report.json files found.")
        raise SystemExit(1)

    rows, metrics_union = build_comparison_rows(
        report_paths,
        metrics_keys=None,
        sort_by=args.sort_by,
        descending=args.descending,
    )
    if not rows:
        logger.error("No readable report.json files found.")
        raise SystemExit(1)

    metric_keys = _parse_list(args.metrics)
    if not metric_keys:
        metric_keys = metrics_union

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = REPO_ROOT / out_path
    else:
        suffix = "json" if args.format == "json" else "csv"
        out_path = REPO_ROOT / "reports" / "summarize_results" / f"run_comparison.{suffix}"

    source_glob = args.glob if not args.run_id and not args.report else None
    write_comparison_table(
        rows,
        metric_keys,
        out_path,
        output_format=args.format,
        source_glob=source_glob,
    )

    logger.info("Wrote comparison table to %s (%d runs).", out_path, len(rows))


if __name__ == "__main__":
    main()
