"""Reporting helpers for deterministic inversion runs."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib
import numpy as np

from src.utils.reporting import write_report_json

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """Append one JSON record to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str))
        handle.write("\n")


def write_score_curve_csv(path: Path, score_curves: Dict[str, List[tuple[float, float]]]) -> None:
    """Write score-vs-fraction curves to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_names = sorted(score_curves.keys())
    fractions = sorted({fraction for curve in score_curves.values() for fraction, _ in curve})
    curve_maps: Dict[str, Dict[float, float]] = {
        metric_name: {fraction: score for fraction, score in curve}
        for metric_name, curve in score_curves.items()
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["fraction"] + metric_names
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for fraction in fractions:
            row: Dict[str, Any] = {"fraction": fraction}
            for metric_name in metric_names:
                row[metric_name] = curve_maps[metric_name].get(fraction)
            writer.writerow(row)


def write_metric_summary_csv(
    path: Path,
    sample_results: Iterable[Dict[str, Any]],
    metric_names: List[str],
) -> None:
    """Write aggregate metric summary across samples."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for metric_name in metric_names:
        fractions: List[float] = []
        scores: List[float] = []
        margins: List[float] = []
        for result in sample_results:
            metric_payload = result.get("best_by_metric", {}).get(metric_name)
            if not metric_payload:
                continue
            fractions.append(float(metric_payload["x_hat"]))
            scores.append(float(metric_payload["score"]))
            top_margin = metric_payload.get("top_margin")
            if top_margin is not None:
                margins.append(float(top_margin))
        if not fractions:
            continue
        rows.append(
            {
                "metric": metric_name,
                "samples": len(fractions),
                "x_hat_mean": float(np.mean(fractions)),
                "x_hat_std": float(np.std(fractions)),
                "score_mean": float(np.mean(scores)),
                "score_std": float(np.std(scores)),
                "top_margin_mean": float(np.mean(margins)) if margins else None,
            }
        )

    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "metric",
            "samples",
            "x_hat_mean",
            "x_hat_std",
            "score_mean",
            "score_std",
            "top_margin_mean",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_primary_fraction_histogram(
    path: Path,
    sample_results: Iterable[Dict[str, Any]],
    primary_metric: str,
) -> bool:
    """Plot histogram of primary-metric mixing fractions."""
    fractions: List[float] = []
    for result in sample_results:
        metric_payload = result.get("best_by_metric", {}).get(primary_metric)
        if metric_payload:
            fractions.append(float(metric_payload["x_hat"]))
    if not fractions:
        return False
    figure, axis = plt.subplots(figsize=(5.0, 3.5))
    axis.hist(fractions, bins=20, range=(0.0, 1.0), color="#1f77b4", alpha=0.85)
    axis.set_xlabel("x_hat")
    axis.set_ylabel("count")
    axis.set_title(f"Primary metric ({primary_metric})")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return True


def write_html_report(
    path: Path,
    sample_results: Iterable[Dict[str, Any]],
    primary_metric: str,
) -> None:
    """Write a compact HTML summary report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[str] = []
    for result in sample_results:
        sample_id = str(result.get("sample_id", "unknown"))
        pair_payload = result.get("best_pair", {})
        metric_payload = result.get("best_by_metric", {}).get(primary_metric, {})
        rows.append(
            "<tr>"
            f"<td>{sample_id}</td>"
            f"<td>{pair_payload.get('a_id', '-')}</td>"
            f"<td>{pair_payload.get('b_id', '-')}</td>"
            f"<td>{metric_payload.get('x_hat', '-')}</td>"
            f"<td>{metric_payload.get('score', '-')}</td>"
            "</tr>"
        )

    html_lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>Deterministic Inversion Report</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:20px;}",
        "table{border-collapse:collapse;width:100%;}",
        "th,td{border:1px solid #ddd;padding:6px;text-align:left;}",
        "th{background:#f5f5f5;}",
        "</style></head><body>",
        "<h1>Deterministic Mixing-Fraction Inversion</h1>",
        f"<p>Primary metric: <strong>{primary_metric}</strong></p>",
        "<table><tr><th>Sample</th><th>A</th><th>B</th><th>x_hat</th><th>score</th></tr>",
        *rows,
        "</table>",
        "</body></html>",
    ]
    path.write_text("\n".join(html_lines), encoding="utf-8")


def update_progress_report(
    run_dir: Path,
    report_payload: Dict[str, Any],
) -> Path:
    """Write/overwrite report.json for progress tracking."""
    return write_report_json(run_dir, report_payload)

