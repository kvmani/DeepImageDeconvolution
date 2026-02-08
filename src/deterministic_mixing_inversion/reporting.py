"""Reporting helpers for deterministic inversion runs."""
from __future__ import annotations

import csv
from html import escape
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

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


def write_synthetic_pair_summary_csv(
    path: Path,
    objective_results: Sequence[Dict[str, Any]],
    metric_names: Sequence[str],
) -> None:
    """Write per-objective-metric summary for a synthetic single-pair run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "objective_metric",
        "x_true",
        "x_hat",
        "x_signed_error",
        "x_abs_error",
        "objective_score",
        "top_margin",
    ] + [f"score_{metric}" for metric in metric_names]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for record in objective_results:
            metrics = record.get("metrics", {})
            row: Dict[str, Any] = {key: record.get(key) for key in fieldnames}
            for metric_name in metric_names:
                row[f"score_{metric_name}"] = metrics.get(metric_name)
            writer.writerow(row)


def write_synthetic_pair_html_report(
    path: Path,
    output_dir: Path,
    payload: Dict[str, Any],
) -> None:
    """Write an HTML report for synthetic single-pair x estimation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    report_dir = path.parent

    def _rel_to_report(artifact_path: str | None) -> str:
        if not artifact_path:
            return ""
        artifact_obj = Path(str(artifact_path))
        if artifact_obj.is_absolute():
            full = artifact_obj
        else:
            candidate = output_dir / artifact_obj
            full = candidate if candidate.exists() else artifact_obj
        rel = os.path.relpath(full.resolve(), report_dir.resolve())
        return Path(rel).as_posix()

    inputs = payload.get("inputs", {}) if isinstance(payload.get("inputs"), dict) else {}
    input_artifacts = inputs.get("artifacts", {}) if isinstance(inputs.get("artifacts"), dict) else {}
    objective_results = payload.get("objective_results", [])
    metrics_enabled = payload.get("metrics_enabled", [])

    rows: List[str] = []
    for record in objective_results:
        objective_metric = escape(str(record.get("objective_metric", "-")))
        x_hat = record.get("x_hat")
        x_abs_error = record.get("x_abs_error")
        objective_score = record.get("objective_score")
        metrics = record.get("metrics", {}) if isinstance(record.get("metrics"), dict) else {}
        artifacts = record.get("artifacts", {}) if isinstance(record.get("artifacts"), dict) else {}

        metric_cells = []
        for metric_name in metrics_enabled:
            value = metrics.get(metric_name)
            if isinstance(value, (int, float)):
                metric_cells.append(f"<td>{value:.6f}</td>")
            else:
                metric_cells.append("<td>-</td>")

        c_hat_rel = _rel_to_report(artifacts.get("c_hat"))
        qual_rel = _rel_to_report(artifacts.get("qual_panel"))
        curve_rel = _rel_to_report(artifacts.get("score_curve_csv"))

        image_cell = "<td>-</td>"
        if c_hat_rel:
            image_cell = f"<td><img src=\"{escape(c_hat_rel)}\" alt=\"C_hat\" class=\"thumb\" /></td>"

        links: List[str] = []
        if qual_rel:
            links.append(f"<a href=\"{escape(qual_rel)}\">panel</a>")
        if curve_rel:
            links.append(f"<a href=\"{escape(curve_rel)}\">curve.csv</a>")
        links_cell = "<td>" + (" | ".join(links) if links else "-") + "</td>"

        def _format(value: Any) -> str:
            if isinstance(value, (int, float)):
                return f"{value:.6f}"
            return "-"

        rows.append(
            "<tr>"
            f"<td>{objective_metric}</td>"
            f"<td>{_format(x_hat)}</td>"
            f"<td>{_format(x_abs_error)}</td>"
            f"<td>{_format(objective_score)}</td>"
            + "".join(metric_cells)
            + image_cell
            + links_cell
            + "</tr>"
        )

    a_img = _rel_to_report(input_artifacts.get("a"))
    b_img = _rel_to_report(input_artifacts.get("b"))
    c_img = _rel_to_report(input_artifacts.get("c"))
    x_true = inputs.get("x_true")
    x_true_text = f"{float(x_true):.6f}" if isinstance(x_true, (int, float)) else "-"

    metric_headers = "".join(f"<th>{escape(str(name))}</th>" for name in metrics_enabled)

    html_lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>Deterministic Synthetic Pair Inversion</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:20px;}",
        ".row{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start;}",
        ".card{border:1px solid #ddd;border-radius:8px;padding:10px;}",
        ".card h3{margin:0 0 8px 0;font-size:14px;}",
        "img{image-rendering:auto;}",
        ".img{width:260px;max-width:90vw;height:auto;border:1px solid #eee;}",
        ".thumb{width:140px;height:auto;border:1px solid #eee;}",
        "table{border-collapse:collapse;width:100%;margin-top:16px;}",
        "th,td{border:1px solid #ddd;padding:6px;text-align:left;vertical-align:top;}",
        "th{background:#f5f5f5;}",
        "code{background:#f6f8fa;padding:1px 4px;border-radius:4px;}",
        "</style></head><body>",
        "<h1>Deterministic Synthetic Pair Inversion</h1>",
        f"<p>x_true: <code>{escape(x_true_text)}</code></p>",
        "<div class='row'>",
        "<div class='card'><h3>A</h3>"
        + (f"<img class='img' src='{escape(a_img)}' alt='A'/>" if a_img else "<p>missing</p>")
        + "</div>",
        "<div class='card'><h3>B</h3>"
        + (f"<img class='img' src='{escape(b_img)}' alt='B'/>" if b_img else "<p>missing</p>")
        + "</div>",
        "<div class='card'><h3>C (synthetic)</h3>"
        + (f"<img class='img' src='{escape(c_img)}' alt='C'/>" if c_img else "<p>missing</p>")
        + "</div>",
        "</div>",
        "<h2>Per-metric optimization</h2>",
        "<table>",
        "<tr>"
        "<th>Optimize</th><th>x_hat</th><th>|x_hat-x_true|</th><th>objective_score</th>"
        + metric_headers
        + "<th>C_hat</th><th>Links</th></tr>",
        *rows,
        "</table>",
        "</body></html>",
    ]
    path.write_text("\n".join(html_lines), encoding="utf-8")


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
