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
        "noise_gaussian_std",
        "noise_a_rotation_deg",
        "noise_b_rotation_deg",
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
    payloads: Sequence[Dict[str, Any]],
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

    payload_list = list(payloads) if payloads else []
    primary_payload = payload_list[0] if payload_list else {}
    inputs = primary_payload.get("inputs", {}) if isinstance(primary_payload.get("inputs"), dict) else {}
    input_artifacts = inputs.get("artifacts", {}) if isinstance(inputs.get("artifacts"), dict) else {}
    metrics_enabled = primary_payload.get("metrics_enabled", [])
    noise_payload = inputs.get("noise", {}) if isinstance(inputs.get("noise"), dict) else {}

    def _format(value: Any) -> str:
        if isinstance(value, (int, float)):
            return f"{value:.6f}"
        return "-"

    a_img = _rel_to_report(input_artifacts.get("a"))
    b_img = _rel_to_report(input_artifacts.get("b"))
    a_noisy_img = _rel_to_report(input_artifacts.get("a_noisy"))
    b_noisy_img = _rel_to_report(input_artifacts.get("b_noisy"))
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
        "<p>"
        f"Noise enabled: <code>{escape(str(noise_payload.get('enabled', False)))}</code> | "
        f"gaussian_std: <code>{escape(_format(noise_payload.get('gaussian_std')))}</code> | "
        f"rotation_deg_max: <code>{escape(_format(noise_payload.get('rotation_deg_max')))}</code> | "
        f"A_rot: <code>{escape(_format(noise_payload.get('a_rotation_deg')))}</code> | "
        f"B_rot: <code>{escape(_format(noise_payload.get('b_rotation_deg')))}</code>"
        "</p>",
        "<div class='row'>",
        "<div class='card'><h3>A</h3>"
        + (f"<img class='img' src='{escape(a_img)}' alt='A'/>" if a_img else "<p>missing</p>")
        + "</div>",
        "<div class='card'><h3>B</h3>"
        + (f"<img class='img' src='{escape(b_img)}' alt='B'/>" if b_img else "<p>missing</p>")
        + "</div>",
        "<div class='card'><h3>A (noisy)</h3>"
        + (f"<img class='img' src='{escape(a_noisy_img)}' alt='A noisy'/>" if a_noisy_img else "<p>missing</p>")
        + "</div>",
        "<div class='card'><h3>B (noisy)</h3>"
        + (f"<img class='img' src='{escape(b_noisy_img)}' alt='B noisy'/>" if b_noisy_img else "<p>missing</p>")
        + "</div>",
        "</div>",
    ]
    for payload in payload_list:
        inputs = payload.get("inputs", {}) if isinstance(payload.get("inputs"), dict) else {}
        input_artifacts = inputs.get("artifacts", {}) if isinstance(inputs.get("artifacts"), dict) else {}
        x_true = inputs.get("x_true")
        x_true_text = f"{float(x_true):.6f}" if isinstance(x_true, (int, float)) else "-"
        c_img = _rel_to_report(input_artifacts.get("c"))
        html_lines.extend(
            [
                "<h2>Results for x_true = "
                + escape(x_true_text)
                + "</h2>",
                "<div class='row'>",
                "<div class='card'><h3>C (synthetic)</h3>"
                + (f"<img class='img' src='{escape(c_img)}' alt='C'/>" if c_img else "<p>missing</p>")
                + "</div>",
                "</div>",
                "<table>",
                "<tr>"
                "<th>Optimize</th><th>x_hat</th><th>|x_hat-x_true|</th><th>objective_score</th>"
                + metric_headers
                + "<th>C_hat</th><th>Links</th></tr>",
            ]
        )
        objective_results = payload.get("objective_results", [])
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

            html_lines.append(
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
        html_lines.extend(["</table>"])

    html_lines.append("</body></html>")
    path.write_text("\n".join(html_lines), encoding="utf-8")


def write_candidate_pool_summary_csv(
    path: Path,
    trial_results: Sequence[Dict[str, Any]],
    metric_names: Sequence[str],
) -> None:
    """Write summary CSV for candidate-pool synthetic trials."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial_id",
        "x_true",
        "x_hat_primary",
        "x_signed_error",
        "x_abs_error",
        "pair_match",
        "true_a_id",
        "true_b_id",
        "pred_a_id",
        "pred_b_id",
        "noise_gaussian_std",
        "noise_rotation_deg_max",
        "noise_true_a_rotation_deg",
        "noise_true_b_rotation_deg",
        "noise_pred_a_rotation_deg",
        "noise_pred_b_rotation_deg",
    ]
    fieldnames += [f"x_hat_{metric}" for metric in metric_names]
    fieldnames += [f"score_{metric}" for metric in metric_names]
    fieldnames += [f"metric_at_primary_{metric}" for metric in metric_names]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in trial_results:
            true_pair = record.get("true_pair", {}) if isinstance(record.get("true_pair"), dict) else {}
            pred_pair = record.get("predicted_pair", {}) if isinstance(record.get("predicted_pair"), dict) else {}
            noise_payload = record.get("noise", {}) if isinstance(record.get("noise"), dict) else {}
            best_by_metric = record.get("best_by_metric", {}) if isinstance(record.get("best_by_metric"), dict) else {}
            metrics_at_primary = record.get("metrics_at_primary", {}) if isinstance(record.get("metrics_at_primary"), dict) else {}

            row: Dict[str, Any] = {
                "trial_id": record.get("trial_id"),
                "x_true": record.get("x_true"),
                "x_hat_primary": record.get("x_hat_primary"),
                "x_signed_error": record.get("x_signed_error"),
                "x_abs_error": record.get("x_abs_error"),
                "pair_match": record.get("pair_match"),
                "true_a_id": true_pair.get("a_id"),
                "true_b_id": true_pair.get("b_id"),
                "pred_a_id": pred_pair.get("a_id"),
                "pred_b_id": pred_pair.get("b_id"),
                "noise_gaussian_std": noise_payload.get("gaussian_std"),
                "noise_rotation_deg_max": noise_payload.get("rotation_deg_max"),
                "noise_true_a_rotation_deg": noise_payload.get("true_a_rotation_deg"),
                "noise_true_b_rotation_deg": noise_payload.get("true_b_rotation_deg"),
                "noise_pred_a_rotation_deg": noise_payload.get("pred_a_rotation_deg"),
                "noise_pred_b_rotation_deg": noise_payload.get("pred_b_rotation_deg"),
            }

            for metric_name in metric_names:
                metric_payload = best_by_metric.get(metric_name, {})
                row[f"x_hat_{metric_name}"] = metric_payload.get("x_hat")
                row[f"score_{metric_name}"] = metric_payload.get("score")
                row[f"metric_at_primary_{metric_name}"] = metrics_at_primary.get(metric_name)

            writer.writerow(row)


def write_candidate_pool_html_report(
    path: Path,
    output_dir: Path,
    trial_results: Sequence[Dict[str, Any]],
    metric_names: Sequence[str],
) -> None:
    """Write HTML report for candidate-pool synthetic trials."""
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

    def _fmt(value: Any) -> str:
        if isinstance(value, (int, float)):
            return f"{value:.6f}"
        return "-"

    metric_headers = "".join(f"<th>{escape(str(name))}</th>" for name in metric_names)

    html_lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>Candidate Pool Synthetic Search</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:20px;}",
        ".row{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start;}",
        ".card{border:1px solid #ddd;border-radius:8px;padding:10px;}",
        ".card h3{margin:0 0 8px 0;font-size:14px;}",
        "img{image-rendering:auto;}",
        ".img{width:220px;max-width:90vw;height:auto;border:1px solid #eee;}",
        ".thumb{width:120px;height:auto;border:1px solid #eee;}",
        "table{border-collapse:collapse;width:100%;margin-top:16px;}",
        "th,td{border:1px solid #ddd;padding:6px;text-align:left;vertical-align:top;}",
        "th{background:#f5f5f5;}",
        "code{background:#f6f8fa;padding:1px 4px;border-radius:4px;}",
        "</style></head><body>",
        "<h1>Candidate Pool Synthetic Search</h1>",
        f"<p>Trials: <code>{len(trial_results)}</code></p>",
    ]

    for record in trial_results:
        trial_id = escape(str(record.get("trial_id", "trial")))
        true_pair = record.get("true_pair", {}) if isinstance(record.get("true_pair"), dict) else {}
        pred_pair = record.get("predicted_pair", {}) if isinstance(record.get("predicted_pair"), dict) else {}
        noise_payload = record.get("noise", {}) if isinstance(record.get("noise"), dict) else {}
        artifacts = record.get("artifacts", {}) if isinstance(record.get("artifacts"), dict) else {}

        a_true = _rel_to_report(artifacts.get("a_true"))
        b_true = _rel_to_report(artifacts.get("b_true"))
        a_true_noisy = _rel_to_report(artifacts.get("a_true_noisy"))
        b_true_noisy = _rel_to_report(artifacts.get("b_true_noisy"))
        a_pred = _rel_to_report(artifacts.get("a_pred"))
        b_pred = _rel_to_report(artifacts.get("b_pred"))
        a_pred_noisy = _rel_to_report(artifacts.get("a_pred_noisy"))
        b_pred_noisy = _rel_to_report(artifacts.get("b_pred_noisy"))
        c_true = _rel_to_report(artifacts.get("c_true"))
        c_hat = _rel_to_report(artifacts.get("c_hat"))

        html_lines.extend(
            [
                f"<h2>{trial_id}</h2>",
                "<p>"
                f"x_true: <code>{_fmt(record.get('x_true'))}</code> | "
                f"x_hat(primary): <code>{_fmt(record.get('x_hat_primary'))}</code> | "
                f"pair_match: <code>{escape(str(record.get('pair_match')))}</code>"
                "</p>",
                "<p>"
                f"True pair: <code>{escape(str(true_pair.get('a_id', '-')))}</code> + "
                f"<code>{escape(str(true_pair.get('b_id', '-')))}</code> | "
                f"Predicted pair: <code>{escape(str(pred_pair.get('a_id', '-')))}</code> + "
                f"<code>{escape(str(pred_pair.get('b_id', '-')))}</code>"
                "</p>",
                "<p>"
                f"Noise enabled: <code>{escape(str(noise_payload.get('enabled', False)))}</code> | "
                f"gaussian_std: <code>{escape(_fmt(noise_payload.get('gaussian_std')))}</code> | "
                f"rotation_deg_max: <code>{escape(_fmt(noise_payload.get('rotation_deg_max')))}</code>"
                "</p>",
                "<div class='row'>",
                "<div class='card'><h3>A true</h3>"
                + (f"<img class='img' src='{escape(a_true)}' alt='A true'/>" if a_true else "<p>missing</p>")
                + "</div>",
                "<div class='card'><h3>B true</h3>"
                + (f"<img class='img' src='{escape(b_true)}' alt='B true'/>" if b_true else "<p>missing</p>")
                + "</div>",
                "<div class='card'><h3>A true (noisy)</h3>"
                + (f"<img class='img' src='{escape(a_true_noisy)}' alt='A true noisy'/>" if a_true_noisy else "<p>missing</p>")
                + "</div>",
                "<div class='card'><h3>B true (noisy)</h3>"
                + (f"<img class='img' src='{escape(b_true_noisy)}' alt='B true noisy'/>" if b_true_noisy else "<p>missing</p>")
                + "</div>",
                "</div>",
                "<div class='row'>",
                "<div class='card'><h3>A pred</h3>"
                + (f"<img class='img' src='{escape(a_pred)}' alt='A pred'/>" if a_pred else "<p>missing</p>")
                + "</div>",
                "<div class='card'><h3>B pred</h3>"
                + (f"<img class='img' src='{escape(b_pred)}' alt='B pred'/>" if b_pred else "<p>missing</p>")
                + "</div>",
                "<div class='card'><h3>A pred (noisy)</h3>"
                + (f"<img class='img' src='{escape(a_pred_noisy)}' alt='A pred noisy'/>" if a_pred_noisy else "<p>missing</p>")
                + "</div>",
                "<div class='card'><h3>B pred (noisy)</h3>"
                + (f"<img class='img' src='{escape(b_pred_noisy)}' alt='B pred noisy'/>" if b_pred_noisy else "<p>missing</p>")
                + "</div>",
                "</div>",
                "<div class='row'>",
                "<div class='card'><h3>C true</h3>"
                + (f"<img class='img' src='{escape(c_true)}' alt='C true'/>" if c_true else "<p>missing</p>")
                + "</div>",
                "<div class='card'><h3>C hat</h3>"
                + (f"<img class='img' src='{escape(c_hat)}' alt='C hat'/>" if c_hat else "<p>missing</p>")
                + "</div>",
                "</div>",
                "<table>",
                "<tr><th>Optimize</th><th>x_hat</th><th>objective_score</th>"
                + metric_headers
                + "<th>Links</th></tr>",
            ]
        )

        objective_results = record.get("objective_results", [])
        for objective in objective_results:
            objective_metric = escape(str(objective.get("objective_metric", "-")))
            x_hat = objective.get("x_hat")
            objective_score = objective.get("objective_score")
            metrics = objective.get("metrics", {}) if isinstance(objective.get("metrics"), dict) else {}
            metric_cells = []
            for metric_name in metric_names:
                value = metrics.get(metric_name)
                metric_cells.append(f"<td>{_fmt(value)}</td>" if isinstance(value, (int, float)) else "<td>-</td>")

            links: List[str] = []
            qual_rel = _rel_to_report(artifacts.get("qual_panel"))
            curve_rel = _rel_to_report(artifacts.get("score_curve_csv"))
            if qual_rel:
                links.append(f"<a href=\"{escape(qual_rel)}\">panel</a>")
            if curve_rel:
                links.append(f"<a href=\"{escape(curve_rel)}\">curve.csv</a>")
            links_cell = "<td>" + (" | ".join(links) if links else "-") + "</td>"

            html_lines.append(
                "<tr>"
                f"<td>{objective_metric}</td>"
                f"<td>{_fmt(x_hat)}</td>"
                f"<td>{_fmt(objective_score)}</td>"
                + "".join(metric_cells)
                + links_cell
                + "</tr>"
            )
        html_lines.append("</table>")

    html_lines.append("</body></html>")
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
