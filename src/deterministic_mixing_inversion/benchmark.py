"""Synthetic robustness benchmark for deterministic inversion."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Sequence

import matplotlib
import numpy as np
from scipy.ndimage import gaussian_filter

from src.deterministic_mixing_inversion.alignment import (
    AlignmentSettings,
    RigidAlignment,
    apply_rigid_alignment,
    estimate_best_alignment,
    parse_alignment_settings,
)
from src.deterministic_mixing_inversion.io import (
    PatternRecord,
    build_pair_indices,
    candidate_paths,
    load_candidate_pools,
)
from src.deterministic_mixing_inversion.metrics import (
    MetricSpec,
    compute_metric_values,
    is_better,
    parse_metric_config,
)
from src.deterministic_mixing_inversion.preprocess import (
    PreprocessSettings,
    build_centered_mask,
    match_shape_to_target,
    parse_preprocess_settings,
    preprocess_pattern,
)
from src.deterministic_mixing_inversion.search import (
    MetricBest,
    SearchSettings,
    parse_search_settings,
    search_mixing_fraction,
)
from src.utils.io import write_image_16bit
from src.utils.logging import ProgressLogger, summarize_images
from src.utils.reporting import safe_relpath, write_report_json

matplotlib.use("Agg")
from matplotlib import pyplot as plt


@dataclass(frozen=True)
class PreparedPattern:
    """Prepared pattern cache entry."""

    record: PatternRecord
    raw_matched: np.ndarray
    processed: np.ndarray
    preprocess_meta: Dict[str, object]


@dataclass(frozen=True)
class PairEvaluation:
    """Candidate pair evaluation output."""

    best_by_metric: Dict[str, MetricBest]
    evaluated_points: int
    alignment: RigidAlignment


@dataclass(frozen=True)
class NuisanceCase:
    """One synthetic nuisance configuration."""

    gain: float
    offset: float
    noise_std: float
    blur_sigma: float
    shift_y: float
    shift_x: float
    rotation_deg: float


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _mix_patterns(image_a: np.ndarray, image_b: np.ndarray, mix_fraction: float) -> np.ndarray:
    return (mix_fraction * image_a + (1.0 - mix_fraction) * image_b).astype(np.float32)


def _sanitize_name(text: str) -> str:
    return text.replace("/", "_").replace("\\", "_").replace(" ", "_")


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str))
        handle.write("\n")


def _prepare_pattern_for_shape(
    record: PatternRecord,
    target_shape: tuple[int, int],
    preprocess_settings: PreprocessSettings,
    mask: np.ndarray | None,
) -> PreparedPattern:
    matched = match_shape_to_target(
        record.image,
        target_shape=target_shape,
        auto_crop_to_target=preprocess_settings.auto_crop_to_target,
    )
    processed, preprocess_meta = preprocess_pattern(matched, preprocess_settings, mask)
    return PreparedPattern(
        record=record,
        raw_matched=matched,
        processed=processed,
        preprocess_meta=preprocess_meta,
    )


def _evaluate_pair(
    pattern_a: PreparedPattern,
    pattern_b: PreparedPattern,
    mixed_processed: np.ndarray,
    mask: np.ndarray | None,
    metric_names: Sequence[str],
    metric_specs: Dict[str, MetricSpec],
    primary_metric: str,
    search_settings: SearchSettings,
    alignment_settings: AlignmentSettings,
) -> PairEvaluation:
    def evaluate_unaligned(mix_fraction: float) -> Dict[str, float]:
        reconstructed = _mix_patterns(pattern_a.processed, pattern_b.processed, mix_fraction)
        return compute_metric_values(reconstructed, mixed_processed, mask, metric_names)

    initial_search = search_mixing_fraction(
        evaluate_fraction=evaluate_unaligned,
        metric_specs=metric_specs,
        primary_metric=primary_metric,
        settings=search_settings,
    )
    initial_primary_fraction = initial_search.metric_best[primary_metric].fraction
    initial_reconstruction = _mix_patterns(
        pattern_a.processed,
        pattern_b.processed,
        initial_primary_fraction,
    )

    if alignment_settings.enabled:
        best_alignment = estimate_best_alignment(
            moving=initial_reconstruction,
            target=mixed_processed,
            mask=mask,
            settings=alignment_settings,
            score_function=lambda pred, target, mask: compute_metric_values(
                pred, target, mask, ["ncc"]
            )["ncc"],
        )
    else:
        best_alignment = RigidAlignment(angle_deg=0.0, shift_y=0.0, shift_x=0.0, score=0.0)

    def evaluate_aligned(mix_fraction: float) -> Dict[str, float]:
        reconstructed = _mix_patterns(pattern_a.processed, pattern_b.processed, mix_fraction)
        aligned = apply_rigid_alignment(
            reconstructed,
            best_alignment,
            interpolation_order=alignment_settings.interpolation_order,
            mask=mask,
        )
        return compute_metric_values(aligned, mixed_processed, mask, metric_names)

    if alignment_settings.enabled:
        final_search = search_mixing_fraction(
            evaluate_fraction=evaluate_aligned,
            metric_specs=metric_specs,
            primary_metric=primary_metric,
            settings=search_settings,
        )
    else:
        final_search = initial_search

    return PairEvaluation(
        best_by_metric=final_search.metric_best,
        evaluated_points=final_search.evaluated_points,
        alignment=best_alignment,
    )


def _nuisance_levels(config: Dict[str, object], key: str, default: list[float]) -> list[float]:
    values = config.get(key, default)
    if not isinstance(values, list) or not values:
        raise ValueError(f"synthetic_benchmark.nuisance.{key} must be a non-empty list.")
    return [float(item) for item in values]


def _build_nuisance_cases(nuisance_cfg: Dict[str, object]) -> List[NuisanceCase]:
    gains = _nuisance_levels(nuisance_cfg, "gain_levels", [1.0])
    offsets = _nuisance_levels(nuisance_cfg, "offset_levels", [0.0])
    noise_stds = _nuisance_levels(nuisance_cfg, "noise_std_levels", [0.0])
    blur_sigmas = _nuisance_levels(nuisance_cfg, "blur_sigma_levels", [0.0])
    shift_x_values = _nuisance_levels(nuisance_cfg, "translation_x_levels", [0.0])
    shift_y_values = _nuisance_levels(nuisance_cfg, "translation_y_levels", [0.0])
    rotation_values = _nuisance_levels(nuisance_cfg, "rotation_deg_levels", [0.0])

    cases: List[NuisanceCase] = []
    for gain in gains:
        for offset in offsets:
            for noise_std in noise_stds:
                for blur_sigma in blur_sigmas:
                    for shift_x in shift_x_values:
                        for shift_y in shift_y_values:
                            for rotation_deg in rotation_values:
                                cases.append(
                                    NuisanceCase(
                                        gain=gain,
                                        offset=offset,
                                        noise_std=noise_std,
                                        blur_sigma=blur_sigma,
                                        shift_y=shift_y,
                                        shift_x=shift_x,
                                        rotation_deg=rotation_deg,
                                    )
                                )
    return cases


def _apply_nuisance(
    clean_mix: np.ndarray,
    nuisance_case: NuisanceCase,
    rng: np.random.Generator,
    mask: np.ndarray | None,
) -> np.ndarray:
    mixed = clean_mix.astype(np.float32, copy=True)
    if abs(nuisance_case.rotation_deg) > 1e-12 or abs(nuisance_case.shift_x) > 1e-12 or abs(nuisance_case.shift_y) > 1e-12:
        mixed = apply_rigid_alignment(
            mixed,
            RigidAlignment(
                angle_deg=nuisance_case.rotation_deg,
                shift_y=nuisance_case.shift_y,
                shift_x=nuisance_case.shift_x,
                score=0.0,
            ),
            interpolation_order=3,
            mask=mask,
        )
    if nuisance_case.blur_sigma > 0.0:
        mixed = gaussian_filter(mixed, sigma=nuisance_case.blur_sigma).astype(np.float32)
    mixed = (mixed * nuisance_case.gain) + nuisance_case.offset
    if nuisance_case.noise_std > 0.0:
        noise = rng.normal(loc=0.0, scale=nuisance_case.noise_std, size=mixed.shape).astype(np.float32)
        mixed = mixed + noise
    if mask is not None:
        mixed = mixed.copy()
        mixed[~mask] = 0.0
    mixed = np.clip(mixed, 0.0, 1.0)
    return mixed.astype(np.float32)


def _metric_error_summary(records: List[Dict[str, Any]], metric_names: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for metric_name in metric_names:
        errors = np.asarray([record[f"{metric_name}_error"] for record in records], dtype=np.float32)
        signed = np.asarray([record[f"{metric_name}_signed_error"] for record in records], dtype=np.float32)
        rows.append(
            {
                "metric": metric_name,
                "samples": int(errors.size),
                "mae": float(np.mean(errors)) if errors.size else None,
                "rmse": float(np.sqrt(np.mean(signed * signed))) if signed.size else None,
                "bias": float(np.mean(signed)) if signed.size else None,
                "std": float(np.std(signed)) if signed.size else None,
            }
        )
    rows.sort(key=lambda row: (row["mae"] is None, row["mae"]))
    return rows


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_metric_mae(path: Path, summary_rows: List[Dict[str, Any]]) -> bool:
    if not summary_rows:
        return False
    metrics = [row["metric"] for row in summary_rows]
    maes = [row["mae"] for row in summary_rows]
    figure, axis = plt.subplots(figsize=(6.0, 3.8))
    axis.bar(metrics, maes, color="#1f77b4", alpha=0.85)
    axis.set_ylabel("MAE(|x_hat - x_true|)")
    axis.set_title("Metric Ranking by MAE")
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return True


def _write_html_summary(
    path: Path,
    summary_rows: List[Dict[str, Any]],
    record_count: int,
    primary_metric: str,
) -> None:
    rows: List[str] = []
    for row in summary_rows:
        rows.append(
            "<tr>"
            f"<td>{row['metric']}</td>"
            f"<td>{row['samples']}</td>"
            f"<td>{row['mae']:.6f}</td>"
            f"<td>{row['rmse']:.6f}</td>"
            f"<td>{row['bias']:.6f}</td>"
            f"<td>{row['std']:.6f}</td>"
            "</tr>"
        )
    html = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>Deterministic Robustness Benchmark</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:20px;}",
        "table{border-collapse:collapse;width:100%;}",
        "th,td{border:1px solid #ddd;padding:6px;text-align:left;}",
        "th{background:#f5f5f5;}",
        "</style></head><body>",
        "<h1>Deterministic Robustness Benchmark</h1>",
        f"<p>Samples evaluated: {record_count}</p>",
        f"<p>Primary metric: <strong>{primary_metric}</strong></p>",
        "<table><tr><th>Metric</th><th>Samples</th><th>MAE</th><th>RMSE</th><th>Bias</th><th>Std</th></tr>",
        *rows,
        "</table>",
        "</body></html>",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(html), encoding="utf-8")


def run_synthetic_robustness_benchmark(
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Run synthetic robustness benchmark for metric ranking."""
    if logger is None:
        logger = logging.getLogger(__name__)

    output_cfg = config.get("output", {})
    benchmark_cfg = config.get("synthetic_benchmark", {})
    data_cfg = config.get("data", {})
    debug_cfg = config.get("debug", {})
    inversion_cfg = config.get("deterministic_inversion", {})

    output_dir = Path(output_cfg.get("out_dir", "outputs/deterministic_robustness"))
    output_dir.mkdir(parents=True, exist_ok=True)

    debug_enabled = bool(debug_cfg.get("enabled", False))
    seed = int(debug_cfg.get("seed", 42))
    _set_seed(seed)
    rng = np.random.default_rng(seed)

    preprocess_settings = parse_preprocess_settings(inversion_cfg.get("preprocess", {}))
    metric_names, primary_metric, metric_specs = parse_metric_config(inversion_cfg.get("metrics", {}))
    search_settings = parse_search_settings(inversion_cfg.get("search", {}))
    alignment_settings = parse_alignment_settings(inversion_cfg.get("alignment", {}))

    candidate_cfg = data_cfg.get("candidate_pool", {})
    candidates_a, candidates_b = load_candidate_pools(candidate_cfg, logger)
    pair_indices = build_pair_indices(
        candidates_a,
        candidates_b,
        int(benchmark_cfg.get("max_candidate_pairs")) if benchmark_cfg.get("max_candidate_pairs") else None,
    )
    if not pair_indices:
        raise ValueError("No candidate pairs available for synthetic benchmark.")

    x_values_raw = benchmark_cfg.get("x_values", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    if not isinstance(x_values_raw, list) or not x_values_raw:
        raise ValueError("synthetic_benchmark.x_values must be a non-empty list.")
    x_values = [float(np.clip(value, 0.0, 1.0)) for value in x_values_raw]

    nuisance_cfg = benchmark_cfg.get("nuisance", {})
    nuisance_cases = _build_nuisance_cases(nuisance_cfg if isinstance(nuisance_cfg, dict) else {})
    if debug_enabled:
        max_nuisance_cases = int(benchmark_cfg.get("debug_max_nuisance_cases", min(3, len(nuisance_cases))))
        nuisance_cases = nuisance_cases[:max_nuisance_cases]

    max_samples = benchmark_cfg.get("max_samples")
    max_samples_int = int(max_samples) if max_samples is not None else None
    use_true_pair_only = bool(benchmark_cfg.get("use_true_pair_only", True))
    save_synthetic_images = bool(benchmark_cfg.get("save_synthetic_images", True))
    save_limit = int(benchmark_cfg.get("save_synthetic_limit", 20))

    a_summary = summarize_images(candidate_paths(candidates_a), sample_n=min(20, len(candidates_a)))
    b_summary = summarize_images(candidate_paths(candidates_b), sample_n=min(20, len(candidates_b)))
    logger.info(
        "Benchmark pre-flight: A=%d B=%d pairs=%d x_values=%d nuisance_cases=%d",
        len(candidates_a),
        len(candidates_b),
        len(pair_indices),
        len(x_values),
        len(nuisance_cases),
    )
    logger.info("A dtypes=%s | B dtypes=%s", a_summary.get("sample_dtypes"), b_summary.get("sample_dtypes"))

    planned_total = len(pair_indices) * len(x_values) * len(nuisance_cases)
    if max_samples_int is not None:
        planned_total = min(planned_total, max_samples_int)
    progress = ProgressLogger(
        total=max(planned_total, 1),
        logger=logger,
        every=max(1, planned_total // 10) if planned_total > 0 else 1,
        unit="sample",
    )

    report_payload: Dict[str, Any] = {
        "run_id": output_dir.name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "status": "running",
        "progress": {"processed": 0, "total": planned_total, "percent": 0.0},
        "stage": "synthetic_robustness_benchmark",
        "primary_metric": primary_metric,
        "latest_sample": None,
    }
    write_report_json(output_dir, report_payload)

    results_jsonl = output_dir / "benchmark_results.jsonl"
    if results_jsonl.exists():
        results_jsonl.unlink()

    synthetic_dir = output_dir / "synthetic_samples"
    if save_synthetic_images:
        synthetic_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    saved_count = 0
    start = time.perf_counter()

    for pair_order, (idx_a, idx_b) in enumerate(pair_indices):
        candidate_a = candidates_a[idx_a]
        candidate_b = candidates_b[idx_b]
        target_shape = (
            min(candidate_a.image.shape[0], candidate_b.image.shape[0]),
            min(candidate_a.image.shape[1], candidate_b.image.shape[1]),
        )
        mask = build_centered_mask(target_shape) if preprocess_settings.mask_enabled else None

        try:
            prepared_a_true = _prepare_pattern_for_shape(candidate_a, target_shape, preprocess_settings, mask)
            prepared_b_true = _prepare_pattern_for_shape(candidate_b, target_shape, preprocess_settings, mask)
        except ValueError as exc:
            logger.warning("Skipping pair (%s,%s): %s", candidate_a.path.name, candidate_b.path.name, exc)
            continue

        prepared_pairs: List[tuple[PreparedPattern, PreparedPattern]] = [(prepared_a_true, prepared_b_true)]
        if not use_true_pair_only:
            prepared_pairs = []
            for search_idx_a, search_idx_b in pair_indices:
                try:
                    prep_a = _prepare_pattern_for_shape(
                        candidates_a[search_idx_a],
                        target_shape,
                        preprocess_settings,
                        mask,
                    )
                    prep_b = _prepare_pattern_for_shape(
                        candidates_b[search_idx_b],
                        target_shape,
                        preprocess_settings,
                        mask,
                    )
                    prepared_pairs.append((prep_a, prep_b))
                except ValueError:
                    continue
            if not prepared_pairs:
                logger.warning("No shape-compatible candidate pairs for target shape %s", target_shape)
                continue

        for x_true in x_values:
            clean_mix = _mix_patterns(prepared_a_true.raw_matched, prepared_b_true.raw_matched, x_true)
            for nuisance_index, nuisance_case in enumerate(nuisance_cases):
                mixed_noisy = _apply_nuisance(clean_mix, nuisance_case, rng=rng, mask=mask)
                mixed_processed, _ = preprocess_pattern(mixed_noisy, preprocess_settings, mask)

                best_pair: Optional[tuple[PreparedPattern, PreparedPattern]] = None
                best_eval: Optional[PairEvaluation] = None
                best_primary_score: Optional[float] = None
                best_l2_score: Optional[float] = None
                for prep_a, prep_b in prepared_pairs:
                    pair_eval = _evaluate_pair(
                        pattern_a=prep_a,
                        pattern_b=prep_b,
                        mixed_processed=mixed_processed,
                        mask=mask,
                        metric_names=metric_names,
                        metric_specs=metric_specs,
                        primary_metric=primary_metric,
                        search_settings=search_settings,
                        alignment_settings=alignment_settings,
                    )
                    score = pair_eval.best_by_metric[primary_metric].score
                    l2_metric = pair_eval.best_by_metric.get("l2")
                    l2_score = float(l2_metric.score) if l2_metric is not None else None
                    better = is_better(score, best_primary_score, metric_specs[primary_metric].objective)
                    tie_break = (
                        not better
                        and best_primary_score is not None
                        and abs(score - best_primary_score) <= 1e-9
                        and l2_score is not None
                        and best_l2_score is not None
                        and l2_score < best_l2_score
                    )
                    if better or tie_break or best_eval is None:
                        best_eval = pair_eval
                        best_pair = (prep_a, prep_b)
                        best_primary_score = score
                        if l2_score is not None:
                            best_l2_score = l2_score

                if best_eval is None or best_pair is None:
                    continue

                record: Dict[str, Any] = {
                    "pair_true": {
                        "a_id": prepared_a_true.record.pattern_id,
                        "b_id": prepared_b_true.record.pattern_id,
                    },
                    "pair_selected": {
                        "a_id": best_pair[0].record.pattern_id,
                        "b_id": best_pair[1].record.pattern_id,
                    },
                    "x_true": float(x_true),
                    "nuisance": {
                        "gain": nuisance_case.gain,
                        "offset": nuisance_case.offset,
                        "noise_std": nuisance_case.noise_std,
                        "blur_sigma": nuisance_case.blur_sigma,
                        "shift_x": nuisance_case.shift_x,
                        "shift_y": nuisance_case.shift_y,
                        "rotation_deg": nuisance_case.rotation_deg,
                    },
                    "best_by_metric": {},
                    "pair_order": pair_order,
                    "nuisance_index": nuisance_index,
                }
                for metric_name in metric_names:
                    metric_best = best_eval.best_by_metric[metric_name]
                    x_hat = float(metric_best.fraction)
                    signed_error = x_hat - float(x_true)
                    abs_error = abs(signed_error)
                    record["best_by_metric"][metric_name] = {
                        "x_hat": x_hat,
                        "score": float(metric_best.score),
                        "top_margin": metric_best.top_margin,
                    }
                    record[f"{metric_name}_signed_error"] = signed_error
                    record[f"{metric_name}_error"] = abs_error
                _append_jsonl(results_jsonl, record)
                records.append(record)

                if save_synthetic_images and saved_count < save_limit:
                    sample_stub = f"pair{pair_order:03d}_x{int(round(x_true * 1000)):03d}_n{nuisance_index:03d}"
                    write_image_16bit(synthetic_dir / f"{sample_stub}_A.png", np.clip(prepared_a_true.raw_matched, 0.0, 1.0))
                    write_image_16bit(synthetic_dir / f"{sample_stub}_B.png", np.clip(prepared_b_true.raw_matched, 0.0, 1.0))
                    write_image_16bit(synthetic_dir / f"{sample_stub}_C.png", np.clip(mixed_noisy, 0.0, 1.0))
                    saved_count += 1

                snapshot = progress.update(1)
                report_payload["progress"] = {
                    "processed": snapshot.processed,
                    "total": snapshot.total,
                    "percent": snapshot.percent,
                    "eta_s": snapshot.eta_s,
                }
                report_payload["latest_sample"] = {
                    "pair_true": record["pair_true"],
                    "x_true": record["x_true"],
                    "primary_x_hat": record["best_by_metric"][primary_metric]["x_hat"],
                    "primary_error": record[f"{primary_metric}_error"],
                }
                report_payload["timestamp"] = datetime.now().isoformat(timespec="seconds")
                write_report_json(output_dir, report_payload)

                if max_samples_int is not None and len(records) >= max_samples_int:
                    break
            if max_samples_int is not None and len(records) >= max_samples_int:
                break
        if max_samples_int is not None and len(records) >= max_samples_int:
            break

    if not records:
        raise ValueError("Synthetic benchmark produced no samples. Check shape compatibility and limits.")

    summary_rows = _metric_error_summary(records, metric_names)
    summary_csv = output_dir / "summary_metrics.csv"
    _write_csv(
        summary_csv,
        summary_rows,
        fieldnames=["metric", "samples", "mae", "rmse", "bias", "std"],
    )

    summary_plots_dir = output_dir / "summary_plots"
    mae_plot = summary_plots_dir / "mae_by_metric.png"
    _plot_metric_mae(mae_plot, summary_rows)

    html_report = output_dir / "report" / "index.html"
    _write_html_summary(html_report, summary_rows, record_count=len(records), primary_metric=primary_metric)

    runtime_s = time.perf_counter() - start
    report_payload["status"] = "completed"
    report_payload["summary"] = {
        "records": len(records),
        "pairs_used": len(pair_indices),
        "x_values": len(x_values),
        "nuisance_cases": len(nuisance_cases),
        "runtime_s": runtime_s,
        "primary_metric": primary_metric,
        "best_metric_by_mae": summary_rows[0]["metric"] if summary_rows else None,
    }
    report_payload["artifacts"] = {
        "benchmark_results_jsonl": safe_relpath(results_jsonl, output_dir),
        "summary_metrics_csv": safe_relpath(summary_csv, output_dir),
        "mae_plot": safe_relpath(mae_plot, output_dir),
        "html_report": safe_relpath(html_report, output_dir),
        "synthetic_samples_dir": safe_relpath(synthetic_dir, output_dir) if save_synthetic_images else None,
    }
    report_payload["timestamp"] = datetime.now().isoformat(timespec="seconds")
    write_report_json(output_dir, report_payload)

    logger.info(
        "Benchmark complete: records=%d | best metric by MAE=%s | runtime=%.2fs",
        len(records),
        summary_rows[0]["metric"] if summary_rows else "n/a",
        runtime_s,
    )
    return {
        "records": len(records),
        "pairs_used": len(pair_indices),
        "x_values": len(x_values),
        "nuisance_cases": len(nuisance_cases),
        "runtime_s": runtime_s,
        "summary_metrics_csv": str(summary_csv),
        "benchmark_results_jsonl": str(results_jsonl),
        "best_metric_by_mae": summary_rows[0]["metric"] if summary_rows else None,
    }

