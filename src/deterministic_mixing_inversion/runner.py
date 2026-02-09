"""Runner for deterministic mixing-fraction inversion."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import csv
import logging
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.ndimage import rotate

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
    build_unique_pair_indices,
    candidate_paths,
    load_pattern,
    load_candidate_pool,
    load_candidate_pools,
    load_mixed_patterns,
)
from src.deterministic_mixing_inversion.metrics import (
    MetricSpec,
    compute_metric_values,
    is_better,
    masked_l2,
    masked_ncc,
    parse_metric_config,
)
from src.deterministic_mixing_inversion.preprocess import (
    PreprocessSettings,
    build_centered_mask,
    match_shape_to_target,
    parse_preprocess_settings,
    preprocess_pattern,
)
from src.deterministic_mixing_inversion.reporting import (
    append_jsonl,
    plot_primary_fraction_histogram,
    update_progress_report,
    write_candidate_pool_html_report,
    write_candidate_pool_summary_csv,
    write_html_report,
    write_metric_summary_csv,
    write_score_curve_csv,
    write_synthetic_pair_html_report,
    write_synthetic_pair_summary_csv,
)
from src.deterministic_mixing_inversion.search import (
    MetricBest,
    SearchSettings,
    parse_search_settings,
    search_mixing_fraction,
)
from src.utils.io import write_image_16bit
from src.utils.logging import ProgressLogger, summarize_images
from src.utils.reporting import make_qual_grid, safe_relpath


@dataclass(frozen=True)
class PreparedPattern:
    """Prepared pattern cache entry."""

    record: PatternRecord
    raw_matched: np.ndarray
    processed: np.ndarray
    preprocess_meta: Dict[str, object]


@dataclass(frozen=True)
class PairEvaluation:
    """One evaluated candidate pair."""

    best_by_metric: Dict[str, MetricBest]
    score_curves: Dict[str, List[tuple[float, float]]]
    evaluated_points: int
    alignment: RigidAlignment


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _sanitize_name(text: str) -> str:
    return text.replace("/", "_").replace("\\", "_").replace(" ", "_")


def _mix_patterns(image_a: np.ndarray, image_b: np.ndarray, mix_fraction: float) -> np.ndarray:
    return (mix_fraction * image_a + (1.0 - mix_fraction) * image_b).astype(np.float32)


def _resolve_data_mode(data_cfg: Dict[str, Any]) -> str:
    mode_raw = data_cfg.get("mode")
    if mode_raw is not None:
        return str(mode_raw).strip().lower()
    synthetic_cfg = data_cfg.get("synthetic_pair")
    if isinstance(synthetic_cfg, dict) and synthetic_cfg.get("a_path") and synthetic_cfg.get("b_path"):
        return "synthetic_pair"
    if data_cfg.get("a_path") and data_cfg.get("b_path"):
        return "synthetic_pair"
    return "mixed_dir"


def _resolve_candidate_pool_x_values(
    candidate_cfg: Dict[str, Any],
    pairs_requested: int,
    rng: np.random.Generator,
) -> List[float]:
    x_true_raw = candidate_cfg.get("x_true", 0.5)
    if isinstance(x_true_raw, (list, tuple)):
        if not x_true_raw:
            raise ValueError("candidate_pool.x_true must be non-empty when provided as a list.")
        x_values = [float(np.clip(value, 0.0, 1.0)) for value in x_true_raw]
    else:
        x_values = [float(np.clip(float(x_true_raw), 0.0, 1.0))]

    if len(x_values) >= pairs_requested:
        return x_values[:pairs_requested]
    if len(x_values) == 1:
        return [x_values[0]] * pairs_requested
    return [float(rng.choice(x_values)) for _ in range(pairs_requested)]


def _parse_candidate_pool_noise_settings(
    candidate_cfg: Dict[str, Any],
    debug_enabled: bool,
    debug_seed: int,
) -> Dict[str, Any]:
    noise_cfg = candidate_cfg.get("noise", {}) if isinstance(candidate_cfg, dict) else {}
    if noise_cfg is None:
        noise_cfg = {}
    if not isinstance(noise_cfg, dict):
        raise ValueError("candidate_pool.noise must be a mapping.")
    enabled = bool(noise_cfg.get("enabled", False))
    gaussian_std = float(noise_cfg.get("gaussian_std", 0.0))
    rotation_deg_max = float(noise_cfg.get("rotation_deg_max", 0.0))
    seed = noise_cfg.get("seed")
    if seed is None and debug_enabled:
        seed = debug_seed
    return {
        "enabled": enabled,
        "gaussian_std": max(0.0, gaussian_std),
        "rotation_deg_max": max(0.0, rotation_deg_max),
        "seed": seed,
    }


def _resolve_synthetic_pair_inputs(data_cfg: Dict[str, Any]) -> tuple[Path, Path, List[float]]:
    synthetic_cfg = data_cfg.get("synthetic_pair", {})
    if synthetic_cfg is None:
        synthetic_cfg = {}
    if not isinstance(synthetic_cfg, dict):
        raise ValueError("data.synthetic_pair must be a mapping.")

    a_path_raw = synthetic_cfg.get("a_path") or data_cfg.get("a_path")
    b_path_raw = synthetic_cfg.get("b_path") or data_cfg.get("b_path")
    if not a_path_raw or not b_path_raw:
        raise ValueError("Synthetic pair mode requires data.synthetic_pair.a_path and data.synthetic_pair.b_path.")
    x_true_raw = synthetic_cfg.get("x_true", 0.5)
    if isinstance(x_true_raw, (list, tuple)):
        if not x_true_raw:
            raise ValueError("data.synthetic_pair.x_true must be a non-empty list when provided as a list.")
        x_values = [float(np.clip(value, 0.0, 1.0)) for value in x_true_raw]
    else:
        x_values = [float(np.clip(float(x_true_raw), 0.0, 1.0))]

    return Path(str(a_path_raw)), Path(str(b_path_raw)), x_values


def _parse_synthetic_noise_settings(
    data_cfg: Dict[str, Any],
    debug_enabled: bool,
    debug_seed: int,
) -> Dict[str, Any]:
    synthetic_cfg = data_cfg.get("synthetic_pair", {})
    noise_cfg = synthetic_cfg.get("noise", {}) if isinstance(synthetic_cfg, dict) else {}
    if noise_cfg is None:
        noise_cfg = {}
    if not isinstance(noise_cfg, dict):
        raise ValueError("data.synthetic_pair.noise must be a mapping.")
    enabled = bool(noise_cfg.get("enabled", False))
    gaussian_std = float(noise_cfg.get("gaussian_std", 0.0))
    rotation_deg_max = float(noise_cfg.get("rotation_deg_max", 0.0))
    seed = noise_cfg.get("seed")
    if seed is None and debug_enabled:
        seed = debug_seed
    return {
        "enabled": enabled,
        "gaussian_std": max(0.0, gaussian_std),
        "rotation_deg_max": max(0.0, rotation_deg_max),
        "seed": seed,
    }


def _apply_input_noise(
    image: np.ndarray,
    settings: Dict[str, Any],
    rng: np.random.Generator,
    mask: np.ndarray | None,
) -> tuple[np.ndarray, Dict[str, float]]:
    if not settings.get("enabled", False):
        if mask is not None:
            image = image.copy()
            image[~mask] = 0.0
        return image.astype(np.float32), {"rotation_deg": 0.0, "gaussian_std": 0.0}

    working = image.astype(np.float32, copy=True)
    rotation_max = float(settings.get("rotation_deg_max", 0.0))
    rotation_deg = 0.0
    if rotation_max > 0.0:
        rotation_deg = float(rng.uniform(-rotation_max, rotation_max))
        working = rotate(
            working,
            angle=rotation_deg,
            reshape=False,
            order=3,
            mode="constant",
            cval=0.0,
            prefilter=False,
        ).astype(np.float32)

    gaussian_std = float(settings.get("gaussian_std", 0.0))
    if gaussian_std > 0.0:
        noise = rng.normal(loc=0.0, scale=gaussian_std, size=working.shape).astype(np.float32)
        working = working + noise

    if mask is not None:
        working = working.copy()
        working[~mask] = 0.0
    working = np.clip(working, 0.0, 1.0).astype(np.float32)
    return working, {"rotation_deg": rotation_deg, "gaussian_std": gaussian_std}


def _prepare_pattern_for_sample(
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
            score_function=masked_ncc,
        )
    else:
        best_alignment = RigidAlignment(
            angle_deg=0.0,
            shift_y=0.0,
            shift_x=0.0,
            score=masked_ncc(initial_reconstruction, mixed_processed, mask),
        )

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
        score_curves=final_search.score_curves,
        evaluated_points=final_search.evaluated_points,
        alignment=best_alignment,
    )


def _build_sample_result(
    sample_record: PatternRecord,
    best_pair_a: PreparedPattern,
    best_pair_b: PreparedPattern,
    pair_evaluation: PairEvaluation,
    primary_metric: str,
    mask: np.ndarray | None,
) -> Dict[str, Any]:
    primary_payload = pair_evaluation.best_by_metric[primary_metric]
    raw_reconstruction = _mix_patterns(
        best_pair_a.raw_matched,
        best_pair_b.raw_matched,
        primary_payload.fraction,
    )
    aligned_reconstruction = apply_rigid_alignment(
        raw_reconstruction,
        pair_evaluation.alignment,
        interpolation_order=3,
        mask=mask,
    )
    residual_l2 = masked_l2(
        aligned_reconstruction,
        sample_record.image[: aligned_reconstruction.shape[0], : aligned_reconstruction.shape[1]],
        mask,
    )
    metric_payload = {
        metric_name: {
            "x_hat": float(metric_result.fraction),
            "score": float(metric_result.score),
            "top_margin": metric_result.top_margin,
        }
        for metric_name, metric_result in pair_evaluation.best_by_metric.items()
    }
    return {
        "sample_id": sample_record.pattern_id,
        "sample_path": str(sample_record.path),
        "best_pair": {
            "a_id": best_pair_a.record.pattern_id,
            "a_path": str(best_pair_a.record.path),
            "b_id": best_pair_b.record.pattern_id,
            "b_path": str(best_pair_b.record.path),
        },
        "best_by_metric": metric_payload,
        "alignment": {
            "angle_deg": float(pair_evaluation.alignment.angle_deg),
            "shift_y": float(pair_evaluation.alignment.shift_y),
            "shift_x": float(pair_evaluation.alignment.shift_x),
            "alignment_score": float(pair_evaluation.alignment.score),
        },
        "evaluated_points": int(pair_evaluation.evaluated_points),
        "residual_l2_primary": float(residual_l2),
    }


def _save_sample_artifacts(
    output_dir: Path,
    sample_record: PatternRecord,
    mixed_matched: np.ndarray,
    best_pair_a: PreparedPattern,
    best_pair_b: PreparedPattern,
    pair_evaluation: PairEvaluation,
    primary_metric: str,
    mask: np.ndarray | None,
    write_curves: bool,
) -> Dict[str, str]:
    artifact_paths: Dict[str, str] = {}
    sample_stub = _sanitize_name(sample_record.pattern_id)
    pair_stub = f"{_sanitize_name(best_pair_a.record.pattern_id)}__{_sanitize_name(best_pair_b.record.pattern_id)}"

    primary_fraction = pair_evaluation.best_by_metric[primary_metric].fraction
    raw_reconstruction = _mix_patterns(
        best_pair_a.raw_matched,
        best_pair_b.raw_matched,
        primary_fraction,
    )
    aligned_reconstruction = apply_rigid_alignment(
        raw_reconstruction,
        pair_evaluation.alignment,
        interpolation_order=3,
        mask=mask,
    )
    recon_dir = output_dir / "reconstructions"
    recon_dir.mkdir(parents=True, exist_ok=True)
    reconstruction_path = recon_dir / f"{sample_stub}__{pair_stub}__C_hat.png"
    write_image_16bit(reconstruction_path, np.clip(aligned_reconstruction, 0.0, 1.0))
    artifact_paths["reconstruction"] = safe_relpath(reconstruction_path, output_dir)

    mixed_path = recon_dir / f"{sample_stub}__C.png"
    write_image_16bit(mixed_path, np.clip(mixed_matched, 0.0, 1.0))
    artifact_paths["mixed"] = safe_relpath(mixed_path, output_dir)

    monitoring_dir = output_dir / "monitoring" / "qualitative"
    monitoring_dir.mkdir(parents=True, exist_ok=True)
    panel_path = monitoring_dir / f"{sample_stub}__{pair_stub}.png"
    make_qual_grid(
        mixed_matched,
        best_pair_a.raw_matched,
        best_pair_b.raw_matched,
        None,
        None,
        aligned_reconstruction,
        panel_path,
    )
    artifact_paths["qual_panel"] = safe_relpath(panel_path, output_dir)

    if write_curves:
        curves_dir = output_dir / "curves"
        curves_dir.mkdir(parents=True, exist_ok=True)
        curves_path = curves_dir / f"{sample_stub}__{pair_stub}.csv"
        write_score_curve_csv(curves_path, pair_evaluation.score_curves)
        artifact_paths["score_curve_csv"] = safe_relpath(curves_path, output_dir)
    return artifact_paths


def run_deterministic_inversion(
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Run deterministic mixing-fraction inversion for a batch of mixed patterns."""
    if logger is None:
        logger = logging.getLogger(__name__)

    data_cfg = config.get("data", {})
    output_cfg = config.get("output", {})
    debug_cfg = config.get("debug", {})
    inversion_cfg = config.get("deterministic_inversion", {})

    output_dir = Path(output_cfg.get("out_dir", "outputs/deterministic_inversion"))
    output_dir.mkdir(parents=True, exist_ok=True)

    debug_enabled = bool(debug_cfg.get("enabled", False))
    debug_seed = int(debug_cfg.get("seed", 42))
    if debug_enabled:
        _set_seed(debug_seed)
        logger.info("Debug mode enabled with seed=%d", debug_seed)

    sample_limit = debug_cfg.get("sample_limit") if debug_enabled else None
    max_pairs = debug_cfg.get("max_pairs") if debug_enabled else None

    data_mode = _resolve_data_mode(data_cfg)
    candidate_cfg = data_cfg.get("candidate_pool", {})

    preprocess_settings = parse_preprocess_settings(inversion_cfg.get("preprocess", {}))
    metric_names, primary_metric, metric_specs = parse_metric_config(inversion_cfg.get("metrics", {}))
    search_settings = parse_search_settings(inversion_cfg.get("search", {}))
    alignment_settings = parse_alignment_settings(inversion_cfg.get("alignment", {}))

    if data_mode == "synthetic_pair":
        a_path, b_path, x_values = _resolve_synthetic_pair_inputs(data_cfg)
        if not a_path.exists():
            raise FileNotFoundError(f"Synthetic A pattern not found: {a_path}")
        if not b_path.exists():
            raise FileNotFoundError(f"Synthetic B pattern not found: {b_path}")

        record_a = load_pattern(a_path)
        record_b = load_pattern(b_path)
        if record_a.image.shape != record_b.image.shape:
            if not preprocess_settings.auto_crop_to_target:
                raise ValueError(
                    "Synthetic pair shapes do not match. Enable deterministic_inversion.preprocess.auto_crop_to_target "
                    "or provide shape-matched inputs."
                )
            target_shape = (
                min(record_a.image.shape[0], record_b.image.shape[0]),
                min(record_a.image.shape[1], record_b.image.shape[1]),
            )
        else:
            target_shape = record_a.image.shape

        mask = build_centered_mask(target_shape) if preprocess_settings.mask_enabled else None
        prepared_clean_a = _prepare_pattern_for_sample(record_a, target_shape, preprocess_settings, mask)
        prepared_clean_b = _prepare_pattern_for_sample(record_b, target_shape, preprocess_settings, mask)

        noise_settings = _parse_synthetic_noise_settings(data_cfg, debug_enabled, debug_seed)
        rng = np.random.default_rng(noise_settings.get("seed"))
        noisy_a_image, noisy_a_meta = _apply_input_noise(prepared_clean_a.raw_matched, noise_settings, rng, mask)
        noisy_b_image, noisy_b_meta = _apply_input_noise(prepared_clean_b.raw_matched, noise_settings, rng, mask)
        noise_payload = {
            "enabled": bool(noise_settings.get("enabled", False)),
            "gaussian_std": float(noise_settings.get("gaussian_std", 0.0)),
            "rotation_deg_max": float(noise_settings.get("rotation_deg_max", 0.0)),
            "seed": noise_settings.get("seed"),
            "a_rotation_deg": float(noisy_a_meta.get("rotation_deg", 0.0)),
            "b_rotation_deg": float(noisy_b_meta.get("rotation_deg", 0.0)),
        }
        noisy_a_record = PatternRecord(
            pattern_id=f"{record_a.pattern_id}_noisy",
            path=record_a.path,
            image=noisy_a_image,
            source_dtype=record_a.source_dtype,
        )
        noisy_b_record = PatternRecord(
            pattern_id=f"{record_b.pattern_id}_noisy",
            path=record_b.path,
            image=noisy_b_image,
            source_dtype=record_b.source_dtype,
        )
        prepared_a = _prepare_pattern_for_sample(noisy_a_record, target_shape, preprocess_settings, mask)
        prepared_b = _prepare_pattern_for_sample(noisy_b_record, target_shape, preprocess_settings, mask)

        logger.info(
            "Synthetic pair: A=%s B=%s | x_true=%s | shape=%s | metrics=%s | search=%s",
            a_path.name,
            b_path.name,
            [float(x) for x in x_values],
            target_shape,
            metric_names,
            search_settings.strategy,
        )

        results_jsonl = output_dir / "results.jsonl"
        if results_jsonl.exists():
            results_jsonl.unlink()

        inputs_dir = output_dir / "synthetic_inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        input_a_path = inputs_dir / "A.png"
        input_b_path = inputs_dir / "B.png"
        write_image_16bit(input_a_path, np.clip(prepared_clean_a.raw_matched, 0.0, 1.0))
        write_image_16bit(input_b_path, np.clip(prepared_clean_b.raw_matched, 0.0, 1.0))
        input_a_noisy_path = inputs_dir / "A_noisy.png"
        input_b_noisy_path = inputs_dir / "B_noisy.png"
        write_image_16bit(input_a_noisy_path, np.clip(prepared_a.raw_matched, 0.0, 1.0))
        write_image_16bit(input_b_noisy_path, np.clip(prepared_b.raw_matched, 0.0, 1.0))
        payloads: List[Dict[str, Any]] = []
        all_objective_results: List[Dict[str, Any]] = []
        recon_dir = output_dir / "reconstructions"
        recon_dir.mkdir(parents=True, exist_ok=True)
        monitoring_dir = output_dir / "monitoring" / "qualitative"
        monitoring_dir.mkdir(parents=True, exist_ok=True)
        curves_dir = output_dir / "curves"
        if bool(output_cfg.get("save_curves", True)):
            curves_dir.mkdir(parents=True, exist_ok=True)

        for x_true in x_values:
            mixed_true = _mix_patterns(prepared_clean_a.raw_matched, prepared_clean_b.raw_matched, x_true)
            if mask is not None:
                mixed_true = mixed_true.copy()
                mixed_true[~mask] = 0.0
            mixed_processed, _ = preprocess_pattern(mixed_true, preprocess_settings, mask)

            x_tag = f"{int(round(x_true * 1000)):04d}"
            input_c_path = inputs_dir / f"C_x{x_tag}.png"
            write_image_16bit(input_c_path, np.clip(mixed_true, 0.0, 1.0))

            objective_results: List[Dict[str, Any]] = []
            for objective_metric in metric_names:
                objective_spec = {objective_metric: metric_specs[objective_metric]}
                pair_eval = _evaluate_pair(
                    pattern_a=prepared_a,
                    pattern_b=prepared_b,
                    mixed_processed=mixed_processed,
                    mask=mask,
                    metric_names=[objective_metric],
                    metric_specs=objective_spec,
                    primary_metric=objective_metric,
                    search_settings=search_settings,
                    alignment_settings=alignment_settings,
                )
                best = pair_eval.best_by_metric[objective_metric]
                x_hat = float(best.fraction)

                reconstruction_processed = _mix_patterns(prepared_a.processed, prepared_b.processed, x_hat)
                reconstruction_processed = apply_rigid_alignment(
                    reconstruction_processed,
                    pair_eval.alignment,
                    interpolation_order=alignment_settings.interpolation_order,
                    mask=mask,
                )
                metrics_at_x = compute_metric_values(reconstruction_processed, mixed_processed, mask, metric_names)

                reconstruction_raw = _mix_patterns(prepared_a.raw_matched, prepared_b.raw_matched, x_hat)
                reconstruction_raw = apply_rigid_alignment(
                    reconstruction_raw,
                    pair_eval.alignment,
                    interpolation_order=alignment_settings.interpolation_order,
                    mask=mask,
                )

                c_hat_path = recon_dir / f"C_hat__opt_{objective_metric}__x{x_tag}.png"
                write_image_16bit(c_hat_path, np.clip(reconstruction_raw, 0.0, 1.0))

                qual_path = monitoring_dir / f"synthetic_pair__opt_{objective_metric}__x{x_tag}.png"
                make_qual_grid(
                    mixed_true,
                    prepared_clean_a.raw_matched,
                    prepared_clean_b.raw_matched,
                    prepared_a.raw_matched,
                    prepared_b.raw_matched,
                    reconstruction_raw,
                    qual_path,
                )

                curve_csv_path: Optional[Path] = None
                if bool(output_cfg.get("save_curves", True)):
                    curve_csv_path = curves_dir / f"score_curve__opt_{objective_metric}__x{x_tag}.csv"
                    write_score_curve_csv(curve_csv_path, pair_eval.score_curves)

                record = {
                    "objective_metric": objective_metric,
                    "x_true": float(x_true),
                    "x_hat": x_hat,
                    "x_signed_error": float(x_hat - float(x_true)),
                    "x_abs_error": float(abs(x_hat - float(x_true))),
                    "objective_score": float(best.score),
                    "top_margin": best.top_margin,
                    "noise_gaussian_std": noise_payload["gaussian_std"],
                    "noise_a_rotation_deg": noise_payload["a_rotation_deg"],
                    "noise_b_rotation_deg": noise_payload["b_rotation_deg"],
                    "metrics": {name: float(metrics_at_x[name]) for name in metric_names},
                    "evaluated_points": int(pair_eval.evaluated_points),
                    "alignment": {
                        "angle_deg": float(pair_eval.alignment.angle_deg),
                        "shift_y": float(pair_eval.alignment.shift_y),
                        "shift_x": float(pair_eval.alignment.shift_x),
                        "alignment_score": float(pair_eval.alignment.score),
                    },
                    "artifacts": {
                        "c_hat": safe_relpath(c_hat_path, output_dir),
                        "qual_panel": safe_relpath(qual_path, output_dir),
                        "score_curve_csv": safe_relpath(curve_csv_path, output_dir) if curve_csv_path else None,
                    },
                }
                objective_results.append(record)
                all_objective_results.append(record)

            summary_payload: Dict[str, Any] = {
                "mode": "synthetic_pair",
                "sample_id": f"x_true_{x_tag}",
                "inputs": {
                    "a_path": str(a_path),
                    "b_path": str(b_path),
                    "x_true": float(x_true),
                    "noise": noise_payload,
                    "artifacts": {
                        "a": safe_relpath(input_a_path, output_dir),
                        "b": safe_relpath(input_b_path, output_dir),
                        "a_noisy": safe_relpath(input_a_noisy_path, output_dir),
                        "b_noisy": safe_relpath(input_b_noisy_path, output_dir),
                        "c": safe_relpath(input_c_path, output_dir),
                    },
                },
                "metrics_enabled": metric_names,
                "objective_results": objective_results,
            }
            append_jsonl(results_jsonl, summary_payload)
            payloads.append(summary_payload)

        summary_csv_path = output_dir / "summary_metrics.csv"
        write_synthetic_pair_summary_csv(summary_csv_path, all_objective_results, metric_names)

        html_path: Optional[Path] = None
        if bool(output_cfg.get("write_html_report", True)):
            html_path = output_dir / "report" / "index.html"
            write_synthetic_pair_html_report(html_path, output_dir=output_dir, payloads=payloads)

        report_payload: Dict[str, Any] = {
            "run_id": output_dir.name,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "status": "completed",
            "stage": "deterministic_synthetic_pair",
            "mode": "synthetic_pair",
            "metrics_enabled": metric_names,
            "summary": {
                "processed": len(x_values),
                "candidate_pairs": 1,
                "objective_metrics": len(metric_names),
                "x_true_values": len(x_values),
            },
            "artifacts": {
                "results_jsonl": safe_relpath(results_jsonl, output_dir),
                "inputs_dir": safe_relpath(inputs_dir, output_dir),
                "reconstructions_dir": safe_relpath(output_dir / "reconstructions", output_dir),
                "summary_metrics_csv": safe_relpath(summary_csv_path, output_dir),
                "html_report": safe_relpath(html_path, output_dir) if html_path else None,
            },
        }
        update_progress_report(output_dir, report_payload)

        return {
            "mode": "synthetic_pair",
            "processed": len(x_values),
            "total_mixed": len(x_values),
            "candidate_pairs": 1,
            "objective_metrics": len(metric_names),
            "x_true_values": len(x_values),
            "results_jsonl": str(results_jsonl),
            "summary_metrics_csv": str(summary_csv_path),
            "html_report": str(html_path) if html_path else None,
        }

    if data_mode == "candidate_pool_synthetic":
        candidate_cfg = data_cfg.get("candidate_pool", {})
        if not isinstance(candidate_cfg, dict):
            raise ValueError("data.candidate_pool must be a mapping for candidate_pool_synthetic mode.")

        pairs_requested = int(candidate_cfg.get("synthetic_pairs", 1))
        if pairs_requested <= 0:
            raise ValueError("candidate_pool.synthetic_pairs must be a positive integer.")

        sample_seed = candidate_cfg.get("sample_seed")
        if sample_seed is None and debug_enabled:
            sample_seed = debug_seed
        synthetic_seed = candidate_cfg.get("synthetic_seed")
        if synthetic_seed is None and debug_enabled:
            synthetic_seed = debug_seed

        candidates_clean = load_candidate_pool(candidate_cfg, logger, sample_seed)
        if len(candidates_clean) < 2:
            raise ValueError("candidate_pool_synthetic requires at least two candidate patterns.")

        shapes = [record.image.shape for record in candidates_clean]
        if any(shape != shapes[0] for shape in shapes):
            if not preprocess_settings.auto_crop_to_target:
                raise ValueError(
                    "Candidate pool shapes do not match. Enable deterministic_inversion.preprocess.auto_crop_to_target "
                    "or provide shape-matched inputs."
                )
            target_shape = (
                min(shape[0] for shape in shapes),
                min(shape[1] for shape in shapes),
            )
        else:
            target_shape = shapes[0]

        mask = build_centered_mask(target_shape) if preprocess_settings.mask_enabled else None

        noise_settings = _parse_candidate_pool_noise_settings(candidate_cfg, debug_enabled, debug_seed)
        noise_rng = np.random.default_rng(noise_settings.get("seed"))

        prepared_clean: List[PreparedPattern] = []
        prepared_noisy: List[PreparedPattern] = []
        noise_meta: Dict[str, Dict[str, float]] = {}
        for record in candidates_clean:
            clean_prepared = _prepare_pattern_for_sample(record, target_shape, preprocess_settings, mask)
            prepared_clean.append(clean_prepared)
            noisy_image, noisy_info = _apply_input_noise(
                clean_prepared.raw_matched,
                noise_settings,
                noise_rng,
                mask,
            )
            noise_meta[clean_prepared.record.pattern_id] = {
                "rotation_deg": float(noisy_info.get("rotation_deg", 0.0)),
                "gaussian_std": float(noisy_info.get("gaussian_std", 0.0)),
            }
            noisy_record = PatternRecord(
                pattern_id=clean_prepared.record.pattern_id,
                path=clean_prepared.record.path,
                image=noisy_image,
                source_dtype=clean_prepared.record.source_dtype,
            )
            prepared_noisy.append(_prepare_pattern_for_sample(noisy_record, target_shape, preprocess_settings, mask))

        pair_indices = build_unique_pair_indices(
            prepared_clean,
            int(max_pairs) if max_pairs else None,
        )
        if not pair_indices:
            raise ValueError("No candidate pairs available to evaluate in candidate_pool_synthetic mode.")

        pair_rng = np.random.default_rng(synthetic_seed)
        if pairs_requested > len(pair_indices):
            logger.warning(
                "Requested synthetic_pairs=%d exceeds available pairs=%d; reducing to available count.",
                pairs_requested,
                len(pair_indices),
            )
            pairs_requested = len(pair_indices)

        chosen_pair_indices = pair_rng.choice(len(pair_indices), size=pairs_requested, replace=False)
        x_values = _resolve_candidate_pool_x_values(candidate_cfg, pairs_requested, pair_rng)

        logger.info(
            "Candidate pool synthetic: candidates=%d pairs=%d trials=%d | metrics=%s primary=%s | noise=%s",
            len(prepared_clean),
            len(pair_indices),
            pairs_requested,
            metric_names,
            primary_metric,
            noise_settings.get("enabled", False),
        )

        start_time = time.perf_counter()

        candidate_manifest_path = output_dir / "candidate_pool.csv"
        with candidate_manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["index", "pattern_id", "path", "noise_rotation_deg", "noise_gaussian_std"],
            )
            writer.writeheader()
            for idx, prepared in enumerate(prepared_clean):
                meta = noise_meta.get(prepared.record.pattern_id, {})
                writer.writerow(
                    {
                        "index": idx,
                        "pattern_id": prepared.record.pattern_id,
                        "path": str(prepared.record.path),
                        "noise_rotation_deg": meta.get("rotation_deg", 0.0),
                        "noise_gaussian_std": meta.get("gaussian_std", 0.0),
                    }
                )

        results_jsonl = output_dir / "results.jsonl"
        if results_jsonl.exists():
            results_jsonl.unlink()

        report_payload: Dict[str, Any] = {
            "run_id": output_dir.name,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "status": "running",
            "stage": "deterministic_candidate_pool",
            "mode": "candidate_pool_synthetic",
            "metrics_enabled": metric_names,
            "primary_metric": primary_metric,
            "progress": {"processed": 0, "total": pairs_requested, "percent": 0.0},
            "latest_trial": None,
            "artifacts": {"candidate_pool_csv": safe_relpath(candidate_manifest_path, output_dir)},
        }
        update_progress_report(output_dir, report_payload)

        trial_results: List[Dict[str, Any]] = []
        monitoring_dir = output_dir / "monitoring" / "qualitative"
        monitoring_dir.mkdir(parents=True, exist_ok=True)
        trials_dir = output_dir / "candidate_trials"
        trials_dir.mkdir(parents=True, exist_ok=True)
        curves_dir = output_dir / "curves"
        if bool(output_cfg.get("save_curves", True)):
            curves_dir.mkdir(parents=True, exist_ok=True)

        for trial_idx, pair_choice_index in enumerate(chosen_pair_indices):
            idx_a, idx_b = pair_indices[int(pair_choice_index)]
            true_a_clean = prepared_clean[idx_a]
            true_b_clean = prepared_clean[idx_b]
            x_true = float(x_values[trial_idx])

            mixed_true = _mix_patterns(true_a_clean.raw_matched, true_b_clean.raw_matched, x_true)
            if mask is not None:
                mixed_true = mixed_true.copy()
                mixed_true[~mask] = 0.0
            mixed_processed, _ = preprocess_pattern(mixed_true, preprocess_settings, mask)

            best_pair_a: Optional[PreparedPattern] = None
            best_pair_b: Optional[PreparedPattern] = None
            best_pair_eval: Optional[PairEvaluation] = None
            best_primary_score: Optional[float] = None
            best_secondary_l2: Optional[float] = None
            best_idx_a: Optional[int] = None
            best_idx_b: Optional[int] = None

            for idx_a_candidate, idx_b_candidate in pair_indices:
                pattern_a = prepared_noisy[idx_a_candidate]
                pattern_b = prepared_noisy[idx_b_candidate]
                pair_eval = _evaluate_pair(
                    pattern_a=pattern_a,
                    pattern_b=pattern_b,
                    mixed_processed=mixed_processed,
                    mask=mask,
                    metric_names=metric_names,
                    metric_specs=metric_specs,
                    primary_metric=primary_metric,
                    search_settings=search_settings,
                    alignment_settings=alignment_settings,
                )
                primary_score = pair_eval.best_by_metric[primary_metric].score
                primary_objective = metric_specs[primary_metric].objective
                primary_is_better = is_better(primary_score, best_primary_score, primary_objective)

                candidate_l2 = pair_eval.best_by_metric.get("l2")
                candidate_l2_score = float(candidate_l2.score) if candidate_l2 is not None else None
                tie_breaker = False
                if (
                    not primary_is_better
                    and best_primary_score is not None
                    and abs(primary_score - best_primary_score) <= 1e-9
                    and candidate_l2_score is not None
                    and best_secondary_l2 is not None
                ):
                    tie_breaker = candidate_l2_score < best_secondary_l2

                if primary_is_better or tie_breaker or best_pair_eval is None:
                    best_pair_a = pattern_a
                    best_pair_b = pattern_b
                    best_pair_eval = pair_eval
                    best_primary_score = primary_score
                    if candidate_l2_score is not None:
                        best_secondary_l2 = candidate_l2_score
                    best_idx_a = idx_a_candidate
                    best_idx_b = idx_b_candidate

            if (
                best_pair_a is None
                or best_pair_b is None
                or best_pair_eval is None
                or best_idx_a is None
                or best_idx_b is None
            ):
                raise RuntimeError("No valid candidate pair produced a score in candidate_pool_synthetic mode.")

            primary_fraction = best_pair_eval.best_by_metric[primary_metric].fraction
            reconstruction_processed = _mix_patterns(best_pair_a.processed, best_pair_b.processed, primary_fraction)
            reconstruction_processed = apply_rigid_alignment(
                reconstruction_processed,
                best_pair_eval.alignment,
                interpolation_order=alignment_settings.interpolation_order,
                mask=mask,
            )
            metrics_at_primary = compute_metric_values(reconstruction_processed, mixed_processed, mask, metric_names)

            reconstruction_raw = _mix_patterns(best_pair_a.raw_matched, best_pair_b.raw_matched, primary_fraction)
            reconstruction_raw = apply_rigid_alignment(
                reconstruction_raw,
                best_pair_eval.alignment,
                interpolation_order=alignment_settings.interpolation_order,
                mask=mask,
            )

            objective_results: List[Dict[str, Any]] = []
            for objective_metric in metric_names:
                metric_best = best_pair_eval.best_by_metric[objective_metric]
                objective_fraction = float(metric_best.fraction)
                objective_recon = _mix_patterns(best_pair_a.processed, best_pair_b.processed, objective_fraction)
                objective_recon = apply_rigid_alignment(
                    objective_recon,
                    best_pair_eval.alignment,
                    interpolation_order=alignment_settings.interpolation_order,
                    mask=mask,
                )
                objective_metrics = compute_metric_values(objective_recon, mixed_processed, mask, metric_names)
                objective_results.append(
                    {
                        "objective_metric": objective_metric,
                        "x_hat": objective_fraction,
                        "objective_score": float(metric_best.score),
                        "top_margin": metric_best.top_margin,
                        "metrics": {name: float(objective_metrics[name]) for name in metric_names},
                    }
                )

            trial_stub = f"trial_{trial_idx + 1:02d}"
            trial_dir = trials_dir / trial_stub
            trial_dir.mkdir(parents=True, exist_ok=True)

            true_a_clean_path = trial_dir / "A_true.png"
            true_b_clean_path = trial_dir / "B_true.png"
            write_image_16bit(true_a_clean_path, np.clip(true_a_clean.raw_matched, 0.0, 1.0))
            write_image_16bit(true_b_clean_path, np.clip(true_b_clean.raw_matched, 0.0, 1.0))

            true_a_noisy = prepared_noisy[idx_a].raw_matched
            true_b_noisy = prepared_noisy[idx_b].raw_matched
            true_a_noisy_path = trial_dir / "A_true_noisy.png"
            true_b_noisy_path = trial_dir / "B_true_noisy.png"
            write_image_16bit(true_a_noisy_path, np.clip(true_a_noisy, 0.0, 1.0))
            write_image_16bit(true_b_noisy_path, np.clip(true_b_noisy, 0.0, 1.0))

            pred_a_clean = prepared_clean[best_idx_a]
            pred_b_clean = prepared_clean[best_idx_b]
            pred_a_noisy = prepared_noisy[best_idx_a].raw_matched
            pred_b_noisy = prepared_noisy[best_idx_b].raw_matched
            pred_a_path = trial_dir / "A_pred.png"
            pred_b_path = trial_dir / "B_pred.png"
            pred_a_noisy_path = trial_dir / "A_pred_noisy.png"
            pred_b_noisy_path = trial_dir / "B_pred_noisy.png"
            write_image_16bit(pred_a_path, np.clip(pred_a_clean.raw_matched, 0.0, 1.0))
            write_image_16bit(pred_b_path, np.clip(pred_b_clean.raw_matched, 0.0, 1.0))
            write_image_16bit(pred_a_noisy_path, np.clip(pred_a_noisy, 0.0, 1.0))
            write_image_16bit(pred_b_noisy_path, np.clip(pred_b_noisy, 0.0, 1.0))

            c_true_path = trial_dir / "C_true.png"
            c_hat_path = trial_dir / "C_hat.png"
            write_image_16bit(c_true_path, np.clip(mixed_true, 0.0, 1.0))
            write_image_16bit(c_hat_path, np.clip(reconstruction_raw, 0.0, 1.0))

            qual_path = monitoring_dir / f"{trial_stub}.png"
            make_qual_grid(
                mixed_true,
                true_a_clean.raw_matched,
                true_b_clean.raw_matched,
                pred_a_clean.raw_matched,
                pred_b_clean.raw_matched,
                reconstruction_raw,
                qual_path,
            )

            curve_csv_path: Optional[Path] = None
            if bool(output_cfg.get("save_curves", True)):
                curve_csv_path = curves_dir / f"{trial_stub}_score_curve.csv"
                write_score_curve_csv(curve_csv_path, best_pair_eval.score_curves)

            true_ids = {true_a_clean.record.pattern_id, true_b_clean.record.pattern_id}
            pred_ids = {pred_a_clean.record.pattern_id, pred_b_clean.record.pattern_id}
            pair_match = true_ids == pred_ids

            trial_payload: Dict[str, Any] = {
                "mode": "candidate_pool_synthetic",
                "trial_id": trial_stub,
                "x_true": x_true,
                "x_hat_primary": float(primary_fraction),
                "x_signed_error": float(primary_fraction - x_true),
                "x_abs_error": float(abs(primary_fraction - x_true)),
                "pair_match": pair_match,
                "primary_metric": primary_metric,
                "best_by_metric": {
                    metric_name: {
                        "x_hat": float(metric_result.fraction),
                        "score": float(metric_result.score),
                        "top_margin": metric_result.top_margin,
                    }
                    for metric_name, metric_result in best_pair_eval.best_by_metric.items()
                },
                "metrics_at_primary": {name: float(metrics_at_primary[name]) for name in metric_names},
                "objective_results": objective_results,
                "true_pair": {
                    "a_id": true_a_clean.record.pattern_id,
                    "a_path": str(true_a_clean.record.path),
                    "b_id": true_b_clean.record.pattern_id,
                    "b_path": str(true_b_clean.record.path),
                },
                "predicted_pair": {
                    "a_id": pred_a_clean.record.pattern_id,
                    "a_path": str(pred_a_clean.record.path),
                    "b_id": pred_b_clean.record.pattern_id,
                    "b_path": str(pred_b_clean.record.path),
                },
                "noise": {
                    "enabled": bool(noise_settings.get("enabled", False)),
                    "gaussian_std": float(noise_settings.get("gaussian_std", 0.0)),
                    "rotation_deg_max": float(noise_settings.get("rotation_deg_max", 0.0)),
                    "seed": noise_settings.get("seed"),
                    "true_a_rotation_deg": noise_meta.get(true_a_clean.record.pattern_id, {}).get("rotation_deg", 0.0),
                    "true_b_rotation_deg": noise_meta.get(true_b_clean.record.pattern_id, {}).get("rotation_deg", 0.0),
                    "pred_a_rotation_deg": noise_meta.get(pred_a_clean.record.pattern_id, {}).get("rotation_deg", 0.0),
                    "pred_b_rotation_deg": noise_meta.get(pred_b_clean.record.pattern_id, {}).get("rotation_deg", 0.0),
                },
                "artifacts": {
                    "a_true": safe_relpath(true_a_clean_path, output_dir),
                    "b_true": safe_relpath(true_b_clean_path, output_dir),
                    "a_true_noisy": safe_relpath(true_a_noisy_path, output_dir),
                    "b_true_noisy": safe_relpath(true_b_noisy_path, output_dir),
                    "a_pred": safe_relpath(pred_a_path, output_dir),
                    "b_pred": safe_relpath(pred_b_path, output_dir),
                    "a_pred_noisy": safe_relpath(pred_a_noisy_path, output_dir),
                    "b_pred_noisy": safe_relpath(pred_b_noisy_path, output_dir),
                    "c_true": safe_relpath(c_true_path, output_dir),
                    "c_hat": safe_relpath(c_hat_path, output_dir),
                    "qual_panel": safe_relpath(qual_path, output_dir),
                    "score_curve_csv": safe_relpath(curve_csv_path, output_dir) if curve_csv_path else None,
                },
            }
            trial_results.append(trial_payload)
            append_jsonl(results_jsonl, trial_payload)

            report_payload["progress"] = {
                "processed": trial_idx + 1,
                "total": pairs_requested,
                "percent": float((trial_idx + 1) / max(pairs_requested, 1)),
            }
            report_payload["latest_trial"] = trial_payload
            report_payload["timestamp"] = datetime.now().isoformat(timespec="seconds")
            update_progress_report(output_dir, report_payload)

        summary_csv_path = output_dir / "summary_metrics.csv"
        write_candidate_pool_summary_csv(summary_csv_path, trial_results, metric_names)

        html_path: Optional[Path] = None
        if bool(output_cfg.get("write_html_report", True)):
            html_path = output_dir / "report" / "index.html"
            write_candidate_pool_html_report(
                html_path,
                output_dir=output_dir,
                trial_results=trial_results,
                metric_names=metric_names,
            )

        report_payload["status"] = "completed"
        report_payload["summary"] = {
            "processed": len(trial_results),
            "candidate_pairs": len(pair_indices),
            "runtime_s": time.perf_counter() - start_time,
        }
        report_payload["artifacts"].update(
            {
                "results_jsonl": safe_relpath(results_jsonl, output_dir),
                "summary_metrics_csv": safe_relpath(summary_csv_path, output_dir),
                "html_report": safe_relpath(html_path, output_dir) if html_path else None,
            }
        )
        report_payload["timestamp"] = datetime.now().isoformat(timespec="seconds")
        update_progress_report(output_dir, report_payload)

        return {
            "mode": "candidate_pool_synthetic",
            "processed": len(trial_results),
            "total_mixed": len(trial_results),
            "candidate_pairs": len(pair_indices),
            "results_jsonl": str(results_jsonl),
            "summary_metrics_csv": str(summary_csv_path),
            "candidate_pool_csv": str(candidate_manifest_path),
            "html_report": str(html_path) if html_path else None,
        }

    if data_mode not in {"mixed_dir"}:
        raise ValueError("data.mode must be one of {'mixed_dir', 'synthetic_pair', 'candidate_pool_synthetic'}.")

    mixed_dir = Path(str(data_cfg.get("mixed_dir", "data/raw/Double Pattern Data/50-50 Double Pattern")))
    mixed_recursive = bool(data_cfg.get("mixed_recursive", False))

    mixed_records = load_mixed_patterns(
        mixed_dir=mixed_dir,
        recursive=mixed_recursive,
        sample_limit=int(sample_limit) if sample_limit is not None else None,
        logger=logger,
    )
    if not mixed_records:
        raise ValueError("No mixed patterns found for deterministic inversion.")

    candidates_a, candidates_b = load_candidate_pools(candidate_cfg, logger)
    pair_indices = build_pair_indices(candidates_a, candidates_b, int(max_pairs) if max_pairs else None)
    if not pair_indices:
        raise ValueError("No A/B candidate pairs available to evaluate.")

    mixed_summary = summarize_images(candidate_paths(mixed_records), sample_n=min(20, len(mixed_records)))
    a_summary = summarize_images(candidate_paths(candidates_a), sample_n=min(20, len(candidates_a)))
    b_summary = summarize_images(candidate_paths(candidates_b), sample_n=min(20, len(candidates_b)))
    logger.info(
        "Pre-flight: mixed=%d A=%d B=%d pairs=%d | mixed size=%s->%s",
        len(mixed_records),
        len(candidates_a),
        len(candidates_b),
        len(pair_indices),
        mixed_summary.get("min_size"),
        mixed_summary.get("max_size"),
    )
    logger.info("A pool dtypes=%s | B pool dtypes=%s", a_summary.get("sample_dtypes"), b_summary.get("sample_dtypes"))
    logger.info(
        "Deterministic settings: metrics=%s primary=%s search=%s alignment=%s",
        metric_names,
        primary_metric,
        search_settings.strategy,
        alignment_settings.enabled,
    )

    results_jsonl = output_dir / "results.jsonl"
    if results_jsonl.exists():
        results_jsonl.unlink()

    report_payload: Dict[str, Any] = {
        "run_id": output_dir.name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "status": "running",
        "progress": {"processed": 0, "total": len(mixed_records), "percent": 0.0},
        "primary_metric": primary_metric,
        "metrics_enabled": metric_names,
        "latest_sample": None,
        "artifacts": {},
    }
    update_progress_report(output_dir, report_payload)

    sample_results: List[Dict[str, Any]] = []
    progress = ProgressLogger(total=len(mixed_records), logger=logger, every=max(1, len(mixed_records) // 5), unit="sample")
    start_time = time.perf_counter()

    for sample_index, mixed_record in enumerate(mixed_records):
        sample_start = time.perf_counter()
        mixed_image = mixed_record.image
        target_shape = mixed_image.shape
        mask = build_centered_mask(target_shape) if preprocess_settings.mask_enabled else None
        mixed_processed, _ = preprocess_pattern(mixed_image, preprocess_settings, mask)

        prepared_a: List[PreparedPattern] = []
        prepared_b: List[PreparedPattern] = []
        for candidate in candidates_a:
            try:
                prepared_a.append(_prepare_pattern_for_sample(candidate, target_shape, preprocess_settings, mask))
            except ValueError as exc:
                logger.debug("Skipping A candidate %s for sample %s: %s", candidate.path, mixed_record.path, exc)
        for candidate in candidates_b:
            try:
                prepared_b.append(_prepare_pattern_for_sample(candidate, target_shape, preprocess_settings, mask))
            except ValueError as exc:
                logger.debug("Skipping B candidate %s for sample %s: %s", candidate.path, mixed_record.path, exc)
        if not prepared_a or not prepared_b:
            raise ValueError(
                f"No shape-compatible candidates for sample {mixed_record.path}. "
                "Enable auto_crop_to_target or use shape-matched inputs."
            )

        best_pair_a: Optional[PreparedPattern] = None
        best_pair_b: Optional[PreparedPattern] = None
        best_pair_eval: Optional[PairEvaluation] = None
        best_primary_score: Optional[float] = None
        best_secondary_l2: Optional[float] = None

        for idx_a, idx_b in pair_indices:
            if idx_a >= len(prepared_a) or idx_b >= len(prepared_b):
                continue
            pattern_a = prepared_a[idx_a]
            pattern_b = prepared_b[idx_b]
            pair_eval = _evaluate_pair(
                pattern_a=pattern_a,
                pattern_b=pattern_b,
                mixed_processed=mixed_processed,
                mask=mask,
                metric_names=metric_names,
                metric_specs=metric_specs,
                primary_metric=primary_metric,
                search_settings=search_settings,
                alignment_settings=alignment_settings,
            )
            primary_score = pair_eval.best_by_metric[primary_metric].score
            primary_objective = metric_specs[primary_metric].objective
            primary_is_better = is_better(primary_score, best_primary_score, primary_objective)

            candidate_l2 = pair_eval.best_by_metric.get("l2")
            candidate_l2_score = float(candidate_l2.score) if candidate_l2 is not None else None
            tie_breaker = False
            if (
                not primary_is_better
                and best_primary_score is not None
                and abs(primary_score - best_primary_score) <= 1e-9
                and candidate_l2_score is not None
                and best_secondary_l2 is not None
            ):
                tie_breaker = candidate_l2_score < best_secondary_l2

            if primary_is_better or tie_breaker or best_pair_eval is None:
                best_pair_a = pattern_a
                best_pair_b = pattern_b
                best_pair_eval = pair_eval
                best_primary_score = primary_score
                if candidate_l2_score is not None:
                    best_secondary_l2 = candidate_l2_score

        if best_pair_a is None or best_pair_b is None or best_pair_eval is None:
            raise RuntimeError(f"No valid candidate pair produced a score for sample {mixed_record.path}.")

        sample_result = _build_sample_result(
            sample_record=mixed_record,
            best_pair_a=best_pair_a,
            best_pair_b=best_pair_b,
            pair_evaluation=best_pair_eval,
            primary_metric=primary_metric,
            mask=mask,
        )
        artifact_paths = _save_sample_artifacts(
            output_dir=output_dir,
            sample_record=mixed_record,
            mixed_matched=mixed_image,
            best_pair_a=best_pair_a,
            best_pair_b=best_pair_b,
            pair_evaluation=best_pair_eval,
            primary_metric=primary_metric,
            mask=mask,
            write_curves=bool(output_cfg.get("save_curves", True)),
        )
        sample_result["artifacts"] = artifact_paths
        sample_result["timing_s"] = time.perf_counter() - sample_start
        sample_results.append(sample_result)
        append_jsonl(results_jsonl, sample_result)

        snapshot = progress.update(1)
        report_payload["progress"] = {
            "processed": snapshot.processed,
            "total": snapshot.total,
            "percent": snapshot.percent,
            "eta_s": snapshot.eta_s,
        }
        report_payload["latest_sample"] = sample_result
        report_payload["timestamp"] = datetime.now().isoformat(timespec="seconds")
        update_progress_report(output_dir, report_payload)
        logger.info(
            "Sample %d/%d complete: %s | best pair=(%s,%s) | %s x_hat=%.4f score=%.6f",
            sample_index + 1,
            len(mixed_records),
            mixed_record.path.name,
            best_pair_a.record.pattern_id,
            best_pair_b.record.pattern_id,
            primary_metric,
            sample_result["best_by_metric"][primary_metric]["x_hat"],
            sample_result["best_by_metric"][primary_metric]["score"],
        )

    summary_csv_path = output_dir / "summary_metrics.csv"
    write_metric_summary_csv(summary_csv_path, sample_results, metric_names)
    report_payload.setdefault("artifacts", {})["summary_metrics_csv"] = safe_relpath(summary_csv_path, output_dir)

    histogram_path = output_dir / "monitoring" / "primary_fraction_hist.png"
    if plot_primary_fraction_histogram(histogram_path, sample_results, primary_metric):
        report_payload.setdefault("artifacts", {})["primary_fraction_hist"] = safe_relpath(histogram_path, output_dir)

    if bool(output_cfg.get("write_html_report", True)):
        html_path = output_dir / "report" / "index.html"
        write_html_report(html_path, sample_results, primary_metric)
        report_payload.setdefault("artifacts", {})["html_report"] = safe_relpath(html_path, output_dir)

    runtime_seconds = time.perf_counter() - start_time
    report_payload["status"] = "completed"
    report_payload["summary"] = {
        "processed": len(sample_results),
        "total_mixed": len(mixed_records),
        "candidate_pairs": len(pair_indices),
        "runtime_s": runtime_seconds,
    }
    report_payload["timestamp"] = datetime.now().isoformat(timespec="seconds")
    update_progress_report(output_dir, report_payload)

    return {
        "processed": len(sample_results),
        "total_mixed": len(mixed_records),
        "candidate_pairs": len(pair_indices),
        "runtime_s": runtime_seconds,
        "results_jsonl": str(results_jsonl),
        "summary_metrics_csv": str(summary_csv_path),
    }
