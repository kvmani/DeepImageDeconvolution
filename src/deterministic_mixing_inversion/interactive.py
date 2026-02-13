"""Interactive deterministic pair-identification pipeline for CLI/GUI use."""
from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Callable, Dict, List, Optional, Sequence

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
    build_unique_pair_indices,
    load_pattern_records_from_paths,
)
from src.deterministic_mixing_inversion.metrics import (
    MetricSpec,
    compute_metric_values,
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
from src.deterministic_mixing_inversion.search import (
    MetricBest,
    SearchSettings,
    parse_search_settings,
    search_mixing_fraction,
)
from src.utils.io import collect_image_paths


@dataclass(frozen=True)
class SyntheticNoiseConfig:
    """Synthetic noise settings applied to A and B before mixing."""

    gaussian_enabled: bool = True
    gaussian_std: float = 0.01
    salt_pepper_enabled: bool = False
    salt_pepper_amount: float = 0.01
    salt_vs_pepper: float = 0.5
    rotation_enabled: bool = True
    rotation_max_deg: float = 2.0


@dataclass(frozen=True)
class SyntheticCase:
    """Synthetic mixed-pattern case."""

    candidate_a: PatternRecord
    candidate_b: PatternRecord
    mix_fraction_true: float
    pattern_a_noisy: np.ndarray
    pattern_b_noisy: np.ndarray
    mixed_c: np.ndarray
    angle_a_deg: float
    angle_b_deg: float


@dataclass(frozen=True)
class RankedPair:
    """One ranked candidate pair entry."""

    rank: int
    index_a: int
    index_b: int
    id_a: str
    id_b: str
    x_hat: float
    primary_score: float
    l2_score: Optional[float]
    metric_scores: Dict[str, float]
    top_margin: Optional[float]
    alignment: Dict[str, float]


@dataclass(frozen=True)
class IdentificationResult:
    """Pair-identification output."""

    winner: RankedPair
    top_k: List[RankedPair]
    total_pairs: int
    runtime_s: float
    primary_metric: str


@dataclass(frozen=True)
class PreparedCandidate:
    """Prepared candidate cache entry."""

    record: PatternRecord
    original_index: int
    raw_matched: np.ndarray
    processed: np.ndarray


@dataclass(frozen=True)
class PairEvaluation:
    """One evaluated pair."""

    best_by_metric: Dict[str, MetricBest]
    alignment: RigidAlignment
    evaluated_points: int


@dataclass(frozen=True)
class PairSearchSettings:
    """Pair search settings for two-stage candidate filtering."""

    two_stage_enabled: bool
    coarse_top_m: int
    coarse_metric: str
    coarse_search: SearchSettings
    coarse_alignment: AlignmentSettings


ProgressCallback = Callable[[int, int, float, str], None]


def sample_random_candidates(
    candidate_dir: Path,
    sample_count: int,
    seed: int | None,
    recursive: bool = False,
    logger: Optional[logging.Logger] = None,
) -> List[PatternRecord]:
    """Load a random subset of candidate patterns from one directory."""
    if sample_count <= 0:
        raise ValueError("sample_count must be a positive integer.")
    all_paths = sorted(collect_image_paths(candidate_dir, recursive=recursive))
    if not all_paths:
        raise ValueError(f"No candidate patterns found under: {candidate_dir}")
    if len(all_paths) <= sample_count:
        selected_paths = all_paths
    else:
        rng = np.random.default_rng(seed)
        sampled_indices = np.sort(rng.choice(len(all_paths), size=sample_count, replace=False))
        selected_paths = [all_paths[int(index)] for index in sampled_indices]
    records = load_pattern_records_from_paths(selected_paths)
    if logger is not None:
        logger.info(
            "Loaded random candidates: selected=%d total=%d seed=%s from %s",
            len(records),
            len(all_paths),
            seed,
            candidate_dir,
        )
    return records


def _apply_gaussian_noise(image: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    if std <= 0.0:
        return image
    noise = rng.normal(loc=0.0, scale=std, size=image.shape).astype(np.float32)
    return (image + noise).astype(np.float32)


def _apply_salt_pepper_noise(
    image: np.ndarray,
    amount: float,
    salt_vs_pepper: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if amount <= 0.0:
        return image
    total_pixels = image.size
    noisy_count = int(round(amount * total_pixels))
    if noisy_count <= 0:
        return image
    noisy_count = min(noisy_count, total_pixels)
    flat = image.reshape(-1).copy()
    indices = rng.choice(total_pixels, size=noisy_count, replace=False)
    salt_count = int(round(noisy_count * np.clip(salt_vs_pepper, 0.0, 1.0)))
    salt_indices = indices[:salt_count]
    pepper_indices = indices[salt_count:]
    flat[salt_indices] = 1.0
    flat[pepper_indices] = 0.0
    return flat.reshape(image.shape).astype(np.float32)


def _apply_rotation_noise(
    image: np.ndarray,
    max_deg: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    bounded_deg = float(np.clip(abs(max_deg), 0.0, 2.0))
    if bounded_deg <= 0.0:
        return image.astype(np.float32), 0.0
    angle = float(rng.uniform(-bounded_deg, bounded_deg))
    rotated = rotate(
        image.astype(np.float32),
        angle=angle,
        reshape=False,
        order=3,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).astype(np.float32)
    return rotated, angle


def _mix_patterns(image_a: np.ndarray, image_b: np.ndarray, mix_fraction: float) -> np.ndarray:
    return (mix_fraction * image_a + (1.0 - mix_fraction) * image_b).astype(np.float32)


def build_synthetic_case(
    candidates: Sequence[PatternRecord],
    index_a: int,
    index_b: int,
    mix_fraction: float,
    noise: SyntheticNoiseConfig,
    seed: int | None,
    mask_enabled: bool = True,
) -> SyntheticCase:
    """Build one synthetic mixed case from selected candidate indices."""
    if index_a == index_b:
        raise ValueError("A and B must be different candidates.")
    if index_a < 0 or index_a >= len(candidates):
        raise IndexError("index_a is out of range.")
    if index_b < 0 or index_b >= len(candidates):
        raise IndexError("index_b is out of range.")

    candidate_a = candidates[index_a]
    candidate_b = candidates[index_b]
    target_shape = (
        min(candidate_a.image.shape[0], candidate_b.image.shape[0]),
        min(candidate_a.image.shape[1], candidate_b.image.shape[1]),
    )
    image_a = match_shape_to_target(candidate_a.image, target_shape, auto_crop_to_target=True)
    image_b = match_shape_to_target(candidate_b.image, target_shape, auto_crop_to_target=True)
    mask = build_centered_mask(target_shape) if mask_enabled else None
    if mask is not None:
        image_a = image_a.copy()
        image_b = image_b.copy()
        image_a[~mask] = 0.0
        image_b[~mask] = 0.0

    rng = np.random.default_rng(seed)
    angle_a = 0.0
    angle_b = 0.0
    noisy_a = image_a.astype(np.float32, copy=True)
    noisy_b = image_b.astype(np.float32, copy=True)

    if noise.rotation_enabled:
        noisy_a, angle_a = _apply_rotation_noise(noisy_a, noise.rotation_max_deg, rng)
        noisy_b, angle_b = _apply_rotation_noise(noisy_b, noise.rotation_max_deg, rng)
    if noise.gaussian_enabled:
        noisy_a = _apply_gaussian_noise(noisy_a, noise.gaussian_std, rng)
        noisy_b = _apply_gaussian_noise(noisy_b, noise.gaussian_std, rng)
    if noise.salt_pepper_enabled:
        noisy_a = _apply_salt_pepper_noise(noisy_a, noise.salt_pepper_amount, noise.salt_vs_pepper, rng)
        noisy_b = _apply_salt_pepper_noise(noisy_b, noise.salt_pepper_amount, noise.salt_vs_pepper, rng)

    noisy_a = np.clip(noisy_a, 0.0, 1.0)
    noisy_b = np.clip(noisy_b, 0.0, 1.0)
    if mask is not None:
        noisy_a[~mask] = 0.0
        noisy_b[~mask] = 0.0

    bounded_fraction = float(np.clip(mix_fraction, 0.0, 1.0))
    mixed_c = _mix_patterns(noisy_a, noisy_b, bounded_fraction)
    if mask is not None:
        mixed_c[~mask] = 0.0

    return SyntheticCase(
        candidate_a=candidate_a,
        candidate_b=candidate_b,
        mix_fraction_true=bounded_fraction,
        pattern_a_noisy=noisy_a.astype(np.float32),
        pattern_b_noisy=noisy_b.astype(np.float32),
        mixed_c=mixed_c.astype(np.float32),
        angle_a_deg=angle_a,
        angle_b_deg=angle_b,
    )


def _prepare_candidates(
    candidates: Sequence[PatternRecord],
    target_shape: tuple[int, int],
    preprocess_settings: PreprocessSettings,
    mask: np.ndarray | None,
) -> List[PreparedCandidate]:
    prepared: List[PreparedCandidate] = []
    for original_index, candidate in enumerate(candidates):
        try:
            matched = match_shape_to_target(
                candidate.image,
                target_shape=target_shape,
                auto_crop_to_target=preprocess_settings.auto_crop_to_target,
            )
        except ValueError:
            continue
        processed, _ = preprocess_pattern(matched, preprocess_settings, mask)
        prepared.append(
            PreparedCandidate(
                record=candidate,
                original_index=original_index,
                raw_matched=matched,
                processed=processed,
            )
        )
    return prepared


def _build_ranked_pairs(
    ranking_rows: Sequence[RankedPair],
    primary_objective: str,
) -> List[RankedPair]:
    if primary_objective == "max":
        sorted_rows = sorted(
            ranking_rows,
            key=lambda row: (row.primary_score, -(row.l2_score if row.l2_score is not None else 1e12)),
            reverse=True,
        )
    else:
        sorted_rows = sorted(
            ranking_rows,
            key=lambda row: (row.primary_score, row.l2_score if row.l2_score is not None else 1e12),
        )
    ranked_rows: List[RankedPair] = []
    for rank_index, row in enumerate(sorted_rows, start=1):
        ranked_rows.append(
            RankedPair(
                rank=rank_index,
                index_a=row.index_a,
                index_b=row.index_b,
                id_a=row.id_a,
                id_b=row.id_b,
                x_hat=row.x_hat,
                primary_score=row.primary_score,
                l2_score=row.l2_score,
                metric_scores=row.metric_scores,
                top_margin=row.top_margin,
                alignment=row.alignment,
            )
        )
    return ranked_rows


def _parse_pair_search_settings(
    pair_search_cfg: Dict[str, object],
    primary_metric: str,
    enabled_metrics: Sequence[str],
    default_search: SearchSettings,
    default_alignment: AlignmentSettings,
) -> PairSearchSettings:
    if not isinstance(pair_search_cfg, dict):
        pair_search_cfg = {}

    two_stage_enabled = bool(pair_search_cfg.get("two_stage_enabled", False))
    coarse_top_m_raw = int(pair_search_cfg.get("coarse_top_m", 20))
    coarse_top_m = max(2, coarse_top_m_raw)

    coarse_metric_raw = str(pair_search_cfg.get("coarse_metric", primary_metric)).lower()
    coarse_metric = coarse_metric_raw if coarse_metric_raw in enabled_metrics else primary_metric

    coarse_search_cfg = pair_search_cfg.get("coarse_search", {})
    if isinstance(coarse_search_cfg, dict) and coarse_search_cfg:
        coarse_search = parse_search_settings(coarse_search_cfg)
    else:
        coarse_search = SearchSettings(
            strategy="coarse_to_fine",
            exhaustive_step=max(default_search.exhaustive_step, 0.02),
            coarse_steps=(0.1, 0.02),
            refine_window_steps=2.0,
        )

    coarse_alignment_cfg = pair_search_cfg.get("coarse_alignment", {})
    if isinstance(coarse_alignment_cfg, dict) and coarse_alignment_cfg:
        coarse_alignment = parse_alignment_settings(coarse_alignment_cfg)
    else:
        coarse_alignment = AlignmentSettings(
            enabled=False,
            translation_enabled=default_alignment.translation_enabled,
            max_shift_px=default_alignment.max_shift_px,
            rotation_enabled=default_alignment.rotation_enabled,
            search_range_deg=default_alignment.search_range_deg,
            hard_max_deg=default_alignment.hard_max_deg,
            rotation_step_deg=default_alignment.rotation_step_deg,
            interpolation_order=default_alignment.interpolation_order,
        )

    return PairSearchSettings(
        two_stage_enabled=two_stage_enabled,
        coarse_top_m=coarse_top_m,
        coarse_metric=coarse_metric,
        coarse_search=coarse_search,
        coarse_alignment=coarse_alignment,
    )


def _evaluate_pair(
    pattern_a: PreparedCandidate,
    pattern_b: PreparedCandidate,
    mixed_processed: np.ndarray,
    mask: np.ndarray | None,
    metric_names: Sequence[str],
    metric_specs: Dict[str, MetricSpec],
    primary_metric: str,
    search_settings: SearchSettings,
    alignment_settings: AlignmentSettings,
) -> PairEvaluation:
    def evaluate_unaligned(fraction: float) -> Dict[str, float]:
        reconstructed = _mix_patterns(pattern_a.processed, pattern_b.processed, fraction)
        return compute_metric_values(reconstructed, mixed_processed, mask, metric_names)

    initial_search = search_mixing_fraction(
        evaluate_fraction=evaluate_unaligned,
        metric_specs=metric_specs,
        primary_metric=primary_metric,
        settings=search_settings,
    )
    initial_fraction = initial_search.metric_best[primary_metric].fraction
    initial_reconstruction = _mix_patterns(pattern_a.processed, pattern_b.processed, initial_fraction)

    if alignment_settings.enabled:
        best_alignment = estimate_best_alignment(
            moving=initial_reconstruction,
            target=mixed_processed,
            mask=mask,
            settings=alignment_settings,
            score_function=masked_ncc,
        )
    else:
        best_alignment = RigidAlignment(angle_deg=0.0, shift_y=0.0, shift_x=0.0, score=0.0)

    def evaluate_aligned(fraction: float) -> Dict[str, float]:
        reconstructed = _mix_patterns(pattern_a.processed, pattern_b.processed, fraction)
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
        alignment=best_alignment,
        evaluated_points=final_search.evaluated_points,
    )


def identify_pair_from_candidates(
    candidates: Sequence[PatternRecord],
    mixed_image: np.ndarray,
    inversion_cfg: Dict[str, object],
    top_k: int = 5,
    progress_callback: Optional[ProgressCallback] = None,
    logger: Optional[logging.Logger] = None,
) -> IdentificationResult:
    """Identify the most likely candidate pair that generated the mixed image."""
    if len(candidates) < 2:
        raise ValueError("At least two candidates are required.")
    if mixed_image.ndim != 2:
        raise ValueError("mixed_image must be a 2D grayscale array.")
    if top_k <= 0:
        raise ValueError("top_k must be > 0.")

    preprocess_settings = parse_preprocess_settings(inversion_cfg.get("preprocess", {}))
    metric_names, primary_metric, metric_specs = parse_metric_config(inversion_cfg.get("metrics", {}))
    search_settings = parse_search_settings(inversion_cfg.get("search", {}))
    alignment_settings = parse_alignment_settings(inversion_cfg.get("alignment", {}))
    pair_search_settings = _parse_pair_search_settings(
        pair_search_cfg=inversion_cfg.get("pair_search", {}) if isinstance(inversion_cfg, dict) else {},
        primary_metric=primary_metric,
        enabled_metrics=metric_names,
        default_search=search_settings,
        default_alignment=alignment_settings,
    )

    target_shape = mixed_image.shape
    mask = build_centered_mask(target_shape) if preprocess_settings.mask_enabled else None
    mixed_processed, _ = preprocess_pattern(mixed_image.astype(np.float32), preprocess_settings, mask)
    prepared_candidates = _prepare_candidates(candidates, target_shape, preprocess_settings, mask)
    if len(prepared_candidates) < 2:
        raise ValueError("Not enough shape-compatible candidates after preprocessing.")

    pair_indices = build_unique_pair_indices(prepared_candidates, max_pairs=None)
    total_pairs = len(pair_indices)
    if total_pairs == 0:
        raise ValueError("No unique candidate pairs available.")

    should_two_stage = pair_search_settings.two_stage_enabled and total_pairs > pair_search_settings.coarse_top_m
    if should_two_stage:
        stage1_total = total_pairs
        stage2_total = min(pair_search_settings.coarse_top_m, total_pairs)
        progress_total = stage1_total + stage2_total
    else:
        stage1_total = total_pairs
        stage2_total = 0
        progress_total = total_pairs

    start = time.perf_counter()

    def emit_progress(processed: int, message: str) -> None:
        if progress_callback is None:
            return
        elapsed = time.perf_counter() - start
        avg = elapsed / max(processed, 1)
        eta = max(progress_total - processed, 0) * avg
        progress_callback(processed, progress_total, eta, message)

    stage1_metric_names = [pair_search_settings.coarse_metric]
    if "l2" in metric_specs and "l2" not in stage1_metric_names:
        stage1_metric_names.append("l2")
    stage1_metric_specs = {name: metric_specs[name] for name in stage1_metric_names}

    stage1_rows: List[RankedPair] = []
    for pair_count, (idx_a, idx_b) in enumerate(pair_indices, start=1):
        pair_start = time.perf_counter()
        prepared_a = prepared_candidates[idx_a]
        prepared_b = prepared_candidates[idx_b]
        evaluation = _evaluate_pair(
            pattern_a=prepared_a,
            pattern_b=prepared_b,
            mixed_processed=mixed_processed,
            mask=mask,
            metric_names=stage1_metric_names if should_two_stage else metric_names,
            metric_specs=stage1_metric_specs if should_two_stage else metric_specs,
            primary_metric=pair_search_settings.coarse_metric if should_two_stage else primary_metric,
            search_settings=pair_search_settings.coarse_search if should_two_stage else search_settings,
            alignment_settings=pair_search_settings.coarse_alignment if should_two_stage else alignment_settings,
        )
        metric_scores = {
            metric_name: float(metric_result.score)
            for metric_name, metric_result in evaluation.best_by_metric.items()
        }
        stage1_primary = pair_search_settings.coarse_metric if should_two_stage else primary_metric
        primary_result = evaluation.best_by_metric[stage1_primary]
        l2_result = evaluation.best_by_metric.get("l2")
        l2_score = float(l2_result.score) if l2_result is not None else None

        stage1_rows.append(
            RankedPair(
                rank=0,
                index_a=prepared_a.original_index,
                index_b=prepared_b.original_index,
                id_a=prepared_a.record.pattern_id,
                id_b=prepared_b.record.pattern_id,
                x_hat=float(primary_result.fraction),
                primary_score=float(primary_result.score),
                l2_score=l2_score,
                metric_scores=metric_scores,
                top_margin=primary_result.top_margin,
                alignment={
                    "angle_deg": float(evaluation.alignment.angle_deg),
                    "shift_y": float(evaluation.alignment.shift_y),
                    "shift_x": float(evaluation.alignment.shift_x),
                },
            )
        )
        stage = "Stage1 coarse" if should_two_stage else "Identify"
        emit_progress(
            pair_count,
            (
                f"{stage} {pair_count}/{stage1_total}: "
                f"{prepared_a.record.pattern_id} + {prepared_b.record.pattern_id} "
                f"({time.perf_counter() - pair_start:.2f}s)"
            ),
        )

    if should_two_stage:
        stage1_ranked = _build_ranked_pairs(
            stage1_rows,
            primary_objective=stage1_metric_specs[pair_search_settings.coarse_metric].objective,
        )
        selected_rows = stage1_ranked[:stage2_total]
        selected_pair_keys = {
            tuple(sorted((selected.index_a, selected.index_b))) for selected in selected_rows
        }
        selected_pair_indices: List[tuple[int, int]] = []
        for idx_a, idx_b in build_unique_pair_indices(prepared_candidates, max_pairs=None):
            original_a = prepared_candidates[idx_a].original_index
            original_b = prepared_candidates[idx_b].original_index
            if tuple(sorted((original_a, original_b))) in selected_pair_keys:
                selected_pair_indices.append((idx_a, idx_b))

        ranking_rows: List[RankedPair] = []
        for refined_count, (idx_a, idx_b) in enumerate(selected_pair_indices, start=1):
            pair_start = time.perf_counter()
            prepared_a = prepared_candidates[idx_a]
            prepared_b = prepared_candidates[idx_b]
            evaluation = _evaluate_pair(
                pattern_a=prepared_a,
                pattern_b=prepared_b,
                mixed_processed=mixed_processed,
                mask=mask,
                metric_names=metric_names,
                metric_specs=metric_specs,
                primary_metric=primary_metric,
                search_settings=search_settings,
                alignment_settings=alignment_settings,
            )
            metric_scores = {
                metric_name: float(metric_result.score)
                for metric_name, metric_result in evaluation.best_by_metric.items()
            }
            primary_result = evaluation.best_by_metric[primary_metric]
            l2_result = evaluation.best_by_metric.get("l2")
            l2_score = float(l2_result.score) if l2_result is not None else None
            ranking_rows.append(
                RankedPair(
                    rank=0,
                    index_a=prepared_a.original_index,
                    index_b=prepared_b.original_index,
                    id_a=prepared_a.record.pattern_id,
                    id_b=prepared_b.record.pattern_id,
                    x_hat=float(primary_result.fraction),
                    primary_score=float(primary_result.score),
                    l2_score=l2_score,
                    metric_scores=metric_scores,
                    top_margin=primary_result.top_margin,
                    alignment={
                        "angle_deg": float(evaluation.alignment.angle_deg),
                        "shift_y": float(evaluation.alignment.shift_y),
                        "shift_x": float(evaluation.alignment.shift_x),
                    },
                )
            )
            emit_progress(
                stage1_total + refined_count,
                (
                    f"Stage2 refine {refined_count}/{len(selected_pair_indices)}: "
                    f"{prepared_a.record.pattern_id} + {prepared_b.record.pattern_id} "
                    f"({time.perf_counter() - pair_start:.2f}s)"
                ),
            )
    else:
        ranking_rows = stage1_rows

    ranked_rows = _build_ranked_pairs(
        ranking_rows,
        primary_objective=metric_specs[primary_metric].objective,
    )

    winner = ranked_rows[0]
    if logger is not None:
        logger.info(
            "Identification complete: winner=%s + %s | %s=%.6f | x_hat=%.4f | total_pairs=%d",
            winner.id_a,
            winner.id_b,
            primary_metric,
            winner.primary_score,
            winner.x_hat,
            total_pairs,
        )

    runtime = time.perf_counter() - start
    return IdentificationResult(
        winner=winner,
        top_k=ranked_rows[: min(top_k, len(ranked_rows))],
        total_pairs=total_pairs,
        runtime_s=runtime,
        primary_metric=primary_metric,
    )
