"""Alignment helpers for deterministic inversion."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
from scipy.ndimage import rotate, shift


@dataclass(frozen=True)
class AlignmentSettings:
    """Rigid-alignment settings."""

    enabled: bool
    translation_enabled: bool
    max_shift_px: float
    rotation_enabled: bool
    search_range_deg: float
    hard_max_deg: float
    rotation_step_deg: float
    interpolation_order: int


@dataclass(frozen=True)
class RigidAlignment:
    """Rigid transform parameters."""

    angle_deg: float
    shift_y: float
    shift_x: float
    score: float


def parse_alignment_settings(alignment_cfg: Dict[str, object]) -> AlignmentSettings:
    """Parse rigid-alignment settings from configuration."""
    translation_cfg = alignment_cfg.get("translation", {}) if isinstance(alignment_cfg, dict) else {}
    rotation_cfg = alignment_cfg.get("rotation", {}) if isinstance(alignment_cfg, dict) else {}

    search_range_deg = float(rotation_cfg.get("search_range_deg", 2.0))
    hard_max_deg = float(rotation_cfg.get("hard_max_deg", 5.0))
    bounded_search_range = min(abs(search_range_deg), abs(hard_max_deg))

    return AlignmentSettings(
        enabled=bool(alignment_cfg.get("enabled", True)),
        translation_enabled=bool(translation_cfg.get("enabled", True)),
        max_shift_px=float(translation_cfg.get("max_shift_px", 15.0)),
        rotation_enabled=bool(rotation_cfg.get("enabled", True)),
        search_range_deg=bounded_search_range,
        hard_max_deg=abs(hard_max_deg),
        rotation_step_deg=float(rotation_cfg.get("step_deg", 0.5)),
        interpolation_order=int(alignment_cfg.get("interpolation_order", 3)),
    )


def apply_rigid_alignment(
    image: np.ndarray,
    alignment: RigidAlignment,
    interpolation_order: int,
    mask: Optional[np.ndarray],
) -> np.ndarray:
    """Apply rotation then translation to an image."""
    transformed = image.astype(np.float32, copy=True)
    if abs(alignment.angle_deg) > 1e-12:
        transformed = rotate(
            transformed,
            angle=alignment.angle_deg,
            reshape=False,
            order=interpolation_order,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
    if abs(alignment.shift_y) > 1e-12 or abs(alignment.shift_x) > 1e-12:
        transformed = shift(
            transformed,
            shift=(alignment.shift_y, alignment.shift_x),
            order=interpolation_order,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
    if mask is not None:
        transformed = transformed.copy()
        transformed[~mask] = 0.0
    return transformed.astype(np.float32)


def estimate_translation_phase_correlation(
    moving: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray],
    max_shift_px: float,
) -> tuple[float, float]:
    """Estimate translation aligning moving image to target image."""
    moving_work = moving.astype(np.float32)
    target_work = target.astype(np.float32)
    if mask is not None:
        moving_work = moving_work * mask.astype(np.float32)
        target_work = target_work * mask.astype(np.float32)

    moving_fft = np.fft.fft2(moving_work)
    target_fft = np.fft.fft2(target_work)
    cross_power = target_fft * np.conj(moving_fft)
    cross_power /= np.maximum(np.abs(cross_power), 1e-12)
    correlation = np.fft.ifft2(cross_power)
    peak_index = np.unravel_index(np.argmax(np.abs(correlation)), correlation.shape)

    raw_shift_y = float(peak_index[0])
    raw_shift_x = float(peak_index[1])
    height, width = moving.shape
    if raw_shift_y > height / 2:
        raw_shift_y -= float(height)
    if raw_shift_x > width / 2:
        raw_shift_x -= float(width)

    bounded_shift_y = float(np.clip(raw_shift_y, -max_shift_px, max_shift_px))
    bounded_shift_x = float(np.clip(raw_shift_x, -max_shift_px, max_shift_px))
    return bounded_shift_y, bounded_shift_x


def estimate_best_alignment(
    moving: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray],
    settings: AlignmentSettings,
    score_function: Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float],
) -> RigidAlignment:
    """Estimate the best rigid alignment for a moving image against target."""
    if not settings.enabled:
        return RigidAlignment(angle_deg=0.0, shift_y=0.0, shift_x=0.0, score=score_function(moving, target, mask))

    angle_candidates = [0.0]
    if settings.rotation_enabled and settings.search_range_deg > 0.0 and settings.rotation_step_deg > 0.0:
        angle_candidates = list(
            np.arange(
                -settings.search_range_deg,
                settings.search_range_deg + (settings.rotation_step_deg * 0.5),
                settings.rotation_step_deg,
                dtype=np.float32,
            )
        )
        if 0.0 not in angle_candidates:
            angle_candidates.append(0.0)
        angle_candidates = sorted(set(float(angle) for angle in angle_candidates))

    best_alignment: Optional[RigidAlignment] = None
    for angle_deg in angle_candidates:
        if abs(angle_deg) > 1e-12:
            rotated = rotate(
                moving,
                angle=angle_deg,
                reshape=False,
                order=settings.interpolation_order,
                mode="constant",
                cval=0.0,
                prefilter=False,
            ).astype(np.float32)
        else:
            rotated = moving.astype(np.float32, copy=True)

        shift_y = 0.0
        shift_x = 0.0
        if settings.translation_enabled and settings.max_shift_px > 0.0:
            shift_y, shift_x = estimate_translation_phase_correlation(
                rotated,
                target,
                mask=mask,
                max_shift_px=settings.max_shift_px,
            )

        aligned = apply_rigid_alignment(
            moving,
            RigidAlignment(angle_deg=angle_deg, shift_y=shift_y, shift_x=shift_x, score=0.0),
            interpolation_order=settings.interpolation_order,
            mask=mask,
        )
        score_value = score_function(aligned, target, mask)

        candidate = RigidAlignment(
            angle_deg=float(angle_deg),
            shift_y=float(shift_y),
            shift_x=float(shift_x),
            score=float(score_value),
        )
        if best_alignment is None or candidate.score > best_alignment.score:
            best_alignment = candidate

    if best_alignment is None:
        return RigidAlignment(angle_deg=0.0, shift_y=0.0, shift_x=0.0, score=0.0)
    return best_alignment

