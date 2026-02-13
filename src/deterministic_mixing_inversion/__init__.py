"""Deterministic mixing-fraction inversion package."""
from __future__ import annotations

from src.deterministic_mixing_inversion.benchmark import run_synthetic_robustness_benchmark
from src.deterministic_mixing_inversion.interactive import (
    IdentificationResult,
    SyntheticCase,
    SyntheticNoiseConfig,
    build_synthetic_case,
    identify_pair_from_candidates,
    sample_random_candidates,
)
from src.deterministic_mixing_inversion.runner import run_deterministic_inversion

__all__ = [
    "IdentificationResult",
    "SyntheticCase",
    "SyntheticNoiseConfig",
    "build_synthetic_case",
    "identify_pair_from_candidates",
    "sample_random_candidates",
    "run_deterministic_inversion",
    "run_synthetic_robustness_benchmark",
]
