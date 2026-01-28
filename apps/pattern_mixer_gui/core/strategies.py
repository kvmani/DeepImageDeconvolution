"""Mixing strategy registry for the pattern mixer GUI."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np

from .config import NormalizationMode
from .processing import ProcessedImages, process_images


@dataclass(frozen=True)
class MixingStrategy:
    """Mixing strategy definition."""

    key: str
    label: str
    description: str
    normalizes_inputs: bool
    normalizes_output: bool

    def apply(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
        mask_a: np.ndarray | None,
        mask_b: np.ndarray | None,
        weight_a: float,
        normalization_mode: NormalizationMode,
        noise_a,
        noise_b,
        seed: int | None,
    ) -> ProcessedImages:
        return process_images(
            image_a=image_a,
            image_b=image_b,
            mask_a=mask_a,
            mask_b=mask_b,
            weight_a=weight_a,
            normalization_mode=normalization_mode,
            normalize_output=self.normalizes_output,
            noise_a=noise_a,
            noise_b=noise_b,
            seed=seed,
            normalize_inputs=self.normalizes_inputs,
        )


def build_strategy_registry() -> Dict[str, MixingStrategy]:
    """Build the mixing strategy registry."""
    strategies = [
        MixingStrategy(
            key="norm_then_mix",
            label="Normalize A/B then mix",
            description="Normalize A and B before mixing. Optional normalize C.",
            normalizes_inputs=True,
            normalizes_output=False,
        ),
        MixingStrategy(
            key="norm_then_mix_then_norm",
            label="Normalize A/B then mix then normalize C",
            description="Normalize A and B, mix, then normalize C.",
            normalizes_inputs=True,
            normalizes_output=True,
        ),
        MixingStrategy(
            key="mix_then_norm",
            label="Mix raw then normalize C",
            description="Mix raw intensities, then normalize the result.",
            normalizes_inputs=False,
            normalizes_output=True,
        ),
        MixingStrategy(
            key="no_normalization",
            label="No normalization (demo)",
            description="Mix without normalization (demo only).",
            normalizes_inputs=False,
            normalizes_output=False,
        ),
    ]
    return {strategy.key: strategy for strategy in strategies}


def strategy_labels(registry: Dict[str, MixingStrategy]) -> List[str]:
    """Return strategy labels in registry order."""
    return [strategy.label for strategy in registry.values()]
