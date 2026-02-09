"""I/O helpers for deterministic mixing-fraction inversion."""
from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Dict, Iterable, List, Sequence

import numpy as np

from src.utils.io import collect_image_paths, read_image_16bit, to_float01


@dataclass(frozen=True)
class PatternRecord:
    """Container for a loaded pattern image.

    Parameters
    ----------
    pattern_id:
        Stable identifier used in outputs.
    path:
        Source path.
    image:
        Image as float32 in [0, 1].
    source_dtype:
        Dtype observed on disk before canonical conversion.
    """

    pattern_id: str
    path: Path
    image: np.ndarray
    source_dtype: str


def _sanitize_id(path: Path) -> str:
    return path.stem.replace(" ", "_").replace("/", "_").replace("\\", "_")


def load_pattern(path: Path) -> PatternRecord:
    uint16_image = read_image_16bit(path)
    float_image = to_float01(uint16_image).astype(np.float32)
    return PatternRecord(
        pattern_id=_sanitize_id(path),
        path=path,
        image=float_image,
        source_dtype=str(uint16_image.dtype),
    )


def _compile_regex(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, flags=re.IGNORECASE)


def _filter_by_regex(paths: Sequence[Path], pattern: str) -> List[Path]:
    compiled = _compile_regex(pattern)
    return [path for path in paths if compiled.search(path.name)]


def _apply_limit(paths: Sequence[Path], max_items: int | None) -> List[Path]:
    if max_items is None or max_items <= 0:
        return list(paths)
    return list(paths[:max_items])


def load_mixed_patterns(
    mixed_dir: Path,
    recursive: bool,
    sample_limit: int | None,
    logger: logging.Logger,
) -> List[PatternRecord]:
    """Load mixed-pattern inputs.

    Parameters
    ----------
    mixed_dir:
        Directory containing mixed patterns.
    recursive:
        Whether to recurse into subdirectories.
    sample_limit:
        Optional maximum number of samples.
    logger:
        Logger used for pre-flight reporting.

    Returns
    -------
    list of PatternRecord
        Loaded mixed-pattern records.
    """
    mixed_paths = collect_image_paths(mixed_dir, recursive=recursive)
    if sample_limit is not None and sample_limit > 0:
        mixed_paths = mixed_paths[:sample_limit]
    logger.info("Mixed inputs discovered: %d under %s", len(mixed_paths), mixed_dir)
    return [load_pattern(path) for path in mixed_paths]


def load_candidate_pools(
    candidate_cfg: Dict[str, object],
    logger: logging.Logger,
) -> tuple[List[PatternRecord], List[PatternRecord]]:
    """Load candidate A/B pattern pools.

    Parameters
    ----------
    candidate_cfg:
        Candidate pool configuration.
    logger:
        Logger used for pre-flight reporting.

    Returns
    -------
    tuple of list of PatternRecord
        Candidate records for A and B pools.
    """
    recursive = bool(candidate_cfg.get("recursive", False))
    max_per_group = candidate_cfg.get("max_per_group")
    max_value = int(max_per_group) if max_per_group is not None else None

    root_dir_raw = candidate_cfg.get("root_dir")
    if root_dir_raw is None:
        raise ValueError("candidate_pool.root_dir is required.")
    root_dir = Path(str(root_dir_raw))
    if not root_dir.exists():
        raise FileNotFoundError(f"Candidate pool root directory not found: {root_dir}")

    a_dir_raw = candidate_cfg.get("a_dir")
    b_dir_raw = candidate_cfg.get("b_dir")
    a_pattern = str(candidate_cfg.get("a_pattern", "(?i)bcc"))
    b_pattern = str(candidate_cfg.get("b_pattern", "(?i)fcc"))

    if a_dir_raw or b_dir_raw:
        if not a_dir_raw or not b_dir_raw:
            raise ValueError("Both candidate_pool.a_dir and candidate_pool.b_dir must be set together.")
        a_dir = Path(str(a_dir_raw))
        b_dir = Path(str(b_dir_raw))
        a_paths = collect_image_paths(a_dir, recursive=recursive)
        b_paths = collect_image_paths(b_dir, recursive=recursive)
    else:
        all_paths = collect_image_paths(root_dir, recursive=recursive)
        a_paths = _filter_by_regex(all_paths, a_pattern)
        b_paths = _filter_by_regex(all_paths, b_pattern)

    a_paths = _apply_limit(sorted(a_paths), max_value)
    b_paths = _apply_limit(sorted(b_paths), max_value)
    if not a_paths:
        raise ValueError("No A-type candidates found. Check candidate_pool.a_pattern or directories.")
    if not b_paths:
        raise ValueError("No B-type candidates found. Check candidate_pool.b_pattern or directories.")

    logger.info(
        "Candidate pools discovered: A=%d, B=%d (recursive=%s, max_per_group=%s)",
        len(a_paths),
        len(b_paths),
        recursive,
        max_value,
    )
    candidates_a = [load_pattern(path) for path in a_paths]
    candidates_b = [load_pattern(path) for path in b_paths]
    return candidates_a, candidates_b


def load_candidate_pool(
    candidate_cfg: Dict[str, object],
    logger: logging.Logger,
    sample_seed: int | None,
) -> List[PatternRecord]:
    """Load a single candidate pool for synthetic pair discovery.

    Parameters
    ----------
    candidate_cfg:
        Candidate pool configuration.
    logger:
        Logger used for pre-flight reporting.
    sample_seed:
        Optional RNG seed for random sampling.

    Returns
    -------
    list of PatternRecord
        Candidate records.
    """
    recursive = bool(candidate_cfg.get("recursive", False))
    max_candidates = candidate_cfg.get("max_candidates")
    max_value = int(max_candidates) if max_candidates is not None else None

    root_dir_raw = candidate_cfg.get("root_dir")
    if root_dir_raw is None:
        raise ValueError("candidate_pool.root_dir is required.")
    root_dir = Path(str(root_dir_raw))
    if not root_dir.exists():
        raise FileNotFoundError(f"Candidate pool root directory not found: {root_dir}")

    all_paths = sorted(collect_image_paths(root_dir, recursive=recursive))
    if not all_paths:
        raise ValueError("No candidate patterns found in candidate_pool.root_dir.")

    if max_value is not None and max_value > 0 and len(all_paths) > max_value:
        rng = np.random.default_rng(sample_seed)
        indices = rng.choice(len(all_paths), size=max_value, replace=False)
        sampled_paths = [all_paths[int(idx)] for idx in sorted(indices)]
    else:
        sampled_paths = list(all_paths)

    logger.info(
        "Candidate pool discovered: total=%d sampled=%d (recursive=%s, max_candidates=%s, seed=%s)",
        len(all_paths),
        len(sampled_paths),
        recursive,
        max_value,
        sample_seed,
    )
    return [load_pattern(path) for path in sampled_paths]


def build_pair_indices(
    candidates_a: Sequence[PatternRecord],
    candidates_b: Sequence[PatternRecord],
    max_pairs: int | None,
) -> List[tuple[int, int]]:
    """Build pair indices for A/B candidate combinations.

    Parameters
    ----------
    candidates_a:
        Candidate records for A.
    candidates_b:
        Candidate records for B.
    max_pairs:
        Optional cap on number of evaluated combinations.

    Returns
    -------
    list of tuple of int
        Index pairs for Cartesian product combinations.
    """
    pair_indices: List[tuple[int, int]] = []
    for idx_a, _ in enumerate(candidates_a):
        for idx_b, _ in enumerate(candidates_b):
            pair_indices.append((idx_a, idx_b))
            if max_pairs is not None and max_pairs > 0 and len(pair_indices) >= max_pairs:
                return pair_indices
    return pair_indices


def build_unique_pair_indices(
    candidates: Sequence[PatternRecord],
    max_pairs: int | None,
) -> List[tuple[int, int]]:
    """Build unique (i<j) pair indices for a single candidate pool."""
    pair_indices: List[tuple[int, int]] = []
    total = len(candidates)
    for idx_a in range(total):
        for idx_b in range(idx_a + 1, total):
            pair_indices.append((idx_a, idx_b))
            if max_pairs is not None and max_pairs > 0 and len(pair_indices) >= max_pairs:
                return pair_indices
    return pair_indices


def candidate_paths(records: Iterable[PatternRecord]) -> List[Path]:
    """Return source paths for a sequence of records."""
    return [record.path for record in records]
