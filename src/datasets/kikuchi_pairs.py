"""Datasets for Kikuchi pattern deconvolution."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, random_split

from src.preprocessing.mask import apply_circular_mask, build_circular_mask, detect_circular_mask
from src.preprocessing.normalise import normalize_image
from src.preprocessing.transforms import apply_flip, apply_rotation_90
from src.utils.io import read_image_16bit, to_float01
from src.utils.logging import get_logger


@dataclass
class PreprocessConfig:
    """Preprocessing configuration for datasets."""

    crop_enabled: bool
    crop_size: Optional[Tuple[int, int]]
    crop_mode: str
    mask_enabled: bool
    detect_existing: bool
    outside_zero_fraction: float
    zero_tolerance: float
    normalize_enabled: bool
    normalize_method: str
    normalize_smart: bool
    histogram_bins: int
    percentile: Tuple[float, float]
    augment_enabled: bool
    flip_horizontal: bool
    flip_vertical: bool
    rotate90: bool


def _parse_preprocess_cfg(cfg: Dict) -> PreprocessConfig:
    crop_cfg = cfg.get("crop", {})
    mask_cfg = cfg.get("mask", {})
    normalize_cfg = cfg.get("normalize", {})
    augment_cfg = cfg.get("augment", {})
    return PreprocessConfig(
        crop_enabled=bool(crop_cfg.get("enabled", False)),
        crop_size=tuple(crop_cfg.get("size", [])) if crop_cfg.get("size") else None,
        crop_mode=str(crop_cfg.get("mode", "center")),
        mask_enabled=bool(mask_cfg.get("enabled", True)),
        detect_existing=bool(mask_cfg.get("detect_existing", True)),
        outside_zero_fraction=float(mask_cfg.get("outside_zero_fraction", 0.98)),
        zero_tolerance=float(mask_cfg.get("zero_tolerance", 1e-6)),
        normalize_enabled=bool(normalize_cfg.get("enabled", False)),
        normalize_method=str(normalize_cfg.get("method", "min_max")),
        normalize_smart=bool(normalize_cfg.get("smart", True)),
        histogram_bins=int(normalize_cfg.get("histogram_bins", 4096)),
        percentile=tuple(normalize_cfg.get("percentile", (1.0, 99.0))),
        augment_enabled=bool(augment_cfg.get("enabled", False)),
        flip_horizontal=bool(augment_cfg.get("flip_horizontal", True)),
        flip_vertical=bool(augment_cfg.get("flip_vertical", True)),
        rotate90=bool(augment_cfg.get("rotate90", True)),
    )


def _collect_images(directory: Path, extensions: Sequence[str], tag: str) -> Dict[str, Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    exts = {ext.lower() for ext in extensions}
    mapping: Dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in exts:
            continue
        stem = path.stem
        sample_id = stem[:-2] if stem.endswith(f"_{tag}") else stem
        if sample_id in mapping:
            raise ValueError(f"Duplicate sample id {sample_id} in {directory}")
        mapping[sample_id] = path
    return mapping


def _apply_augmentations(
    image: np.ndarray,
    flip_h: bool,
    flip_v: bool,
    rot_k: int,
) -> np.ndarray:
    image = apply_flip(image, flip_horizontal=flip_h, flip_vertical=flip_v)
    image = apply_rotation_90(image, k=rot_k)
    return image


def _compute_crop_indices(
    shape: Tuple[int, int],
    size: Tuple[int, int],
    mode: str,
    rng: np.random.Generator,
) -> Tuple[int, int]:
    height, width = shape
    target_h, target_w = size
    if target_h > height or target_w > width:
        raise ValueError("Crop size must be <= image size.")
    if mode == "center":
        top = (height - target_h) // 2
        left = (width - target_w) // 2
        return top, left
    if mode == "random":
        top = int(rng.integers(0, height - target_h + 1))
        left = int(rng.integers(0, width - target_w + 1))
        return top, left
    raise ValueError(f"Unknown crop mode: {mode}")


def _apply_crop(image: np.ndarray, top: int, left: int, size: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = size
    return image[top : top + target_h, left : left + target_w]


def _build_mask(image: np.ndarray, cfg: PreprocessConfig) -> Tuple[np.ndarray, Dict[str, float]]:
    mask = build_circular_mask(image.shape)
    meta: Dict[str, float] = {}
    if cfg.detect_existing:
        detected, outside_fraction = detect_circular_mask(
            image,
            mask,
            zero_tolerance=cfg.zero_tolerance,
            outside_zero_fraction=cfg.outside_zero_fraction,
        )
        meta["mask_detected"] = float(detected)
        meta["outside_zero_fraction"] = outside_fraction
    return mask, meta


def _apply_preprocess(
    image: np.ndarray,
    cfg: PreprocessConfig,
    mask: Optional[np.ndarray],
) -> np.ndarray:
    if cfg.mask_enabled and mask is not None:
        image = apply_circular_mask(image, mask)

    if cfg.normalize_enabled:
        image = normalize_image(
            image,
            method=cfg.normalize_method,
            histogram_bins=cfg.histogram_bins,
            percentile=cfg.percentile,
            mask=mask,
            smart_minmax=cfg.normalize_smart,
        )

    return image.astype(np.float32)


def split_dataset(
    dataset: Dataset,
    val_split: float,
    seed: int,
) -> Tuple[Dataset, Optional[Dataset]]:
    if not 0.0 <= val_split < 1.0:
        raise ValueError("val_split must be in [0, 1).")
    if val_split == 0.0:
        return dataset, None
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    if val_size == 0:
        return dataset, None
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


class KikuchiPairDataset(Dataset):
    """Dataset returning (C, A, B) triplets.

    Notes
    -----
    Each sample returns a dict with keys: ``C``, ``A``, ``B``, and ``sample_id``.
    Tensors are float32 in [0, 1] with shape ``(1, H, W)``.
    """

    def __init__(
        self,
        root_dir: Path,
        a_dir: str,
        b_dir: str,
        c_dir: str,
        extensions: Sequence[str],
        preprocess_cfg: Dict,
        seed: int = 42,
        limit: Optional[int] = None,
    ) -> None:
        self.logger = get_logger(__name__)
        self.root_dir = root_dir
        self.extensions = extensions
        self.preprocess = _parse_preprocess_cfg(preprocess_cfg)
        self.seed = int(seed)

        a_dir = root_dir / a_dir
        b_dir = root_dir / b_dir
        c_dir = root_dir / c_dir

        a_map = _collect_images(a_dir, extensions, "A")
        b_map = _collect_images(b_dir, extensions, "B")
        c_map = _collect_images(c_dir, extensions, "C")

        sample_ids = sorted(set(a_map) & set(b_map) & set(c_map))
        missing = (set(a_map) | set(b_map) | set(c_map)) - set(sample_ids)
        if missing:
            self.logger.warning("Missing paired samples for %d ids", len(missing))
        if not sample_ids:
            raise ValueError("No paired samples found under the provided root_dir.")

        if limit is not None:
            sample_ids = sample_ids[: int(limit)]

        self.sample_ids = sample_ids
        self.a_map = a_map
        self.b_map = b_map
        self.c_map = c_map

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_id = self.sample_ids[idx]
        path_a = self.a_map[sample_id]
        path_b = self.b_map[sample_id]
        path_c = self.c_map[sample_id]

        image_a = to_float01(read_image_16bit(path_a))
        image_b = to_float01(read_image_16bit(path_b))
        image_c = to_float01(read_image_16bit(path_c))

        if image_a.shape != image_b.shape or image_a.shape != image_c.shape:
            raise ValueError(
                f"Shape mismatch for {sample_id}: {image_a.shape}, {image_b.shape}, {image_c.shape}"
            )

        rng = np.random.default_rng(self.seed + idx)
        if self.preprocess.crop_enabled:
            if self.preprocess.crop_size is None or len(self.preprocess.crop_size) != 2:
                raise ValueError("crop.size must be a 2-element list when cropping is enabled.")
            top, left = _compute_crop_indices(
                image_c.shape, self.preprocess.crop_size, self.preprocess.crop_mode, rng
            )
            image_a = _apply_crop(image_a, top, left, self.preprocess.crop_size)
            image_b = _apply_crop(image_b, top, left, self.preprocess.crop_size)
            image_c = _apply_crop(image_c, top, left, self.preprocess.crop_size)
        if self.preprocess.augment_enabled:
            flip_h = self.preprocess.flip_horizontal and bool(rng.random() < 0.5)
            flip_v = self.preprocess.flip_vertical and bool(rng.random() < 0.5)
            rot_k = int(rng.integers(0, 4)) if self.preprocess.rotate90 else 0
            image_a = _apply_augmentations(image_a, flip_h, flip_v, rot_k)
            image_b = _apply_augmentations(image_b, flip_h, flip_v, rot_k)
            image_c = _apply_augmentations(image_c, flip_h, flip_v, rot_k)

        mask = None
        if self.preprocess.mask_enabled:
            mask, _ = _build_mask(image_c, self.preprocess)

        image_a = _apply_preprocess(image_a, self.preprocess, mask)
        image_b = _apply_preprocess(image_b, self.preprocess, mask)
        image_c = _apply_preprocess(image_c, self.preprocess, mask)

        return {
            "C": torch.from_numpy(image_c).unsqueeze(0),
            "A": torch.from_numpy(image_a).unsqueeze(0),
            "B": torch.from_numpy(image_b).unsqueeze(0),
            "sample_id": sample_id,
        }


class KikuchiMixedDataset(Dataset):
    """Dataset returning mixed patterns only for inference."""

    def __init__(
        self,
        mixed_dir: Path,
        extensions: Sequence[str],
        preprocess_cfg: Dict,
        seed: int = 42,
        limit: Optional[int] = None,
        return_mask: bool = False,
    ) -> None:
        self.logger = get_logger(__name__)
        self.mixed_dir = mixed_dir
        self.extensions = extensions
        self.preprocess = _parse_preprocess_cfg(preprocess_cfg)
        self.seed = int(seed)
        self.return_mask = return_mask

        c_map = _collect_images(mixed_dir, extensions, "C")
        if self.preprocess.augment_enabled:
            self.logger.warning("Augmentations are ignored for inference datasets.")
        sample_ids = sorted(c_map.keys())
        if not sample_ids:
            raise ValueError("No mixed samples found in mixed_dir.")
        if limit is not None:
            sample_ids = sample_ids[: int(limit)]

        self.sample_ids = sample_ids
        self.c_map = c_map

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        sample_id = self.sample_ids[idx]
        path_c = self.c_map[sample_id]
        image_c = to_float01(read_image_16bit(path_c))

        mask = None
        rng = np.random.default_rng(self.seed + idx)
        if self.preprocess.crop_enabled:
            if self.preprocess.crop_size is None or len(self.preprocess.crop_size) != 2:
                raise ValueError("crop.size must be a 2-element list when cropping is enabled.")
            top, left = _compute_crop_indices(
                image_c.shape, self.preprocess.crop_size, self.preprocess.crop_mode, rng
            )
            image_c = _apply_crop(image_c, top, left, self.preprocess.crop_size)
        if self.preprocess.mask_enabled:
            mask, _ = _build_mask(image_c, self.preprocess)

        image_c = _apply_preprocess(image_c, self.preprocess, mask)

        item: Dict[str, torch.Tensor | str] = {
            "C": torch.from_numpy(image_c).unsqueeze(0),
            "sample_id": sample_id,
            "path": str(path_c),
        }
        if self.return_mask and mask is not None:
            item["mask"] = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        return item
