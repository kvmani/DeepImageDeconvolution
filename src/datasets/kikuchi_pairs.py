"""Datasets for Kikuchi pattern deconvolution."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import csv

from src.preprocessing.mask import build_mask_with_metadata
from src.preprocessing.pipeline import PreprocessConfig, apply_preprocess, parse_preprocess_cfg
from src.preprocessing.transforms import (
    apply_crop_indices,
    apply_flip,
    apply_rotation_90,
    compute_crop_indices,
)
from src.utils.io import read_image_16bit, to_float01
from src.utils.logging import get_logger




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


def _normalize_sample_id(path: Path, tag: str) -> str:
    stem = path.stem
    suffix = f"_{tag}"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def _collect_paths(paths: Iterable[Path], tag: str) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for path in paths:
        sample_id = _normalize_sample_id(path, tag)
        if sample_id in mapping:
            raise ValueError(f"Duplicate sample id {sample_id} from {path}")
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


def _apply_preprocess(
    image: np.ndarray,
    cfg: PreprocessConfig,
    mask: Optional[np.ndarray],
) -> np.ndarray:
    return apply_preprocess(image, cfg, mask)


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
    Each sample returns a dict with keys: ``C``, ``A``, ``B``, ``x``, and ``sample_id``.
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
        weights_csv: Optional[Path] = None,
        require_weights: bool = True,
        weight_tolerance: float = 1e-3,
    ) -> None:
        self.logger = get_logger(__name__)
        self.root_dir = root_dir
        self.extensions = extensions
        self.preprocess = parse_preprocess_cfg(preprocess_cfg)
        self.seed = int(seed)
        self.weight_tolerance = float(weight_tolerance)

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
        self.x_map = self._load_weights(root_dir, weights_csv, require_weights)

    def _load_weights(
        self,
        root_dir: Path,
        weights_csv: Optional[Path],
        require_weights: bool,
    ) -> Dict[str, float]:
        if weights_csv is None:
            candidate = root_dir / "metadata.csv"
            weights_csv = candidate if candidate.exists() else None
        if weights_csv is None:
            if require_weights:
                raise ValueError(
                    "Weights CSV not found. Provide data.weights_csv or place metadata.csv in root_dir."
                )
            return {}
        if not weights_csv.exists():
            raise FileNotFoundError(f"Weights CSV not found: {weights_csv}")

        x_map: Dict[str, float] = {}
        with weights_csv.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                sample_id = row.get("sample_id")
                if not sample_id:
                    continue
                if "x" in row and row["x"] != "":
                    x_val = float(row["x"])
                elif "weight_a" in row and row["weight_a"] != "":
                    x_val = float(row["weight_a"])
                else:
                    continue
                x_map[sample_id] = x_val
        if require_weights and not x_map:
            raise ValueError(f"No weights found in {weights_csv}.")
        return x_map

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
            top, left = compute_crop_indices(
                image_c.shape, self.preprocess.crop_size, self.preprocess.crop_mode, rng
            )
            image_a = apply_crop_indices(image_a, top, left, self.preprocess.crop_size)
            image_b = apply_crop_indices(image_b, top, left, self.preprocess.crop_size)
            image_c = apply_crop_indices(image_c, top, left, self.preprocess.crop_size)
        if self.preprocess.augment_enabled:
            flip_h = self.preprocess.flip_horizontal and bool(rng.random() < 0.5)
            flip_v = self.preprocess.flip_vertical and bool(rng.random() < 0.5)
            rot_k = int(rng.integers(0, 4)) if self.preprocess.rotate90 else 0
            image_a = _apply_augmentations(image_a, flip_h, flip_v, rot_k)
            image_b = _apply_augmentations(image_b, flip_h, flip_v, rot_k)
            image_c = _apply_augmentations(image_c, flip_h, flip_v, rot_k)

        mask = None
        if self.preprocess.mask_enabled:
            mask, _ = build_mask_with_metadata(
                image_c,
                detect_existing=self.preprocess.detect_existing,
                zero_tolerance=self.preprocess.zero_tolerance,
                outside_zero_fraction=self.preprocess.outside_zero_fraction,
            )

        image_a = _apply_preprocess(image_a, self.preprocess, mask)
        image_b = _apply_preprocess(image_b, self.preprocess, mask)
        image_c = _apply_preprocess(image_c, self.preprocess, mask)

        if self.x_map:
            if sample_id not in self.x_map:
                raise ValueError(f"Missing weight for sample_id {sample_id}.")
            x_val = float(self.x_map[sample_id])
            if not 0.0 <= x_val <= 1.0:
                raise ValueError(f"x must be in [0, 1], got {x_val} for {sample_id}.")
            y_val = 1.0 - x_val
            if abs((x_val + y_val) - 1.0) > self.weight_tolerance:
                raise ValueError(f"x + y must be 1 for {sample_id}.")
        else:
            x_val = 0.5

        return {
            "C": torch.from_numpy(image_c).unsqueeze(0),
            "A": torch.from_numpy(image_a).unsqueeze(0),
            "B": torch.from_numpy(image_b).unsqueeze(0),
            "x": torch.tensor([x_val], dtype=torch.float32),
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
        self.preprocess = parse_preprocess_cfg(preprocess_cfg)
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
        crop_top = 0
        crop_left = 0
        crop_height = image_c.shape[0]
        crop_width = image_c.shape[1]
        if self.preprocess.crop_enabled:
            if self.preprocess.crop_size is None or len(self.preprocess.crop_size) != 2:
                raise ValueError("crop.size must be a 2-element list when cropping is enabled.")
            top, left = compute_crop_indices(
                image_c.shape, self.preprocess.crop_size, self.preprocess.crop_mode, rng
            )
            image_c = apply_crop_indices(image_c, top, left, self.preprocess.crop_size)
            crop_top = top
            crop_left = left
            crop_height, crop_width = self.preprocess.crop_size
        if self.preprocess.mask_enabled:
            mask, _ = build_mask_with_metadata(
                image_c,
                detect_existing=self.preprocess.detect_existing,
                zero_tolerance=self.preprocess.zero_tolerance,
                outside_zero_fraction=self.preprocess.outside_zero_fraction,
            )

        image_c = _apply_preprocess(image_c, self.preprocess, mask)

        item: Dict[str, torch.Tensor | str] = {
            "C": torch.from_numpy(image_c).unsqueeze(0),
            "sample_id": sample_id,
            "path": str(path_c),
            "crop": {
                "enabled": self.preprocess.crop_enabled,
                "top": crop_top,
                "left": crop_left,
                "height": crop_height,
                "width": crop_width,
            },
        }
        if self.return_mask and mask is not None:
            item["mask"] = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        return item


class KikuchiMixedListDataset(Dataset):
    """Dataset returning mixed patterns from a list of image paths."""

    def __init__(
        self,
        paths: Sequence[Path],
        preprocess_cfg: Dict,
        seed: int = 42,
        limit: Optional[int] = None,
        return_mask: bool = False,
    ) -> None:
        self.logger = get_logger(__name__)
        self.preprocess = parse_preprocess_cfg(preprocess_cfg)
        self.seed = int(seed)
        self.return_mask = return_mask

        if not paths:
            raise ValueError("No mixed samples provided for inference.")
        if limit is not None:
            paths = list(paths)[: int(limit)]

        c_map = _collect_paths(paths, "C")
        if not c_map:
            raise ValueError("No valid mixed samples provided.")
        self.sample_ids = sorted(c_map.keys())
        self.c_map = c_map

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        sample_id = self.sample_ids[idx]
        path_c = self.c_map[sample_id]
        image_c = to_float01(read_image_16bit(path_c))

        mask = None
        rng = np.random.default_rng(self.seed + idx)
        crop_top = 0
        crop_left = 0
        crop_height = image_c.shape[0]
        crop_width = image_c.shape[1]
        if self.preprocess.crop_enabled:
            if self.preprocess.crop_size is None or len(self.preprocess.crop_size) != 2:
                raise ValueError("crop.size must be a 2-element list when cropping is enabled.")
            top, left = compute_crop_indices(
                image_c.shape, self.preprocess.crop_size, self.preprocess.crop_mode, rng
            )
            image_c = apply_crop_indices(image_c, top, left, self.preprocess.crop_size)
            crop_top = top
            crop_left = left
            crop_height, crop_width = self.preprocess.crop_size
        if self.preprocess.mask_enabled:
            mask, _ = build_mask_with_metadata(
                image_c,
                detect_existing=self.preprocess.detect_existing,
                zero_tolerance=self.preprocess.zero_tolerance,
                outside_zero_fraction=self.preprocess.outside_zero_fraction,
            )

        image_c = _apply_preprocess(image_c, self.preprocess, mask)

        item: Dict[str, torch.Tensor | str] = {
            "C": torch.from_numpy(image_c).unsqueeze(0),
            "sample_id": sample_id,
            "path": str(path_c),
            "crop": {
                "enabled": self.preprocess.crop_enabled,
                "top": crop_top,
                "left": crop_left,
                "height": crop_height,
                "width": crop_width,
            },
        }
        if self.return_mask and mask is not None:
            item["mask"] = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        return item
