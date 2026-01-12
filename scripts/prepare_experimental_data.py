"""Prepare experimental images into a canonical training/eval format."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image


SUPPORTED_EXTENSIONS = (".bmp", ".png", ".tif", ".tiff")


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _to_grayscale_array(arr: np.ndarray, method: str) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if method == "average":
            return arr.astype(np.float32).mean(axis=2)
        weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        if arr.shape[2] < 3:
            return arr.astype(np.float32).mean(axis=2)
        return (arr[..., :3].astype(np.float32) @ weights).astype(np.float32)
    raise ValueError(f"Unsupported image array shape: {arr.shape}")


def _convert_bit_depth(
    arr: np.ndarray, output_bit_depth: str, input_dtype: np.dtype
) -> Tuple[np.ndarray, Dict[str, Any]]:
    meta: Dict[str, Any] = {"scale_factor": 1.0, "scale_applied": False}
    if output_bit_depth == "keep":
        if np.issubdtype(input_dtype, np.integer):
            max_val = np.iinfo(input_dtype).max
            arr = np.clip(arr, 0, max_val).round().astype(input_dtype)
            return arr, meta
        if np.issubdtype(arr.dtype, np.floating):
            max_val = float(arr.max()) if arr.size else 0.0
            scale = 1.0 if max_val <= 1.0 else 1.0 / max_val
            meta.update({"scale_factor": scale, "scale_applied": scale != 1.0})
            arr = (arr * scale).astype(np.float32)
            return arr, meta
        return arr, meta

    if output_bit_depth != "16":
        raise ValueError("output_bit_depth must be 'keep' or '16'.")

    if np.issubdtype(arr.dtype, np.uint16):
        return arr, meta
    if np.issubdtype(input_dtype, np.integer):
        max_val = np.iinfo(input_dtype).max
        scale = 65535.0 / float(max_val)
        meta.update({"scale_factor": scale, "scale_applied": True})
        return np.clip(arr.astype(np.float32) * scale, 0, 65535).astype(np.uint16), meta
    if np.issubdtype(arr.dtype, np.floating):
        max_val = float(arr.max()) if arr.size else 0.0
        scale = 1.0 if max_val <= 1.0 else 1.0 / max_val
        meta.update({"scale_factor": scale, "scale_applied": True})
        return np.clip(arr.astype(np.float32) * scale * 65535.0, 0, 65535).astype(np.uint16), meta
    raise ValueError(f"Unsupported dtype for conversion: {arr.dtype}")


def _write_image(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if array.dtype == np.uint16:
        Image.fromarray(array, mode="I;16").save(path)
        return
    if array.dtype == np.uint8:
        Image.fromarray(array, mode="L").save(path)
        return
    if np.issubdtype(array.dtype, np.floating):
        scaled = np.clip(array, 0.0, 1.0)
        Image.fromarray((scaled * 255.0).astype(np.uint8), mode="L").save(path)
        return
    raise ValueError(f"Unsupported output dtype: {array.dtype}")


def _iter_images(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare experimental images into a canonical training/eval format."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw/Double Pattern Data",
        help="Root folder with experimental images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/Double Pattern Data",
        help="Destination root for prepared outputs.",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="png",
        choices=("png", "tif", "tiff"),
        help="Output image format for prepared data.",
    )
    parser.add_argument(
        "--output-bit-depth",
        type=str,
        default="16",
        choices=("16", "keep"),
        help="Output bit depth (scale to 16-bit or keep raw bit depth).",
    )
    parser.add_argument(
        "--grayscale-method",
        type=str,
        default="luma",
        choices=("luma", "average"),
        help="Grayscale conversion for multi-channel inputs.",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help="Optional path for manifest JSON (defaults to output-dir/manifest.json).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing prepared files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_format = ".tif" if args.output_format in ("tif", "tiff") else ".png"
    manifest_path = Path(args.manifest_path) if args.manifest_path else output_dir / "manifest.json"

    images = _iter_images(input_dir)
    manifest: Dict[str, Any] = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "output_format": output_format,
        "output_bit_depth": args.output_bit_depth,
        "grayscale_method": args.grayscale_method,
        "file_count": len(images),
        "files": [],
    }

    for path in images:
        rel_path = path.relative_to(input_dir)
        out_path = output_dir / rel_path.with_suffix(output_format)
        if out_path.exists() and not args.overwrite:
            continue
        with Image.open(path) as img:
            raw_arr = np.array(img)
            input_mode = img.mode
        arr = _to_grayscale_array(raw_arr, args.grayscale_method)
        converted, meta = _convert_bit_depth(arr, args.output_bit_depth, raw_arr.dtype)
        if converted.dtype == np.float32 and args.output_bit_depth == "keep":
            converted = np.clip(converted, 0.0, 1.0)
        _write_image(out_path, converted)

        record = {
            "input_path": str(path),
            "output_path": str(out_path),
            "input_checksum": _sha256(path),
            "output_checksum": _sha256(out_path),
            "input_shape": list(arr.shape),
            "input_dtype": str(raw_arr.dtype),
            "input_mode": input_mode,
            "output_dtype": str(converted.dtype),
            "output_bit_depth": args.output_bit_depth,
        }
        record.update(meta)
        manifest["files"].append(record)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
