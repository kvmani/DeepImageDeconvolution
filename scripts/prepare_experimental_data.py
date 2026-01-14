"""Prepare experimental images into a canonical training/eval format."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

from src.utils.logging import (
    ProgressLogger,
    collect_environment,
    get_git_commit,
    resolve_log_level,
    setup_logging,
    summarize_images,
    write_manifest,
)

SUPPORTED_EXTENSIONS = (".bmp", ".png", ".tif", ".tiff", ".jpg", ".jpeg")


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
        choices=("png",),
        help="Output image format for prepared data (16-bit PNG only).",
    )
    parser.add_argument(
        "--output-bit-depth",
        type=str,
        default="16",
        choices=("16",),
        help="Output bit depth (always scales to 16-bit).",
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
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Limit the number of input files processed.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (limits samples and enables verbose logging).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional log file path.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging to WARNING and above.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run identifier to include in logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_level = resolve_log_level(args.log_level, debug=args.debug, quiet=args.quiet)
    log_file = Path(args.log_file) if args.log_file else None
    logger = setup_logging("prepare_experimental_data", level=log_level, log_file=log_file, run_id=args.run_id)

    start_time = time.perf_counter()
    attempted = 0
    succeeded = 0
    failed = 0
    skipped = 0
    failures: List[Dict[str, str]] = []

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_format = ".png"
    manifest_path = Path(args.manifest_path) if args.manifest_path else output_dir / "manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": get_git_commit(Path(__file__).resolve().parents[1]),
        "args": vars(args),
        "environment": collect_environment(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "output_format": output_format,
        "output_bit_depth": args.output_bit_depth,
        "grayscale_method": args.grayscale_method,
        "files": [],
        "failures": failures,
    }

    try:
        if not input_dir.exists():
            logger.error("Input directory does not exist: %s", input_dir)
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        images = _iter_images(input_dir)
        sample_limit = args.sample_limit
        if args.debug and sample_limit is None:
            sample_limit = 20
        if sample_limit is not None:
            images = images[: int(sample_limit)]
            logger.info("Debug/sample limit enabled: %d files", len(images))

        if not images:
            logger.error("No input images found in %s", input_dir)
            raise ValueError("No input images found.")

        input_summary = summarize_images(images)
        logger.info("Resolved input dir: %s", input_dir.resolve())
        logger.info("Resolved output dir: %s", output_dir.resolve())
        logger.info("Discovered %d input files", len(images))
        logger.info(
            "Input summary: extensions=%s | size range=%s -> %s | sample dtypes=%s | modes=%s",
            input_summary.get("extensions"),
            input_summary.get("min_size"),
            input_summary.get("max_size"),
            input_summary.get("sample_dtypes"),
            input_summary.get("sample_modes"),
        )
        logger.info(
            "Config: output_bit_depth=%s | grayscale=%s | overwrite=%s",
            args.output_bit_depth,
            args.grayscale_method,
            args.overwrite,
        )

        progress_every = max(1, len(images) // 10)
        progress = ProgressLogger(total=len(images), logger=logger, every=progress_every, unit="img")

        for path in images:
            attempted += 1
            rel_path = path.relative_to(input_dir)
            out_path = output_dir / rel_path.with_suffix(output_format)
            if out_path.exists() and not args.overwrite:
                skipped += 1
                progress.update(1)
                continue
            try:
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
                succeeded += 1
            except Exception as exc:
                failed += 1
                failures.append({"path": str(path), "error": str(exc)})
                logger.exception("Failed to process %s", path)
                raise
            finally:
                progress.update(1)
    except Exception:
        logger.error("Preparation aborted after %d attempted files.", attempted)
        raise
    finally:
        duration = time.perf_counter() - start_time
        output_images = _iter_images(output_dir) if output_dir.exists() else []
        output_summary = summarize_images(output_images) if output_images else {}
        manifest.update(
            {
                "timings": {"wall_time_s": duration},
                "counts": {
                    "attempted": attempted,
                    "succeeded": succeeded,
                    "failed": failed,
                    "skipped": skipped,
                    "outputs": len(output_images),
                },
                "output_summary": output_summary,
            }
        )
        write_manifest(output_dir, manifest)
        if manifest_path != output_dir / "manifest.json":
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, indent=2))
        if failures:
            error_report = output_dir / "error_report.json"
            error_report.write_text(json.dumps(failures, indent=2))
            logger.warning("Failures recorded in %s", error_report)

        logger.info(
            "Summary: attempted=%d | succeeded=%d | failed=%d | skipped=%d | outputs=%d | runtime=%.2fs | output_dir=%s",
            attempted,
            succeeded,
            failed,
            skipped,
            len(output_images),
            duration,
            output_dir,
        )


if __name__ == "__main__":
    main()
