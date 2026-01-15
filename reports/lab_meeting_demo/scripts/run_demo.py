"""Run the lab meeting demo pipeline and generate report artifacts."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.generation.mix import mix_normalize_then_mix, mix_then_normalize
from src.preprocessing.mask import apply_circular_mask, build_mask_with_metadata
from src.preprocessing.normalise import normalize_image
from src.preprocessing.transforms import center_crop_to_min
from src.utils.io import collect_image_paths, read_image_16bit, to_float01, write_image_16bit
from src.utils.logging import (
    collect_environment,
    get_git_commit,
    log_timer,
    resolve_log_level,
    setup_logging,
    write_manifest,
)
from src.utils.reporting import safe_relpath


DEFAULT_WEIGHTS = (0.10, 0.25, 0.50)
DEFAULT_STRATEGY = "mix_then_normalize"
NORMALIZE_METHOD = "min_max"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run the lab meeting mixing demo.")
    parser.add_argument(
        "--out-root",
        default="outputs",
        help="Root output directory (relative to repo root by default).",
    )
    parser.add_argument(
        "--strategy",
        default="weight_sweep=mxnorm",
        help="Weight sweep strategy, e.g. weight_sweep=mxnorm or weight_sweep=normmix.",
    )
    parser.add_argument(
        "--weights",
        default=",".join(f"{w:.2f}" for w in DEFAULT_WEIGHTS),
        help="Comma-separated weights for A (e.g. 0.10,0.25,0.50).",
    )
    parser.add_argument("--A", dest="a_path", default=None, help="Optional path to input A.")
    parser.add_argument("--B", dest="b_path", default=None, help="Optional path to input B.")
    parser.add_argument("--run-tag", default=None, help="Optional run tag for output naming.")
    parser.add_argument("--strict", action="store_true", help="Fail on validation errors.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level.",
    )
    parser.add_argument("--log-file", default=None, help="Optional log file path.")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging to WARNING.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    return parser.parse_args()


def _resolve_path(path_str: str, repo_root: Path) -> Path:
    """Resolve a possibly relative path against the repository root.

    Parameters
    ----------
    path_str:
        Input path string.
    repo_root:
        Repository root path.

    Returns
    -------
    pathlib.Path
        Absolute path to the target.
    """
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return repo_root / path


def _parse_weights(weights_str: str) -> List[float]:
    """Parse comma-separated weights into a list of floats.

    Parameters
    ----------
    weights_str:
        Comma-separated weight list.

    Returns
    -------
    list[float]
        Parsed weights.
    """
    weights: List[float] = []
    for entry in weights_str.split(","):
        entry = entry.strip()
        if not entry:
            continue
        weights.append(float(entry))
    if not weights:
        raise ValueError("No weights provided.")
    for weight in weights:
        if not 0.0 <= weight <= 1.0:
            raise ValueError("Weights must be in [0, 1].")
    return weights


def _parse_strategy(strategy_str: str) -> str:
    """Parse the strategy string into a pipeline name.

    Parameters
    ----------
    strategy_str:
        CLI strategy value.

    Returns
    -------
    str
        Strategy name ('mix_then_normalize' or 'normalize_then_mix').
    """
    value = strategy_str.strip()
    if "=" in value:
        _, value = value.split("=", 1)
    normalized = value.strip().lower()
    if normalized in {"mxnorm", "mix_then_normalize", "mix-then-normalize", "mix_then_norm"}:
        return "mix_then_normalize"
    if normalized in {"normmix", "normalize_then_mix", "normalize-then-mix"}:
        return "normalize_then_mix"
    raise ValueError(f"Unknown strategy: {strategy_str}")


def _collect_images(directory: Path) -> List[Path]:
    """Collect image paths from a directory when it exists.

    Parameters
    ----------
    directory:
        Directory to scan.

    Returns
    -------
    list[pathlib.Path]
        Collected image paths.
    """
    if not directory.exists():
        return []
    return collect_image_paths(directory, recursive=True)


def _prefer_good_patterns(paths: List[Path]) -> List[Path]:
    """Prefer images from the 'Good Pattern' folder when available.

    Parameters
    ----------
    paths:
        Candidate image paths.

    Returns
    -------
    list[pathlib.Path]
        Preferred image paths.
    """
    good = [p for p in paths if any(part.lower() == "good pattern" for part in p.parts)]
    return good or paths


def _find_by_keyword(paths: Iterable[Path], keywords: Iterable[str]) -> Optional[Path]:
    """Find the first path that matches any of the keywords.

    Parameters
    ----------
    paths:
        Candidate image paths.
    keywords:
        Lowercase keywords to match in filenames.

    Returns
    -------
    pathlib.Path or None
        First matching path if found.
    """
    for path in paths:
        name = path.name.lower()
        if any(keyword in name for keyword in keywords):
            return path
    return None


def _select_default_inputs(repo_root: Path) -> Tuple[Path, Path, str]:
    """Select default input paths from preferred or fallback directories.

    Parameters
    ----------
    repo_root:
        Repository root path.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path, str]
        (A_path, B_path, source_label)
    """
    preferred = repo_root / "data" / "raw" / "Double Pattern Data"
    candidates = _collect_images(preferred)
    if candidates:
        source = "data/raw/Double Pattern Data"
        path_a, path_b = _pick_input_pair(candidates)
        return path_a, path_b, source
    fallback_dirs = [
        repo_root / "data" / "processed",
        repo_root / "tests" / "assets",
        repo_root / "tests",
        repo_root / "data",
    ]
    for directory in fallback_dirs:
        candidates = _collect_images(directory)
        if candidates:
            path_a, path_b = _pick_input_pair(candidates)
            return path_a, path_b, str(directory.relative_to(repo_root))
    raise FileNotFoundError("No image inputs found under data/ or tests/.")


def _pick_input_pair(paths: List[Path]) -> Tuple[Path, Path]:
    """Pick two distinct input images, preferring BCC/FCC labels.

    Parameters
    ----------
    paths:
        Candidate image paths.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        Selected input pair.
    """
    paths = sorted(_prefer_good_patterns(paths))
    if not paths:
        raise FileNotFoundError("No candidate input images found.")
    bcc = _find_by_keyword(paths, ("bcc",))
    fcc = _find_by_keyword(paths, ("fcc",))
    if bcc and fcc and bcc != fcc:
        return bcc, fcc
    for i, path in enumerate(paths):
        for other in paths[i + 1 :]:
            if other != path:
                return path, other
    raise FileNotFoundError("Need at least two distinct input images.")


def _read_image_with_meta(path: Path, repo_root: Path) -> Tuple[np.ndarray, Dict[str, object]]:
    """Read an image and return a float32 array with metadata.

    Parameters
    ----------
    path:
        Input image path.
    repo_root:
        Repository root path for relative metadata.

    Returns
    -------
    tuple[numpy.ndarray, dict]
        Float image in [0, 1] and metadata.
    """
    with Image.open(path) as img:
        array = np.array(img)
        meta = {
            "path": safe_relpath(path, repo_root),
            "dtype": str(array.dtype),
            "mode": img.mode,
            "shape": list(array.shape),
            "is_16bit": bool(array.dtype == np.uint16),
        }
    image_16 = read_image_16bit(path)
    image = to_float01(image_16)
    meta["converted_to_16bit"] = bool(array.dtype != np.uint16)
    return image.astype(np.float32), meta


def _build_mix_filename(strategy: str, weight_a: float) -> str:
    """Build a filename for a mixed output image.

    Parameters
    ----------
    strategy:
        Mixing strategy name.
    weight_a:
        Weight for image A.

    Returns
    -------
    str
        Filename for the mixed image.
    """
    weight_b = 1.0 - weight_a
    tag = f"{int(round(weight_a * 100))}_{int(round(weight_b * 100))}"
    if strategy == "mix_then_normalize":
        prefix = "C_mix_then_norm"
    else:
        prefix = "C_normalize_then_mix"
    return f"{prefix}_{tag}.png"


def _mix_images(
    strategy: str,
    weight_a: float,
    image_a_raw: np.ndarray,
    image_b_raw: np.ndarray,
    image_a_norm: np.ndarray,
    image_b_norm: np.ndarray,
    mask: Optional[np.ndarray],
) -> np.ndarray:
    """Mix images using the selected strategy.

    Parameters
    ----------
    strategy:
        Mixing strategy name.
    weight_a:
        Weight for image A.
    image_a_raw:
        Raw (masked) image A.
    image_b_raw:
        Raw (masked) image B.
    image_a_norm:
        Normalized image A.
    image_b_norm:
        Normalized image B.
    mask:
        Optional circular mask.

    Returns
    -------
    numpy.ndarray
        Mixed image in [0, 1].
    """
    if strategy == "normalize_then_mix":
        mixed = mix_normalize_then_mix(
            image_a_norm,
            image_b_norm,
            weight_a=weight_a,
            normalize_after_mix=False,
            normalize_method=NORMALIZE_METHOD,
            mask=mask,
            normalize_smart=True,
        )
    else:
        mixed = mix_then_normalize(
            image_a_raw,
            image_b_raw,
            weight_a=weight_a,
            normalize_method=NORMALIZE_METHOD,
            mask=mask,
            normalize_smart=True,
        )
    if mask is not None:
        mixed = apply_circular_mask(mixed, mask)
    return np.clip(mixed, 0.0, 1.0).astype(np.float32)


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    """Write JSON to disk.

    Parameters
    ----------
    path:
        Output path.
    payload:
        JSON-serializable payload.
    """
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    """Run the demo pipeline.

    Returns
    -------
    None
    """
    args = parse_args()
    weights = _parse_weights(args.weights)
    strategy = _parse_strategy(args.strategy)

    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"lab_meeting_demo_{run_tag}"

    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root
    run_dir = out_root / run_id
    demo_dir = run_dir / "demo_artifacts"
    figures_dir = run_dir / "figures"
    logs_dir = run_dir / "logs"

    log_level = resolve_log_level(args.log_level, debug=args.debug, quiet=args.quiet)
    log_file = Path(args.log_file) if args.log_file else logs_dir / "demo.log"
    logger = setup_logging(
        "lab_meeting_demo", level=log_level, log_file=log_file, run_id=run_id
    )

    start_time = time.perf_counter()
    status = "failed"
    error: Optional[str] = None
    git_commit = get_git_commit(REPO_ROOT) or "unknown"
    run_meta: Dict[str, object] = {}
    inputs_meta: Dict[str, object] = {}
    report_path = run_dir / "report.json"
    meta_path = run_dir / "demo_metadata.json"
    timestamp = datetime.now().isoformat(timespec="seconds")

    logger.info("Starting lab meeting demo run: %s", run_id)
    logger.info("Output directory: %s", run_dir)

    try:
        if args.debug:
            np.random.seed(0)
            logger.debug("Debug mode enabled with deterministic seed.")

        if args.a_path:
            a_path = _resolve_path(args.a_path, REPO_ROOT)
        else:
            a_path = None
        if args.b_path:
            b_path = _resolve_path(args.b_path, REPO_ROOT)
        else:
            b_path = None

        source_note = "manual"
        if a_path is None or b_path is None:
            auto_a, auto_b, source = _select_default_inputs(REPO_ROOT)
            source_note = source
            a_path = a_path or auto_a
            b_path = b_path or auto_b

        if a_path == b_path:
            raise ValueError("A and B paths must be different.")
        if not a_path.exists() or not b_path.exists():
            raise FileNotFoundError(f"Missing inputs: {a_path}, {b_path}")

        logger.info("Input selection (%s): A=%s", source_note, a_path)
        logger.info("Input selection (%s): B=%s", source_note, b_path)

        with log_timer(logger, "Load inputs"):
            image_a_raw, meta_a = _read_image_with_meta(a_path, REPO_ROOT)
            image_b_raw, meta_b = _read_image_with_meta(b_path, REPO_ROOT)

        logger.info(
            "A dtype=%s shape=%s converted=%s",
            meta_a.get("dtype"),
            meta_a.get("shape"),
            meta_a.get("converted_to_16bit"),
        )
        logger.info(
            "B dtype=%s shape=%s converted=%s",
            meta_b.get("dtype"),
            meta_b.get("shape"),
            meta_b.get("converted_to_16bit"),
        )

        original_shapes = {"A": list(image_a_raw.shape), "B": list(image_b_raw.shape)}
        crop_applied = False
        if image_a_raw.shape != image_b_raw.shape:
            image_a_raw, image_b_raw = center_crop_to_min(image_a_raw, image_b_raw)
            crop_applied = True
            logger.warning(
                "Center-cropped inputs to %s (A was %s, B was %s)",
                image_a_raw.shape,
                original_shapes["A"],
                original_shapes["B"],
            )

        mask = None
        mask_meta: Dict[str, object] = {}
        mask_enabled = True
        if mask_enabled:
            mask, _ = build_mask_with_metadata(
                image_a_raw,
                detect_existing=False,
                zero_tolerance=5e-4,
                outside_zero_fraction=0.98,
            )
            _, meta_mask_a = build_mask_with_metadata(
                image_a_raw,
                mask=mask,
                detect_existing=True,
                zero_tolerance=5e-4,
                outside_zero_fraction=0.98,
            )
            _, meta_mask_b = build_mask_with_metadata(
                image_b_raw,
                mask=mask,
                detect_existing=True,
                zero_tolerance=5e-4,
                outside_zero_fraction=0.98,
            )
            mask_meta = {
                "mask_detected_a": meta_mask_a.get("already_masked"),
                "mask_detected_b": meta_mask_b.get("already_masked"),
                "mask_outside_zero_fraction_a": meta_mask_a.get("outside_zero_fraction"),
                "mask_outside_zero_fraction_b": meta_mask_b.get("outside_zero_fraction"),
            }
            logger.info(
                "Mask detection A=%s B=%s",
                mask_meta["mask_detected_a"],
                mask_meta["mask_detected_b"],
            )
            image_a_raw = apply_circular_mask(image_a_raw, mask)
            image_b_raw = apply_circular_mask(image_b_raw, mask)

        image_a_norm = normalize_image(
            image_a_raw,
            method=NORMALIZE_METHOD,
            mask=mask,
            smart_minmax=True,
        )
        image_b_norm = normalize_image(
            image_b_raw,
            method=NORMALIZE_METHOD,
            mask=mask,
            smart_minmax=True,
        )

        demo_dir.mkdir(parents=True, exist_ok=True)
        write_image_16bit(demo_dir / "A.png", image_a_raw)
        write_image_16bit(demo_dir / "B.png", image_b_raw)

        logger.info("Saved demo artifacts A/B to %s", demo_dir)

        weight_outputs: List[Dict[str, object]] = []
        sweep_strategy = strategy
        for weight in weights:
            mixed = _mix_images(
                sweep_strategy,
                weight,
                image_a_raw,
                image_b_raw,
                image_a_norm,
                image_b_norm,
                mask,
            )
            filename = _build_mix_filename(sweep_strategy, weight)
            out_path = demo_dir / filename
            write_image_16bit(out_path, mixed)
            weight_outputs.append(
                {
                    "x": float(weight),
                    "y": float(1.0 - weight),
                    "path": safe_relpath(out_path, REPO_ROOT),
                }
            )
            logger.info("Saved weight sweep %s: %s", filename, out_path)

        compare_weight = 0.50
        compare_paths: Dict[str, str] = {}
        for strategy_name in ("mix_then_normalize", "normalize_then_mix"):
            if strategy_name == sweep_strategy:
                match = next(
                    (
                        item
                        for item in weight_outputs
                        if abs(item["x"] - compare_weight) < 1e-6
                    ),
                    None,
                )
                if match:
                    compare_paths[strategy_name] = str(match["path"])
                    continue
            mixed = _mix_images(
                strategy_name,
                compare_weight,
                image_a_raw,
                image_b_raw,
                image_a_norm,
                image_b_norm,
                mask,
            )
            filename = _build_mix_filename(strategy_name, compare_weight)
            out_path = demo_dir / filename
            write_image_16bit(out_path, mixed)
            compare_paths[strategy_name] = safe_relpath(out_path, REPO_ROOT)
            logger.info("Saved comparison %s: %s", filename, out_path)

        strategy_label = (
            "mix->normalize"
            if sweep_strategy == "mix_then_normalize"
            else "normalize->mix"
        )
        weights_label = ", ".join(
            f"{weight:.2f}/{1.0 - weight:.2f}" for weight in weights
        )
        notes = [
            f"Generated C for x/y = {weights_label} using {strategy_label}",
            f"Comparison generated at x/y={compare_weight:.2f}/{1.0 - compare_weight:.2f}",
        ]
        if crop_applied:
            notes.append("Inputs center-cropped to match shapes")

        run_meta = {
            "run_id": run_id,
            "timestamp": timestamp,
            "git_commit": git_commit,
            "inputs": {
                "A_source": safe_relpath(a_path, REPO_ROOT),
                "B_source": safe_relpath(b_path, REPO_ROOT),
                "A_demo": safe_relpath(demo_dir / "A.png", REPO_ROOT),
                "B_demo": safe_relpath(demo_dir / "B.png", REPO_ROOT),
            },
            "preflight": {
                "input_source": source_note,
                "weights": weights,
                "weight_sweep_strategy": sweep_strategy,
            },
            "processing": {
                "mask_enabled": mask_enabled,
                "normalize_method": NORMALIZE_METHOD,
                "normalize_smart": True,
                "crop_applied": crop_applied,
                "original_shapes": original_shapes,
                "final_shape": list(image_a_raw.shape),
                **mask_meta,
                "meta_a": meta_a,
                "meta_b": meta_b,
            },
            "weight_sweep": {
                "strategy": sweep_strategy,
                "weights": weight_outputs,
            },
            "strategy_compare": {
                "weight": {"x": compare_weight, "y": 1.0 - compare_weight},
                "mix_then_normalize": compare_paths.get("mix_then_normalize"),
                "normalize_then_mix": compare_paths.get("normalize_then_mix"),
            },
            "notes": notes,
        }
        inputs_meta = run_meta["inputs"]

        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(meta_path, run_meta)
        logger.info("Wrote demo metadata: %s", meta_path)

        make_figures_cmd = [
            sys.executable,
            str(
                REPO_ROOT
                / "reports"
                / "lab_meeting_demo"
                / "scripts"
                / "make_figures.py"
            ),
            "--run-id",
            run_id,
            "--out-root",
            str(out_root),
        ]
        write_report_cmd = [
            sys.executable,
            str(
                REPO_ROOT
                / "reports"
                / "lab_meeting_demo"
                / "scripts"
                / "write_report.py"
            ),
            "--run-id",
            run_id,
            "--out-root",
            str(out_root),
        ]
        validate_cmd = [
            sys.executable,
            str(
                REPO_ROOT
                / "reports"
                / "lab_meeting_demo"
                / "scripts"
                / "validate_report.py"
            ),
            "--run-id",
            run_id,
            "--out-root",
            str(out_root),
        ]

        logger.info("Generating figures...")
        subprocess.run(make_figures_cmd, check=True)
        logger.info("Writing report.json...")
        subprocess.run(write_report_cmd, check=True)
        logger.info("Validating report.json...")
        try:
            subprocess.run(validate_cmd, check=True)
        except subprocess.CalledProcessError:
            if args.strict:
                raise
            logger.warning(
                "Report validation failed; continue because --strict is not set."
            )

        data_dir = REPO_ROOT / "reports" / "lab_meeting_demo" / "_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        latest_path = data_dir / "latest.json"
        if report_path.exists():
            latest_path.write_text(report_path.read_text())
            logger.info("Updated latest report pointer: %s", latest_path)

        reports_fig_dir = REPO_ROOT / "reports" / "lab_meeting_demo" / "figures"
        reports_fig_dir.mkdir(parents=True, exist_ok=True)
        for fig_name in ("weight_sweep_grid.png", "strategy_compare_grid.png"):
            src = run_dir / "figures" / fig_name
            if src.exists():
                dest = reports_fig_dir / fig_name
                dest.write_bytes(src.read_bytes())

        status = "success"
    except Exception as exc:  # pragma: no cover - logging and manifest on failure
        error = str(exc)
        logger.exception("Demo failed")
        raise
    finally:
        elapsed = time.perf_counter() - start_time
        env_meta = collect_environment()
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "run_id": run_id,
            "stage": "demo",
            "timestamp": run_meta.get("timestamp", timestamp),
            "git_commit": git_commit,
            "status": status,
            "args": vars(args),
            "inputs": inputs_meta,
            "outputs": {
                "run_dir": safe_relpath(run_dir, REPO_ROOT),
                "demo_metadata": safe_relpath(meta_path, REPO_ROOT),
                "report": safe_relpath(report_path, REPO_ROOT),
            },
            "environment": env_meta,
            "timing": {"elapsed_s": round(elapsed, 3)},
        }
        if error:
            manifest["error"] = error
        write_manifest(run_dir, manifest)
        if status == "success":
            logger.info("Demo completed in %.2f seconds", elapsed)
            logger.info("Output folder: %s", run_dir)
            logger.info("Report: %s", report_path)
            logger.info("Figures: %s", figures_dir)


if __name__ == "__main__":
    main()
