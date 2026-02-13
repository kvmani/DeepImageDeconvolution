"""CLI for synthetic deterministic pair-identification demo (A/B -> C -> recover A/B/x)."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.deterministic_mixing_inversion import (  # noqa: E402
    IdentificationResult,
    SyntheticNoiseConfig,
    build_synthetic_case,
    identify_pair_from_candidates,
    sample_random_candidates,
)
from src.deterministic_mixing_inversion.io import PatternRecord  # noqa: E402
from src.deterministic_mixing_inversion.alignment import (  # noqa: E402
    RigidAlignment,
    apply_rigid_alignment,
    parse_alignment_settings,
)
from src.deterministic_mixing_inversion.preprocess import (  # noqa: E402
    build_centered_mask,
    match_shape_to_target,
    parse_preprocess_settings,
)
from src.deterministic_mixing_inversion.reporting import update_progress_report  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.io import write_image_16bit  # noqa: E402
from src.utils.logging import (  # noqa: E402
    collect_environment,
    get_git_commit,
    resolve_log_level,
    setup_logging,
    write_manifest,
)
from src.utils.run import resolve_run_dir  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs/deterministic_inversion/pair_demo_default.yaml"
DEBUG_CONFIG = REPO_ROOT / "configs/deterministic_inversion/pair_demo_debug.yaml"


def _parse_set_overrides(raw_overrides: List[str]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for raw_override in raw_overrides:
        if "=" not in raw_override:
            raise ValueError(f"Invalid --set override '{raw_override}'. Expected key=value.")
        key_path, value_text = raw_override.split("=", 1)
        value = yaml.safe_load(value_text)
        cursor = parsed
        parts = key_path.split(".")
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = value
    return parsed


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _ranked_payload(result: IdentificationResult) -> Dict[str, Any]:
    return {
        "winner": {
            "rank": result.winner.rank,
            "index_a": result.winner.index_a,
            "index_b": result.winner.index_b,
            "id_a": result.winner.id_a,
            "id_b": result.winner.id_b,
            "x_hat": result.winner.x_hat,
            "primary_score": result.winner.primary_score,
            "l2_score": result.winner.l2_score,
            "metric_scores": result.winner.metric_scores,
            "alignment": result.winner.alignment,
            "top_margin": result.winner.top_margin,
        },
        "top_k": [
            {
                "rank": row.rank,
                "index_a": row.index_a,
                "index_b": row.index_b,
                "id_a": row.id_a,
                "id_b": row.id_b,
                "x_hat": row.x_hat,
                "primary_score": row.primary_score,
                "l2_score": row.l2_score,
                "metric_scores": row.metric_scores,
                "alignment": row.alignment,
                "top_margin": row.top_margin,
            }
            for row in result.top_k
        ],
        "primary_metric": result.primary_metric,
        "total_pairs": result.total_pairs,
        "runtime_s": result.runtime_s,
    }


def _write_topk_csv(path: Path, result: IdentificationResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "index_a",
                "index_b",
                "id_a",
                "id_b",
                "x_hat",
                "primary_score",
                "l2_score",
                "top_margin",
                "angle_deg",
                "shift_y",
                "shift_x",
            ],
        )
        writer.writeheader()
        for row in result.top_k:
            writer.writerow(
                {
                    "rank": row.rank,
                    "index_a": row.index_a,
                    "index_b": row.index_b,
                    "id_a": row.id_a,
                    "id_b": row.id_b,
                    "x_hat": row.x_hat,
                    "primary_score": row.primary_score,
                    "l2_score": row.l2_score,
                    "top_margin": row.top_margin,
                    "angle_deg": row.alignment.get("angle_deg"),
                    "shift_y": row.alignment.get("shift_y"),
                    "shift_x": row.alignment.get("shift_x"),
                }
            )


def _build_c_hat_from_winner(
    candidates: List[PatternRecord],
    mixed_c: np.ndarray,
    result: IdentificationResult,
    inversion_cfg: Dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    preprocess_settings = parse_preprocess_settings(inversion_cfg.get("preprocess", {}))
    alignment_settings = parse_alignment_settings(inversion_cfg.get("alignment", {}))
    target_shape = mixed_c.shape
    mask = build_centered_mask(target_shape) if preprocess_settings.mask_enabled else None

    winner_a = candidates[result.winner.index_a]
    winner_b = candidates[result.winner.index_b]
    image_a = match_shape_to_target(
        winner_a.image,
        target_shape=target_shape,
        auto_crop_to_target=preprocess_settings.auto_crop_to_target,
    ).astype(np.float32)
    image_b = match_shape_to_target(
        winner_b.image,
        target_shape=target_shape,
        auto_crop_to_target=preprocess_settings.auto_crop_to_target,
    ).astype(np.float32)
    if mask is not None:
        image_a = image_a.copy()
        image_b = image_b.copy()
        image_a[~mask] = 0.0
        image_b[~mask] = 0.0

    c_hat = (result.winner.x_hat * image_a + (1.0 - result.winner.x_hat) * image_b).astype(np.float32)
    rigid = RigidAlignment(
        angle_deg=float(result.winner.alignment.get("angle_deg", 0.0)),
        shift_y=float(result.winner.alignment.get("shift_y", 0.0)),
        shift_x=float(result.winner.alignment.get("shift_x", 0.0)),
        score=0.0,
    )
    c_hat_aligned = apply_rigid_alignment(
        c_hat,
        rigid,
        interpolation_order=alignment_settings.interpolation_order,
        mask=mask,
    ).astype(np.float32)
    residual_abs = np.abs(c_hat_aligned - mixed_c).astype(np.float32)
    return image_a, image_b, np.clip(c_hat_aligned, 0.0, 1.0), np.clip(residual_abs, 0.0, 1.0)


def _write_html_report(path: Path, payload: Dict[str, Any]) -> None:
    winner = payload.get("winner", {})
    true_pair = payload.get("true_pair", {})
    html = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Deterministic Pair Demo</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:20px;}",
        ".row{display:flex;gap:12px;flex-wrap:wrap;}",
        ".card{border:1px solid #ddd;border-radius:8px;padding:8px;}",
        ".img{width:240px;height:auto;border:1px solid #eee;}",
        "table{border-collapse:collapse;width:100%;margin-top:16px;}",
        "th,td{border:1px solid #ddd;padding:6px;text-align:left;}",
        "th{background:#f5f5f5;}",
        "</style></head><body>",
        "<h1>Deterministic Pair Identification Demo</h1>",
        (
            f"<p>True pair: {true_pair.get('id_a')} + {true_pair.get('id_b')} | "
            f"x_true={true_pair.get('x_true')} | "
            f"Predicted: {winner.get('id_a')} + {winner.get('id_b')} | "
            f"x_hat={winner.get('x_hat')}</p>"
        ),
        "<div class='row'>",
        f"<div class='card'><h3>A noisy</h3><img class='img' src='{payload['images']['a_noisy']}'/></div>",
        f"<div class='card'><h3>B noisy</h3><img class='img' src='{payload['images']['b_noisy']}'/></div>",
        f"<div class='card'><h3>C synthetic</h3><img class='img' src='{payload['images']['c_synthetic']}'/></div>",
        f"<div class='card'><h3>C_hat winner</h3><img class='img' src='{payload['images']['winner_c_hat']}'/></div>",
        f"<div class='card'><h3>|Residual|</h3><img class='img' src='{payload['images']['winner_residual']}'/></div>",
        "</div>",
        "<table><tr><th>Rank</th><th>A</th><th>B</th><th>x_hat</th><th>Primary</th><th>L2</th></tr>",
    ]
    for row in payload.get("top_k", []):
        html.append(
            "<tr>"
            f"<td>{row.get('rank')}</td>"
            f"<td>{row.get('id_a')}</td>"
            f"<td>{row.get('id_b')}</td>"
            f"<td>{row.get('x_hat')}</td>"
            f"<td>{row.get('primary_score')}</td>"
            f"<td>{row.get('l2_score')}</td>"
            "</tr>"
        )
    html.extend(["</table>", "</body></html>"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(html), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic deterministic pair-identification demo.")
    parser.add_argument(
        "--candidate-dir",
        type=str,
        default="data/raw/Double Pattern Data/Good Pattern",
        help="Directory containing candidate patterns.",
    )
    parser.add_argument("--candidate-count", type=int, default=10, help="Number of candidates to sample.")
    parser.add_argument("--sample-seed", type=int, default=7, help="Seed for candidate sampling.")
    parser.add_argument("--index-a", type=int, required=True, help="Selected A index from sampled candidates.")
    parser.add_argument("--index-b", type=int, required=True, help="Selected B index from sampled candidates.")
    parser.add_argument("--x", type=float, required=True, help="Mixing fraction x for A in [0,1].")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k pair rankings to keep.")
    parser.add_argument("--coarse-top-m", type=int, default=20, help="Refine top-m pairs in stage-2.")
    parser.add_argument(
        "--gaussian-enabled",
        dest="gaussian_enabled",
        action="store_true",
        default=None,
        help="Enable Gaussian noise on A/B.",
    )
    parser.add_argument(
        "--gaussian-disabled",
        dest="gaussian_enabled",
        action="store_false",
        default=None,
        help="Disable Gaussian noise on A/B.",
    )
    parser.add_argument("--gaussian-std", type=float, default=None, help="Gaussian noise sigma.")
    parser.add_argument(
        "--salt-pepper-enabled",
        dest="salt_pepper_enabled",
        action="store_true",
        default=None,
        help="Enable salt-pepper noise.",
    )
    parser.add_argument(
        "--salt-pepper-disabled",
        dest="salt_pepper_enabled",
        action="store_false",
        default=None,
        help="Disable salt-pepper noise.",
    )
    parser.add_argument("--salt-pepper-amount", type=float, default=None, help="Salt-pepper pixel fraction.")
    parser.add_argument("--salt-vs-pepper", type=float, default=None, help="Salt ratio in salt-pepper noise.")
    parser.add_argument(
        "--rotation-enabled",
        dest="rotation_enabled",
        action="store_true",
        default=None,
        help="Enable rotation noise.",
    )
    parser.add_argument(
        "--rotation-disabled",
        dest="rotation_enabled",
        action="store_false",
        default=None,
        help="Disable rotation noise.",
    )
    parser.add_argument("--rotation-max-deg", type=float, default=None, help="Max absolute rotation in degrees.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    parser.add_argument("--out-dir", type=str, default="outputs/gui_pair_demo", help="Output directory.")
    parser.add_argument("--run-tag", type=str, default=None, help="Append timestamped run tag.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="key=value",
        help="Repeatable dot-path override.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug defaults.")
    parser.add_argument("--log-level", type=str, default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    parser.add_argument("--log-file", type=str, default=None, help="Optional log file path.")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging verbosity.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_level = resolve_log_level(args.log_level, debug=args.debug, quiet=args.quiet)
    logger = setup_logging(
        "run_deterministic_pair_demo",
        level=log_level,
        log_file=Path(args.log_file) if args.log_file else None,
        run_id=args.run_tag,
    )

    config_path = Path(args.config) if args.config else (DEBUG_CONFIG if args.debug else DEFAULT_CONFIG)
    config = load_config(config_path)
    overrides = _parse_set_overrides(args.set)
    if overrides:
        config = _deep_update(config, overrides)

    inversion_cfg = config.get("deterministic_inversion", {})
    if not isinstance(inversion_cfg, dict):
        raise ValueError("Config field deterministic_inversion must be a mapping.")
    inversion_cfg.setdefault("pair_search", {})
    if isinstance(inversion_cfg["pair_search"], dict):
        inversion_cfg["pair_search"]["two_stage_enabled"] = True
        inversion_cfg["pair_search"]["coarse_top_m"] = int(max(args.coarse_top_m, 2))

    out_dir = Path(args.out_dir)
    if args.run_tag:
        out_dir = resolve_run_dir(out_dir, args.run_tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": get_git_commit(REPO_ROOT),
        "args": vars(args),
        "environment": collect_environment(),
        "config_path": str(config_path),
        "config": config,
        "output_dir": str(out_dir),
        "failures": [],
    }
    start = time.perf_counter()
    summary: Dict[str, Any] = {}
    update_progress_report(
        out_dir,
        {
            "status": "running",
            "progress": {"current": 0, "total": 1, "percent": 0.0, "eta_s": None},
            "summary": {},
        },
    )

    try:
        candidates = sample_random_candidates(
            candidate_dir=Path(args.candidate_dir),
            sample_count=args.candidate_count,
            seed=args.sample_seed,
            recursive=False,
            logger=logger,
        )
        synthetic_cfg = config.get("synthetic_case", {}) if isinstance(config, dict) else {}
        noise_defaults = synthetic_cfg.get("noise", {}) if isinstance(synthetic_cfg, dict) else {}
        gaussian_enabled = (
            bool(noise_defaults.get("gaussian_enabled", True))
            if args.gaussian_enabled is None
            else bool(args.gaussian_enabled)
        )
        gaussian_std = (
            float(noise_defaults.get("gaussian_std", 0.01))
            if args.gaussian_std is None
            else float(max(args.gaussian_std, 0.0))
        )
        salt_pepper_enabled = (
            bool(noise_defaults.get("salt_pepper_enabled", False))
            if args.salt_pepper_enabled is None
            else bool(args.salt_pepper_enabled)
        )
        salt_pepper_amount = (
            float(noise_defaults.get("salt_pepper_amount", 0.01))
            if args.salt_pepper_amount is None
            else float(max(args.salt_pepper_amount, 0.0))
        )
        salt_vs_pepper = (
            float(noise_defaults.get("salt_vs_pepper", 0.5))
            if args.salt_vs_pepper is None
            else float(np.clip(args.salt_vs_pepper, 0.0, 1.0))
        )
        rotation_enabled = (
            bool(noise_defaults.get("rotation_enabled", True))
            if args.rotation_enabled is None
            else bool(args.rotation_enabled)
        )
        rotation_max_deg = (
            float(noise_defaults.get("rotation_max_deg", 2.0))
            if args.rotation_max_deg is None
            else float(max(args.rotation_max_deg, 0.0))
        )

        candidate_manifest = {
            "candidate_paths": [str(record.path) for record in candidates],
            "candidate_ids": [record.pattern_id for record in candidates],
        }
        (out_dir / "candidate_manifest.json").write_text(json.dumps(candidate_manifest, indent=2), encoding="utf-8")

        noise = SyntheticNoiseConfig(
            gaussian_enabled=gaussian_enabled,
            gaussian_std=gaussian_std,
            salt_pepper_enabled=salt_pepper_enabled,
            salt_pepper_amount=salt_pepper_amount,
            salt_vs_pepper=salt_vs_pepper,
            rotation_enabled=rotation_enabled,
            rotation_max_deg=rotation_max_deg,
        )
        case = build_synthetic_case(
            candidates=candidates,
            index_a=args.index_a,
            index_b=args.index_b,
            mix_fraction=float(np.clip(args.x, 0.0, 1.0)),
            noise=noise,
            seed=args.sample_seed,
            mask_enabled=True,
        )

        synth_dir = out_dir / "synthetic"
        synth_dir.mkdir(parents=True, exist_ok=True)
        write_image_16bit(synth_dir / "a_noisy.png", case.pattern_a_noisy)
        write_image_16bit(synth_dir / "b_noisy.png", case.pattern_b_noisy)
        write_image_16bit(synth_dir / "c_synthetic.png", case.mixed_c)

        def on_progress(processed: int, total: int, eta_s: float, message: str) -> None:
            logger.info("%s | ETA %.2fs", message, eta_s)
            update_progress_report(
                out_dir,
                {
                    "status": "running",
                    "progress": {
                        "current": processed,
                        "total": total,
                        "percent": 100.0 * (processed / max(total, 1)),
                        "eta_s": eta_s,
                    },
                    "summary": {},
                },
            )

        result = identify_pair_from_candidates(
            candidates=candidates,
            mixed_image=case.mixed_c,
            inversion_cfg=inversion_cfg,
            top_k=args.top_k,
            progress_callback=on_progress,
            logger=logger,
        )

        winner_a, winner_b, winner_c_hat, winner_residual = _build_c_hat_from_winner(
            candidates=candidates,
            mixed_c=case.mixed_c,
            result=result,
            inversion_cfg=inversion_cfg,
        )
        recon_dir = out_dir / "reconstructions"
        recon_dir.mkdir(parents=True, exist_ok=True)
        write_image_16bit(recon_dir / "winner_a.png", winner_a)
        write_image_16bit(recon_dir / "winner_b.png", winner_b)
        write_image_16bit(recon_dir / "winner_c_hat.png", winner_c_hat)
        write_image_16bit(recon_dir / "winner_residual_abs.png", winner_residual)

        payload = _ranked_payload(result)
        payload["true_pair"] = {
            "index_a": args.index_a,
            "index_b": args.index_b,
            "id_a": case.candidate_a.pattern_id,
            "id_b": case.candidate_b.pattern_id,
            "x_true": case.mix_fraction_true,
        }
        true_pair_key = tuple(sorted((args.index_a, args.index_b)))
        pred_pair_key = tuple(sorted((result.winner.index_a, result.winner.index_b)))
        payload["pair_match"] = bool(true_pair_key == pred_pair_key)
        same_order = bool(result.winner.index_a == args.index_a and result.winner.index_b == args.index_b)
        x_hat_in_true_order = float(result.winner.x_hat if same_order else 1.0 - result.winner.x_hat)
        payload["x_hat_in_true_order"] = x_hat_in_true_order
        payload["x_abs_error"] = abs(x_hat_in_true_order - float(case.mix_fraction_true))
        payload["images"] = {
            "a_noisy": "synthetic/a_noisy.png",
            "b_noisy": "synthetic/b_noisy.png",
            "c_synthetic": "synthetic/c_synthetic.png",
            "winner_a": "reconstructions/winner_a.png",
            "winner_b": "reconstructions/winner_b.png",
            "winner_c_hat": "reconstructions/winner_c_hat.png",
            "winner_residual": "reconstructions/winner_residual_abs.png",
        }
        payload["noise"] = {
            "gaussian_enabled": noise.gaussian_enabled,
            "gaussian_std": noise.gaussian_std,
            "salt_pepper_enabled": noise.salt_pepper_enabled,
            "salt_pepper_amount": noise.salt_pepper_amount,
            "salt_vs_pepper": noise.salt_vs_pepper,
            "rotation_enabled": noise.rotation_enabled,
            "rotation_max_deg": noise.rotation_max_deg,
            "angle_a_deg": case.angle_a_deg,
            "angle_b_deg": case.angle_b_deg,
        }
        payload["candidates"] = [
            {"index": idx, "id": record.pattern_id, "path": str(record.path)}
            for idx, record in enumerate(candidates)
        ]

        (out_dir / "demo_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _write_topk_csv(out_dir / "top_k_pairs.csv", result)
        _write_html_report(out_dir / "report" / "index.html", payload)

        summary = {
            "true_a": case.candidate_a.pattern_id,
            "true_b": case.candidate_b.pattern_id,
            "pred_a": result.winner.id_a,
            "pred_b": result.winner.id_b,
            "pair_match": payload["pair_match"],
            "x_true": case.mix_fraction_true,
            "x_hat": result.winner.x_hat,
            "x_hat_in_true_order": payload["x_hat_in_true_order"],
            "x_abs_error": payload["x_abs_error"],
            "primary_metric": result.primary_metric,
            "primary_score": result.winner.primary_score,
            "total_pairs": result.total_pairs,
            "runtime_s": result.runtime_s,
        }
        logger.info(
            "Demo complete: true=(%s,%s) predicted=(%s,%s) pair_match=%s x_true=%.4f x_hat=%.4f",
            case.candidate_a.pattern_id,
            case.candidate_b.pattern_id,
            result.winner.id_a,
            result.winner.id_b,
            payload["pair_match"],
            case.mix_fraction_true,
            result.winner.x_hat,
        )
    except Exception as exc:
        manifest["failures"].append({"error": str(exc)})
        logger.exception("Deterministic pair demo failed")
        raise
    finally:
        wall_time = time.perf_counter() - start
        manifest["summary"] = summary
        manifest["timings"] = {"wall_time_s": wall_time}
        write_manifest(out_dir, manifest)
        status = "failed" if manifest["failures"] else "completed"
        update_progress_report(
            out_dir,
            {
                "status": status,
                "progress": {"current": 1, "total": 1, "percent": 100.0, "eta_s": 0.0},
                "summary": summary,
            },
        )
        if manifest["failures"]:
            (out_dir / "error_report.json").write_text(json.dumps(manifest["failures"], indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
