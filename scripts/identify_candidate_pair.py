"""CLI to identify the most likely candidate pair for a mixed pattern."""
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
    identify_pair_from_candidates,
    sample_random_candidates,
)
from src.deterministic_mixing_inversion.alignment import (  # noqa: E402
    RigidAlignment,
    apply_rigid_alignment,
    parse_alignment_settings,
)
from src.deterministic_mixing_inversion.io import (  # noqa: E402
    PatternRecord,
    load_pattern,
    load_pattern_records_from_paths,
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


def _parse_set_overrides(raw_overrides: list[str]) -> Dict[str, Any]:
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


def _ranked_pair_payload(result: IdentificationResult) -> Dict[str, Any]:
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
        "total_pairs": result.total_pairs,
        "runtime_s": result.runtime_s,
        "primary_metric": result.primary_metric,
    }


def _build_winner_reconstruction(
    mixed_image: np.ndarray,
    candidates: List[PatternRecord],
    result: IdentificationResult,
    inversion_cfg: Dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    preprocess_settings = parse_preprocess_settings(inversion_cfg.get("preprocess", {}))
    alignment_settings = parse_alignment_settings(inversion_cfg.get("alignment", {}))
    target_shape = mixed_image.shape
    mask = build_centered_mask(target_shape) if preprocess_settings.mask_enabled else None

    candidate_a = candidates[result.winner.index_a]
    candidate_b = candidates[result.winner.index_b]
    a_matched = match_shape_to_target(
        candidate_a.image,
        target_shape=target_shape,
        auto_crop_to_target=preprocess_settings.auto_crop_to_target,
    ).astype(np.float32)
    b_matched = match_shape_to_target(
        candidate_b.image,
        target_shape=target_shape,
        auto_crop_to_target=preprocess_settings.auto_crop_to_target,
    ).astype(np.float32)
    if mask is not None:
        a_matched = a_matched.copy()
        b_matched = b_matched.copy()
        a_matched[~mask] = 0.0
        b_matched[~mask] = 0.0

    c_hat = (result.winner.x_hat * a_matched + (1.0 - result.winner.x_hat) * b_matched).astype(np.float32)
    alignment = result.winner.alignment
    rigid = RigidAlignment(
        angle_deg=float(alignment.get("angle_deg", 0.0)),
        shift_y=float(alignment.get("shift_y", 0.0)),
        shift_x=float(alignment.get("shift_x", 0.0)),
        score=0.0,
    )
    c_hat_aligned = apply_rigid_alignment(
        c_hat,
        rigid,
        interpolation_order=alignment_settings.interpolation_order,
        mask=mask,
    ).astype(np.float32)
    residual = np.abs(c_hat_aligned - mixed_image).astype(np.float32)
    return a_matched, b_matched, c_hat_aligned, residual


def _write_top_k_csv(path: Path, result: IdentificationResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
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
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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


def _write_html_report(
    path: Path,
    payload: Dict[str, Any],
) -> None:
    winner = payload.get("winner", {})
    top_k = payload.get("top_k", [])
    images = payload.get("images", {})
    html = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Candidate Pair Identification</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:20px;}",
        ".row{display:flex;gap:14px;flex-wrap:wrap;}",
        ".card{border:1px solid #ddd;padding:10px;border-radius:8px;}",
        ".img{width:260px;height:auto;border:1px solid #eee;}",
        "table{border-collapse:collapse;width:100%;margin-top:16px;}",
        "th,td{border:1px solid #ddd;padding:6px;text-align:left;}",
        "th{background:#f5f5f5;}",
        "</style></head><body>",
        "<h1>Deterministic Candidate Pair Identification</h1>",
        (
            f"<p>Winner: {winner.get('id_a')} + {winner.get('id_b')} | "
            f"x_hat={winner.get('x_hat')} | "
            f"{payload.get('primary_metric')}={winner.get('primary_score')}</p>"
        ),
        "<div class='row'>",
        f"<div class='card'><h3>Mixed C</h3><img class='img' src='{images.get('mixed', '')}'/></div>",
        f"<div class='card'><h3>Winner A</h3><img class='img' src='{images.get('winner_a', '')}'/></div>",
        f"<div class='card'><h3>Winner B</h3><img class='img' src='{images.get('winner_b', '')}'/></div>",
        f"<div class='card'><h3>C_hat</h3><img class='img' src='{images.get('winner_c_hat', '')}'/></div>",
        f"<div class='card'><h3>|Residual|</h3><img class='img' src='{images.get('residual', '')}'/></div>",
        "</div>",
        "<table><tr><th>Rank</th><th>A</th><th>B</th><th>x_hat</th><th>Primary</th><th>L2</th></tr>",
    ]
    for row in top_k:
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
    parser = argparse.ArgumentParser(description="Identify the best candidate pair for a mixed pattern.")
    parser.add_argument("--mixed-path", type=str, required=True, help="Path to the mixed pattern C.")
    parser.add_argument(
        "--candidate-dir",
        type=str,
        default="data/raw/Double Pattern Data/Good Pattern",
        help="Directory containing candidate pure patterns.",
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=10,
        help="Number of random candidates to sample when manifest is not provided.",
    )
    parser.add_argument(
        "--candidate-manifest",
        type=str,
        default=None,
        help="Optional JSON file containing candidate_paths.",
    )
    parser.add_argument("--sample-seed", type=int, default=7, help="Optional seed for random sampling.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top results to keep.")
    parser.add_argument(
        "--coarse-top-m",
        type=int,
        default=20,
        help="Two-stage search: number of coarse pairs refined in stage-2.",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/gui_pair_demo",
        help="Output directory.",
    )
    parser.add_argument("--run-tag", type=str, default=None, help="Append timestamped run tag to output.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="key=value",
        help="Repeatable dot-path override, e.g. --set deterministic_inversion.search.grid_steps=[0.1,0.02].",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug defaults.")
    parser.add_argument("--log-level", type=str, default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    parser.add_argument("--log-file", type=str, default=None, help="Optional log file path.")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging verbosity.")
    return parser.parse_args()


def _load_candidate_records(args: argparse.Namespace, logger: logging.Logger) -> List[PatternRecord]:
    if args.candidate_manifest:
        manifest_path = Path(args.candidate_manifest)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_paths = payload.get("candidate_paths", [])
        if not isinstance(raw_paths, list) or not raw_paths:
            raise ValueError("candidate_manifest must contain non-empty 'candidate_paths'.")
        candidate_paths = [Path(str(path)) for path in raw_paths]
        records = load_pattern_records_from_paths(candidate_paths)
        logger.info("Loaded %d candidates from manifest: %s", len(records), manifest_path)
        return records

    records = sample_random_candidates(
        candidate_dir=Path(args.candidate_dir),
        sample_count=args.candidate_count,
        seed=args.sample_seed,
        recursive=False,
        logger=logger,
    )
    return records


def main() -> None:
    args = parse_args()
    log_level = resolve_log_level(args.log_level, debug=args.debug, quiet=args.quiet)
    logger = setup_logging(
        "identify_candidate_pair",
        level=log_level,
        log_file=Path(args.log_file) if args.log_file else None,
        run_id=args.run_tag,
    )
    logging.getLogger("PIL").setLevel(logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.INFO)

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
        inversion_cfg["pair_search"]["coarse_top_m"] = int(max(args.coarse_top_m, 2))
        inversion_cfg["pair_search"]["two_stage_enabled"] = True

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

    report_payload: Dict[str, Any] = {
        "status": "running",
        "progress": {"current": 0, "total": 1, "percent": 0.0, "eta_s": None},
        "summary": {},
    }
    update_progress_report(out_dir, report_payload)

    try:
        mixed_record = load_pattern(Path(args.mixed_path))
        candidates = _load_candidate_records(args, logger)
        candidate_manifest = {
            "candidate_paths": [str(record.path) for record in candidates],
            "candidate_ids": [record.pattern_id for record in candidates],
        }
        (out_dir / "candidate_manifest.json").write_text(json.dumps(candidate_manifest, indent=2), encoding="utf-8")

        def on_progress(processed: int, total: int, eta_s: float, message: str) -> None:
            logger.info("%s | ETA %.2fs", message, eta_s)
            progress_percent = 100.0 * (processed / max(total, 1))
            update_progress_report(
                out_dir,
                {
                    "status": "running",
                    "progress": {
                        "current": processed,
                        "total": total,
                        "percent": progress_percent,
                        "eta_s": eta_s,
                    },
                    "summary": {},
                },
            )

        result = identify_pair_from_candidates(
            candidates=candidates,
            mixed_image=mixed_record.image,
            inversion_cfg=inversion_cfg,
            top_k=args.top_k,
            progress_callback=on_progress,
            logger=logger,
        )
        payload = _ranked_pair_payload(result)
        payload["mixed"] = {"id": mixed_record.pattern_id, "path": str(mixed_record.path)}
        payload["candidates"] = [
            {"index": idx, "id": record.pattern_id, "path": str(record.path)}
            for idx, record in enumerate(candidates)
        ]

        recon_dir = out_dir / "reconstructions"
        recon_dir.mkdir(parents=True, exist_ok=True)
        a_img, b_img, c_hat_img, residual_img = _build_winner_reconstruction(
            mixed_image=mixed_record.image,
            candidates=candidates,
            result=result,
            inversion_cfg=inversion_cfg,
        )
        write_image_16bit(recon_dir / "mixed_c.png", mixed_record.image)
        write_image_16bit(recon_dir / "winner_a.png", a_img)
        write_image_16bit(recon_dir / "winner_b.png", b_img)
        write_image_16bit(recon_dir / "winner_c_hat.png", c_hat_img)
        write_image_16bit(recon_dir / "winner_residual_abs.png", np.clip(residual_img, 0.0, 1.0))

        payload["images"] = {
            "mixed": "reconstructions/mixed_c.png",
            "winner_a": "reconstructions/winner_a.png",
            "winner_b": "reconstructions/winner_b.png",
            "winner_c_hat": "reconstructions/winner_c_hat.png",
            "residual": "reconstructions/winner_residual_abs.png",
        }

        (out_dir / "identification_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _write_top_k_csv(out_dir / "top_k_pairs.csv", result)
        _write_html_report(out_dir / "report" / "index.html", payload)

        summary = {
            "mixed_id": mixed_record.pattern_id,
            "winner_a": result.winner.id_a,
            "winner_b": result.winner.id_b,
            "x_hat": result.winner.x_hat,
            "primary_metric": result.primary_metric,
            "primary_score": result.winner.primary_score,
            "total_pairs": result.total_pairs,
            "runtime_s": result.runtime_s,
        }
        logger.info(
            "Identification complete: winner=%s + %s | x_hat=%.4f | %s=%.6f | pairs=%d",
            result.winner.id_a,
            result.winner.id_b,
            result.winner.x_hat,
            result.primary_metric,
            result.winner.primary_score,
            result.total_pairs,
        )
    except Exception as exc:
        manifest["failures"].append({"error": str(exc)})
        logger.exception("Candidate identification failed")
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
