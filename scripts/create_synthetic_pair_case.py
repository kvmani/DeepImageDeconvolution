"""CLI to generate one synthetic noisy mixed-pattern case from candidate patterns."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.deterministic_mixing_inversion import (  # noqa: E402
    SyntheticNoiseConfig,
    build_synthetic_case,
    sample_random_candidates,
)
from src.utils.io import write_image_16bit  # noqa: E402
from src.utils.logging import (  # noqa: E402
    collect_environment,
    get_git_commit,
    resolve_log_level,
    setup_logging,
    write_manifest,
)
from src.utils.run import resolve_run_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create one synthetic noisy A/B -> C case.")
    parser.add_argument(
        "--candidate-dir",
        type=str,
        default="data/raw/Double Pattern Data/Good Pattern",
        help="Directory containing candidate patterns.",
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=10,
        help="Number of random candidates to sample from directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for reproducible candidate sampling and noise.",
    )
    parser.add_argument(
        "--index-a",
        type=int,
        required=True,
        help="Index of selected A in sampled candidate list.",
    )
    parser.add_argument(
        "--index-b",
        type=int,
        required=True,
        help="Index of selected B in sampled candidate list.",
    )
    parser.add_argument(
        "--x",
        type=float,
        required=True,
        help="Mixing fraction for A, in [0,1].",
    )
    parser.add_argument(
        "--rotation-max-deg",
        type=float,
        default=2.0,
        help="Max absolute random rotation angle for A/B noise.",
    )
    parser.add_argument(
        "--gaussian-std",
        type=float,
        default=0.01,
        help="Gaussian noise standard deviation.",
    )
    parser.add_argument(
        "--salt-pepper-amount",
        type=float,
        default=0.0,
        help="Salt-pepper pixel fraction in [0,1].",
    )
    parser.add_argument(
        "--salt-vs-pepper",
        type=float,
        default=0.5,
        help="Salt ratio in salt-pepper noise, in [0,1].",
    )
    parser.add_argument(
        "--disable-gaussian",
        action="store_true",
        help="Disable Gaussian noise.",
    )
    parser.add_argument(
        "--disable-salt-pepper",
        action="store_true",
        help="Disable salt-pepper noise.",
    )
    parser.add_argument(
        "--disable-rotation",
        action="store_true",
        help="Disable rotation noise.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/synthetic_pair_case",
        help="Output directory.",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Append timestamped run tag to output directory.",
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
        "--debug",
        action="store_true",
        help="Enable deterministic debug behavior.",
    )
    return parser.parse_args()


def _save_case_outputs(out_dir: Path, payload: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "case.json").write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    if args.debug and args.seed is None:
        args.seed = 7

    log_level = resolve_log_level(args.log_level, debug=args.debug, quiet=args.quiet)
    logger = setup_logging(
        "create_synthetic_pair_case",
        level=log_level,
        log_file=Path(args.log_file) if args.log_file else None,
        run_id=args.run_tag,
    )
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    logging.getLogger("PIL").setLevel(logging.INFO)

    out_dir = Path(args.out_dir)
    if args.run_tag:
        out_dir = resolve_run_dir(out_dir, args.run_tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": get_git_commit(REPO_ROOT),
        "args": vars(args),
        "environment": collect_environment(),
        "output_dir": str(out_dir),
        "failures": [],
    }
    start = time.perf_counter()

    summary: Dict[str, Any] = {}
    try:
        candidate_dir = Path(args.candidate_dir)
        candidates = sample_random_candidates(
            candidate_dir=candidate_dir,
            sample_count=args.candidate_count,
            seed=args.seed,
            recursive=False,
            logger=logger,
        )
        for idx, candidate in enumerate(candidates):
            logger.info("Candidate[%d] = %s", idx, candidate.pattern_id)

        noise = SyntheticNoiseConfig(
            gaussian_enabled=not args.disable_gaussian,
            gaussian_std=max(float(args.gaussian_std), 0.0),
            salt_pepper_enabled=not args.disable_salt_pepper,
            salt_pepper_amount=float(max(args.salt_pepper_amount, 0.0)),
            salt_vs_pepper=float(np.clip(args.salt_vs_pepper, 0.0, 1.0)),
            rotation_enabled=not args.disable_rotation,
            rotation_max_deg=float(max(args.rotation_max_deg, 0.0)),
        )
        case = build_synthetic_case(
            candidates=candidates,
            index_a=args.index_a,
            index_b=args.index_b,
            mix_fraction=float(np.clip(args.x, 0.0, 1.0)),
            noise=noise,
            seed=args.seed,
            mask_enabled=True,
        )

        write_image_16bit(out_dir / "A_noisy.png", case.pattern_a_noisy)
        write_image_16bit(out_dir / "B_noisy.png", case.pattern_b_noisy)
        write_image_16bit(out_dir / "C_synthetic.png", case.mixed_c)

        case_payload = {
            "candidate_dir": str(candidate_dir),
            "candidate_count": len(candidates),
            "sample_seed": args.seed,
            "candidate_manifest": [
                {"index": idx, "id": candidate.pattern_id, "path": str(candidate.path)}
                for idx, candidate in enumerate(candidates)
            ],
            "selection": {
                "index_a": args.index_a,
                "index_b": args.index_b,
                "id_a": case.candidate_a.pattern_id,
                "id_b": case.candidate_b.pattern_id,
                "x_true": case.mix_fraction_true,
            },
            "noise": {
                "gaussian_enabled": noise.gaussian_enabled,
                "gaussian_std": noise.gaussian_std,
                "salt_pepper_enabled": noise.salt_pepper_enabled,
                "salt_pepper_amount": noise.salt_pepper_amount,
                "salt_vs_pepper": noise.salt_vs_pepper,
                "rotation_enabled": noise.rotation_enabled,
                "rotation_max_deg": noise.rotation_max_deg,
                "angle_a_deg": case.angle_a_deg,
                "angle_b_deg": case.angle_b_deg,
            },
            "outputs": {
                "a_noisy": str((out_dir / "A_noisy.png")),
                "b_noisy": str((out_dir / "B_noisy.png")),
                "c_synthetic": str((out_dir / "C_synthetic.png")),
            },
        }
        _save_case_outputs(out_dir, case_payload)
        summary = {
            "candidate_count": len(candidates),
            "selected_a": case.candidate_a.pattern_id,
            "selected_b": case.candidate_b.pattern_id,
            "x_true": case.mix_fraction_true,
            "runtime_s": time.perf_counter() - start,
        }
        logger.info(
            "Synthetic case complete: A=%s B=%s x=%.4f",
            case.candidate_a.pattern_id,
            case.candidate_b.pattern_id,
            case.mix_fraction_true,
        )
    except Exception as exc:
        manifest["failures"].append({"error": str(exc)})
        logger.exception("Synthetic case generation failed")
        raise
    finally:
        manifest["summary"] = summary
        manifest["timings"] = {"wall_time_s": time.perf_counter() - start}
        write_manifest(out_dir, manifest)
        if manifest["failures"]:
            (out_dir / "error_report.json").write_text(json.dumps(manifest["failures"], indent=2))


if __name__ == "__main__":
    main()
