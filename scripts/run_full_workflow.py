"""Run the full workflow: prepare inputs -> generate synthetic data -> train."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.generation.dataset import generate_synthetic_dataset
from src.preprocessing.prepare import prepare_experimental_dataset
from src.training.train import train_model
from src.utils.config import deep_update, load_config
from src.utils.io import collect_image_paths
from src.utils.logging import collect_environment, resolve_log_level, setup_logging, write_manifest


DEFAULT_CONFIG = REPO_ROOT / "configs/workflow_default.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end workflow: prepare inputs, generate synthetic dataset, "
            "and train the model."
        )
    )
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config.")
    parser.add_argument("--input-dir", type=str, default=None, help="Raw input image folder.")
    parser.add_argument("--processed-dir", type=str, default=None, help="Prepared output folder.")
    parser.add_argument("--synthetic-dir", type=str, default=None, help="Synthetic dataset output folder.")
    parser.add_argument("--train-out-dir", type=str, default=None, help="Training output folder.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for all stages.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="scope.key=value",
        help=(
            "Repeatable override scoped to prepare/generate/train configs. "
            "Example: --set generate.data.num_samples=2000 --set train.train.epochs=50"
        ),
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
    return parser.parse_args()


def set_by_path(target: Dict[str, Any], path: str, value: Any) -> None:
    """Set a nested dict value by dot path, creating intermediate dicts."""
    parts = path.split(".")
    cursor = target
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor.get(part), dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def parse_scoped_overrides(values: List[str]) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Set[str]]]:
    overrides: Dict[str, Dict[str, Any]] = {"prepare": {}, "generate": {}, "train": {}}
    touched: Dict[str, Set[str]] = {"prepare": set(), "generate": set(), "train": set()}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"Invalid --set override '{raw}'. Expected scope.key=value.")
        key, value_str = raw.split("=", 1)
        if "." not in key:
            raise ValueError(f"Invalid --set key '{key}'. Use scope.key like generate.data.num_samples.")
        scope, path = key.split(".", 1)
        if scope not in overrides:
            raise ValueError(f"Unknown override scope '{scope}'. Use prepare, generate, or train.")
        try:
            value = yaml.safe_load(value_str)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML value for --set {key}: {exc}") from exc
        set_by_path(overrides[scope], path, value)
        touched[scope].add(path)
    return overrides, touched


def _resolve_config_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def _ensure_dir_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} directory not found: {path}")


def _assert_synthetic_structure(root_dir: Path) -> None:
    for sub in ("A", "B", "C"):
        if not (root_dir / sub).exists():
            raise FileNotFoundError(
                f"Synthetic dataset missing required folder '{sub}' under {root_dir}."
            )


def main() -> None:
    args = parse_args()
    log_level = resolve_log_level(args.log_level, debug=args.debug, quiet=args.quiet)
    log_file = Path(args.log_file) if args.log_file else None
    logger = setup_logging("run_full_workflow", level=log_level, log_file=log_file)

    workflow_start = time.perf_counter()
    config_path = Path(args.config) if args.config else DEFAULT_CONFIG
    if not config_path.exists():
        raise FileNotFoundError(f"Workflow config not found: {config_path}")
    workflow_cfg = load_config(config_path)

    overrides, touched = parse_scoped_overrides(args.set)

    if args.input_dir:
        workflow_cfg.setdefault("prepare", {})["input_dir"] = args.input_dir
    if args.processed_dir:
        workflow_cfg.setdefault("prepare", {})["output_dir"] = args.processed_dir
    if args.synthetic_dir:
        workflow_cfg.setdefault("generate", {})["output_dir_override"] = args.synthetic_dir
    if args.train_out_dir:
        workflow_cfg.setdefault("train", {})["out_dir_override"] = args.train_out_dir

    if args.debug:
        workflow_cfg.setdefault("prepare", {})["debug"] = True

    prepare_cfg = deep_update(workflow_cfg.get("prepare", {}), overrides["prepare"])
    generate_cfg = workflow_cfg.get("generate", {})
    train_cfg = workflow_cfg.get("train", {})

    generate_config_path = _resolve_config_path(generate_cfg.get("config", "configs/default.yaml"))
    train_config_path = _resolve_config_path(train_cfg.get("config", "configs/train_default.yaml"))

    generate_config = load_config(generate_config_path)
    train_config = load_config(train_config_path)

    if args.debug:
        generate_config.setdefault("debug", {})["enabled"] = True
        train_config.setdefault("debug", {})["enabled"] = True

    generate_config = deep_update(generate_config, overrides["generate"])
    train_config = deep_update(train_config, overrides["train"])

    prepare_enabled = bool(prepare_cfg.get("enabled", True))
    generate_enabled = bool(generate_cfg.get("enabled", True))
    train_enabled = bool(train_cfg.get("enabled", True))

    overall: Dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": workflow_cfg,
        "environment": collect_environment(),
        "stages": {},
        "warnings": [],
    }

    try:
        if prepare_enabled:
            logger.info("Stage: prepare experimental data.")
            stage_start = time.perf_counter()
            prepare_manifest = prepare_experimental_dataset(prepare_cfg, logger)
            stage_time = time.perf_counter() - stage_start
            prepared_dir = Path(prepare_manifest.get("output_dir", prepare_cfg.get("output_dir", "")))
            counts = prepare_manifest.get("counts", {})
            failures = prepare_manifest.get("failures", [])
            output_summary = prepare_manifest.get("output_summary", {})
            logger.info(
                "Prepare summary: attempted=%s | succeeded=%s | failed=%s | skipped=%s | outputs=%s | time=%.2fs",
                counts.get("attempted"),
                counts.get("succeeded"),
                counts.get("failed"),
                counts.get("skipped"),
                counts.get("outputs"),
                stage_time,
            )
            logger.info(
                "Prepare outputs: dir=%s | sizes=%s -> %s | dtypes=%s",
                prepared_dir,
                output_summary.get("min_size"),
                output_summary.get("max_size"),
                output_summary.get("sample_dtypes"),
            )
            if failures:
                logger.warning("Prepare reported %d failures (see error_report.json).", len(failures))
                overall["warnings"].append(f"prepare_failures={len(failures)}")
            overall["stages"]["prepare"] = {
                "duration_s": stage_time,
                "manifest": prepare_manifest,
                "output_dir": str(prepared_dir),
            }
        else:
            prepared_dir = Path(prepare_cfg.get("output_dir", ""))
            logger.info("Stage: prepare skipped.")

        if generate_enabled:
            logger.info("Stage: generate synthetic dataset.")
            stage_start = time.perf_counter()
            generate_data_cfg = generate_config.setdefault("data", {})
            if "data.input_dir" not in touched["generate"]:
                if prepare_enabled:
                    generate_data_cfg["input_dir"] = str(prepared_dir)
            if "output_dir_override" in generate_cfg:
                generate_data_cfg["output_dir"] = str(generate_cfg["output_dir_override"])

            input_dir = Path(generate_data_cfg.get("input_dir", ""))
            _ensure_dir_exists(input_dir, "Generate input")
            input_paths = collect_image_paths(
                input_dir, recursive=bool(generate_data_cfg.get("input_recursive", False))
            )
            logger.info("Generate inputs: %d images found in %s.", len(input_paths), input_dir)

            generate_summary = generate_synthetic_dataset(generate_config)
            stage_time = time.perf_counter() - stage_start
            synthetic_dir = Path(generate_summary.get("output_dir", generate_data_cfg.get("output_dir", "")))
            logger.info(
                "Generate summary: samples=%s | input_images=%s | output_dir=%s | time=%.2fs",
                generate_summary.get("samples"),
                generate_summary.get("input_images"),
                synthetic_dir,
                stage_time,
            )
            overall["stages"]["generate"] = {
                "duration_s": stage_time,
                "summary": generate_summary,
                "output_dir": str(synthetic_dir),
            }
        else:
            synthetic_dir = Path(generate_config.get("data", {}).get("output_dir", ""))
            logger.info("Stage: generate skipped.")

        if train_enabled:
            logger.info("Stage: train model.")
            stage_start = time.perf_counter()
            train_data_cfg = train_config.setdefault("data", {})
            if "data.root_dir" not in touched["train"]:
                if generate_enabled:
                    train_data_cfg["root_dir"] = str(synthetic_dir)
            if "out_dir_override" in train_cfg:
                train_config.setdefault("output", {})["out_dir"] = str(train_cfg["out_dir_override"])

            root_dir = Path(train_data_cfg.get("root_dir", ""))
            _ensure_dir_exists(root_dir, "Training data root")
            _assert_synthetic_structure(root_dir)
            train_summary = train_model(train_config)
            stage_time = time.perf_counter() - stage_start
            logger.info(
                "Train summary: epochs=%s | best_val=%s | train_samples=%s | val_samples=%s | output_dir=%s | time=%.2fs",
                train_summary.get("epochs"),
                train_summary.get("best_val"),
                train_summary.get("train_samples"),
                train_summary.get("val_samples"),
                train_summary.get("output_dir"),
                stage_time,
            )
            overall["stages"]["train"] = {
                "duration_s": stage_time,
                "summary": train_summary,
                "output_dir": str(train_summary.get("output_dir", "")),
            }
        else:
            logger.info("Stage: train skipped.")

        total_time = time.perf_counter() - workflow_start
        overall["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        overall["duration_s"] = total_time
        logger.info("Workflow complete in %.2fs.", total_time)

        output_dir = None
        if train_enabled and overall["stages"].get("train", {}).get("output_dir"):
            output_dir = Path(overall["stages"]["train"]["output_dir"])
        elif generate_enabled and overall["stages"].get("generate", {}).get("output_dir"):
            output_dir = Path(overall["stages"]["generate"]["output_dir"])
        elif prepare_enabled and overall["stages"].get("prepare", {}).get("output_dir"):
            output_dir = Path(overall["stages"]["prepare"]["output_dir"])

        if output_dir is not None:
            write_manifest(output_dir, overall, filename="workflow_manifest.json")
            logger.info("Workflow manifest written to %s", output_dir / "workflow_manifest.json")
            report_path = output_dir / "workflow_report.txt"
            _write_human_report(report_path, overall)
            logger.info("Workflow report written to %s", report_path)
    except Exception as exc:
        logger.error("Workflow failed: %s", exc)
        raise


def _write_human_report(path: Path, summary: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("Full Workflow Report")
    lines.append("=" * 80)
    lines.append(f"Started: {summary.get('started_at')}")
    lines.append(f"Completed: {summary.get('completed_at')}")
    duration = summary.get("duration_s")
    if duration is not None:
        lines.append(f"Total duration (s): {duration:.2f}")
    lines.append("")

    env = summary.get("environment", {})
    lines.append("Environment")
    lines.append("-" * 80)
    lines.append(f"Python: {env.get('python')}")
    lines.append(f"Platform: {env.get('platform')}")
    lines.append(f"CWD: {env.get('cwd')}")
    lines.append("")

    warnings = summary.get("warnings", [])
    if warnings:
        lines.append("Warnings")
        lines.append("-" * 80)
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")

    stages = summary.get("stages", {})
    for stage_name in ("prepare", "generate", "train"):
        if stage_name not in stages:
            continue
        stage = stages[stage_name]
        lines.append(f"Stage: {stage_name}")
        lines.append("-" * 80)
        lines.append(f"Duration (s): {stage.get('duration_s', 0):.2f}")
        lines.append(f"Output dir: {stage.get('output_dir', 'n/a')}")

        if stage_name == "prepare":
            manifest = stage.get("manifest", {})
            counts = manifest.get("counts", {})
            lines.append(
                "Counts: attempted={attempted} | succeeded={succeeded} | failed={failed} | "
                "skipped={skipped} | outputs={outputs}".format(
                    attempted=counts.get("attempted"),
                    succeeded=counts.get("succeeded"),
                    failed=counts.get("failed"),
                    skipped=counts.get("skipped"),
                    outputs=counts.get("outputs"),
                )
            )
            output_summary = manifest.get("output_summary", {})
            lines.append(
                "Output sizes: min={min_size} max={max_size}".format(
                    min_size=output_summary.get("min_size"),
                    max_size=output_summary.get("max_size"),
                )
            )
            failures = manifest.get("failures", [])
            lines.append(f"Failures: {len(failures)}")

        if stage_name == "generate":
            gen = stage.get("summary", {})
            lines.append(f"Input images: {gen.get('input_images')}")
            lines.append(f"Samples generated: {gen.get('samples')}")

        if stage_name == "train":
            train = stage.get("summary", {})
            lines.append(f"Epochs: {train.get('epochs')}")
            lines.append(f"Best val: {train.get('best_val')}")
            lines.append(
                "Samples: train={train_samples} | val={val_samples}".format(
                    train_samples=train.get("train_samples"),
                    val_samples=train.get("val_samples"),
                )
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
