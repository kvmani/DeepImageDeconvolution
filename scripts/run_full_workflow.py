"""Run the full workflow: prepare inputs -> generate synthetic data -> train."""
from __future__ import annotations

import argparse
import sys
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
from src.utils.logging import resolve_log_level, setup_logging


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

    try:
        if prepare_enabled:
            logger.info("Stage: prepare experimental data.")
            prepare_manifest = prepare_experimental_dataset(prepare_cfg, logger)
            prepared_dir = Path(prepare_manifest.get("output_dir", prepare_cfg.get("output_dir", "")))
        else:
            prepared_dir = Path(prepare_cfg.get("output_dir", ""))
            logger.info("Stage: prepare skipped.")

        if generate_enabled:
            logger.info("Stage: generate synthetic dataset.")
            generate_data_cfg = generate_config.setdefault("data", {})
            if "data.input_dir" not in touched["generate"]:
                if prepare_enabled:
                    generate_data_cfg["input_dir"] = str(prepared_dir)
            if "output_dir_override" in generate_cfg:
                generate_data_cfg["output_dir"] = str(generate_cfg["output_dir_override"])

            input_dir = Path(generate_data_cfg.get("input_dir", ""))
            _ensure_dir_exists(input_dir, "Generate input")
            collect_image_paths(input_dir, recursive=bool(generate_data_cfg.get("input_recursive", False)))

            generate_summary = generate_synthetic_dataset(generate_config)
            synthetic_dir = Path(generate_summary.get("output_dir", generate_data_cfg.get("output_dir", "")))
        else:
            synthetic_dir = Path(generate_config.get("data", {}).get("output_dir", ""))
            logger.info("Stage: generate skipped.")

        if train_enabled:
            logger.info("Stage: train model.")
            train_data_cfg = train_config.setdefault("data", {})
            if "data.root_dir" not in touched["train"]:
                if generate_enabled:
                    train_data_cfg["root_dir"] = str(synthetic_dir)
            if "out_dir_override" in train_cfg:
                train_config.setdefault("output", {})["out_dir"] = str(train_cfg["out_dir_override"])

            root_dir = Path(train_data_cfg.get("root_dir", ""))
            _ensure_dir_exists(root_dir, "Training data root")
            _assert_synthetic_structure(root_dir)
            train_model(train_config)
        else:
            logger.info("Stage: train skipped.")

        logger.info("Workflow complete.")
    except Exception as exc:
        logger.error("Workflow failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
