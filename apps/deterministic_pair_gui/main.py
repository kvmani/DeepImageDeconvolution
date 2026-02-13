"""Entry point for deterministic pair-identification GUI."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import logging

from src.utils.logging import add_file_handler, resolve_log_level, setup_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic candidate-pair identification GUI")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/deterministic_inversion/pair_demo_default.yaml"),
        help="Path to GUI defaults YAML.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    parser.add_argument("--log-file", type=Path, default=None, help="Optional log file path.")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging verbosity.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        from PySide6 import QtWidgets
    except ImportError as exc:
        raise RuntimeError(
            "PySide6 is required for deterministic_pair_gui. Install with: pip install PySide6"
        ) from exc

    from apps.pattern_mixer_gui.gui.logging_handler import LogPanel, QtLogEmitter, QtLogHandler
    from .gui.main_window import DeterministicPairWindow

    level = resolve_log_level(args.log_level, debug=args.debug, quiet=args.quiet)
    logger = setup_logging("deterministic_pair_gui", level=level, log_file=args.log_file, run_id=None)

    if args.log_file is not None:
        add_file_handler(args.log_file, level=level)

    app = QtWidgets.QApplication(sys.argv)
    window = DeterministicPairWindow(config_path=args.config, debug=args.debug)

    emitter = QtLogEmitter()
    handler = QtLogHandler(emitter)
    handler.setLevel(level)
    root_logger = logging.getLogger()
    formatter = root_logger.handlers[0].formatter if root_logger.handlers else None
    if formatter is not None:
        handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    if isinstance(window.log_panel, LogPanel):
        emitter.message_emitted.connect(window.log_panel.append_line)

    window.show()
    logger.info("Deterministic pair GUI started with config: %s", args.config)
    return app.exec()
