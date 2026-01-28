"""Entry point for the pattern mixer GUI."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import logging

from PySide6 import QtWidgets

from src.utils.logging import add_file_handler, get_logger, resolve_log_level

from .gui.logging_handler import LogPanel, QtLogEmitter, QtLogHandler
from .gui.main_window import PatternMixerWindow


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kikuchi Pattern Mixing Playground GUI")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    parser.add_argument("--log-file", type=Path, default=None, help="Optional log file.")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging verbosity.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    level = resolve_log_level(args.log_level, debug=args.debug, quiet=args.quiet)
    logger = get_logger("pattern_mixer_gui", level=level)

    if args.log_file is not None:
        add_file_handler(args.log_file, level=level)

    app = QtWidgets.QApplication(sys.argv)
    window = PatternMixerWindow(debug=args.debug)

    emitter = QtLogEmitter()
    handler = QtLogHandler(emitter)
    handler.setLevel(level)

    root_logger = logging.getLogger()
    formatter = root_logger.handlers[0].formatter if root_logger.handlers else None
    if formatter is not None:
        handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    if isinstance(window._log_panel, LogPanel):
        emitter.message_emitted.connect(window._log_panel.append_line)

    window.show()
    logger.info("Pattern mixer GUI started.")
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
