"""Logging utilities for the project."""
from __future__ import annotations

import logging
from pathlib import Path


_LOGGER_CONFIGURED = False


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger.

    Parameters
    ----------
    name:
        Logger name, typically ``__name__``.
    level:
        Logging level.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    global _LOGGER_CONFIGURED

    logger = logging.getLogger(name)

    if not _LOGGER_CONFIGURED:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
        _LOGGER_CONFIGURED = True

    logger.setLevel(level)
    return logger


def add_file_handler(log_path: Path, level: int = logging.INFO) -> None:
    """Add a file handler to the root logger.

    Parameters
    ----------
    log_path:
        Path to the log file.
    level:
        Logging level for the file handler.
    """
    root = logging.getLogger()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == str(log_path):
            return
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
