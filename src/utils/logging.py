"""Logging utilities for the project."""
from __future__ import annotations

import logging
from typing import Optional


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
