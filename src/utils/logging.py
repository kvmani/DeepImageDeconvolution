"""Logging utilities for the project."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
from PIL import Image
_LOGGER_CONFIGURED = False
_CONTEXT_FILTER: Optional["ScriptContextFilter"] = None


@dataclass
class ProgressSnapshot:
    """Progress snapshot for logging updates."""

    processed: int
    total: int
    elapsed_s: float
    avg_s_per_item: float
    eta_s: float
    percent: float


class StructuredFormatter(logging.Formatter):
    """Format log records with a structured CLI-friendly prefix."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        script_name = getattr(record, "script_name", record.name)
        run_id = getattr(record, "run_id", None)
        prefix = f"[{timestamp}][{record.levelname}][{script_name}]"
        if run_id:
            prefix = f"{prefix}[{run_id}]"
        message = record.getMessage()
        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
            return f"{prefix} {message}\n{exc_text}"
        return f"{prefix} {message}"


class ScriptContextFilter(logging.Filter):
    """Inject script-level context into log records."""

    def __init__(self, script_name: str, run_id: Optional[str]) -> None:
        super().__init__()
        self.script_name = script_name
        self.run_id = run_id

    def update(self, script_name: str, run_id: Optional[str]) -> None:
        self.script_name = script_name
        self.run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.script_name = self.script_name
        record.run_id = self.run_id
        return True


def _ensure_root_logger(level: int, script_name: str, run_id: Optional[str]) -> None:
    global _LOGGER_CONFIGURED, _CONTEXT_FILTER
    root = logging.getLogger()
    root.setLevel(level)

    if not _LOGGER_CONFIGURED:
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredFormatter())
        root.addHandler(handler)
        _LOGGER_CONFIGURED = True

    for handler in root.handlers:
        handler.setLevel(level)

    if _CONTEXT_FILTER is None:
        _CONTEXT_FILTER = ScriptContextFilter(script_name, run_id)
        root.addFilter(_CONTEXT_FILTER)
    else:
        _CONTEXT_FILTER.update(script_name, run_id)


def resolve_log_level(level: str | int | None, debug: bool = False, quiet: bool = False) -> int:
    """Resolve a logging level from CLI-style inputs."""
    if quiet:
        return logging.WARNING
    if debug:
        return logging.DEBUG
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        normalized = level.upper()
        return getattr(logging, normalized, logging.INFO)
    return logging.INFO


def setup_logging(
    script_name: str,
    level: int | str | None = logging.INFO,
    log_file: Optional[Path] = None,
    run_id: Optional[str] = None,
) -> logging.Logger:
    """Configure structured logging for CLI scripts."""
    resolved = resolve_log_level(level)
    _ensure_root_logger(resolved, script_name, run_id)
    logger = logging.getLogger(script_name)
    logger.setLevel(resolved)
    if log_file is not None:
        add_file_handler(log_file, level=resolved)
    return logger


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
    logger = logging.getLogger(name)

    if not _LOGGER_CONFIGURED:
        _ensure_root_logger(level, name, None)

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
    file_handler.setFormatter(StructuredFormatter())
    root.addHandler(file_handler)


def summarize_images(paths: Sequence[Path], sample_n: int = 20) -> Dict[str, Any]:
    """Summarize a collection of image paths."""
    extensions: Dict[str, int] = {}
    for path in paths:
        extensions[path.suffix.lower()] = extensions.get(path.suffix.lower(), 0) + 1

    sample_sizes = []
    sample_dtypes = []
    sample_modes = []
    errors = []

    for path in paths[:sample_n]:
        try:
            with Image.open(path) as img:
                sample_modes.append(img.mode)
                array = np.array(img)
                sample_dtypes.append(str(array.dtype))
                sample_sizes.append((img.width, img.height))
        except Exception as exc:  # pragma: no cover - defensive for corrupt files
            errors.append({"path": str(path), "error": str(exc)})

    min_size = None
    max_size = None
    if sample_sizes:
        widths, heights = zip(*sample_sizes)
        min_size = {"width": int(min(widths)), "height": int(min(heights))}
        max_size = {"width": int(max(widths)), "height": int(max(heights))}

    return {
        "total": len(paths),
        "extensions": extensions,
        "sample_count": len(sample_sizes),
        "sample_sizes": sample_sizes,
        "min_size": min_size,
        "max_size": max_size,
        "sample_dtypes": sample_dtypes,
        "sample_modes": sample_modes,
        "errors": errors,
    }


def _format_duration(seconds: float) -> str:
    seconds = max(seconds, 0.0)
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


class ProgressLogger:
    """Log periodic progress updates with ETA estimates."""

    def __init__(
        self,
        total: int,
        logger: logging.Logger,
        every: int = 100,
        unit: str = "items",
    ) -> None:
        self.total = total
        self.logger = logger
        self.every = max(1, every)
        self.unit = unit
        self.start = time.perf_counter()
        self.last_log = self.start
        self.count = 0

    def update(self, increment: int = 1) -> ProgressSnapshot:
        self.count += increment
        now = time.perf_counter()
        elapsed = now - self.start
        avg = elapsed / max(self.count, 1)
        remaining = max(self.total - self.count, 0)
        eta = remaining * avg
        percent = (self.count / self.total * 100.0) if self.total else 100.0

        if (
            self.count == self.total
            or self.count % self.every == 0
            or (now - self.last_log) > 30
        ):
            self.logger.info(
                "Processed %d/%d (%.1f%%) | avg %.4f s/%s | ETA %s",
                self.count,
                self.total,
                percent,
                avg,
                self.unit,
                _format_duration(eta),
            )
            self.last_log = now

        return ProgressSnapshot(
            processed=self.count,
            total=self.total,
            elapsed_s=elapsed,
            avg_s_per_item=avg,
            eta_s=eta,
            percent=percent,
        )


@contextmanager
def log_timer(logger: logging.Logger, label: str, level: int = logging.INFO) -> Iterable[None]:
    """Log the wall-time duration of a code block."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.log(level, "%s completed in %s", label, _format_duration(elapsed))


def write_manifest(output_dir: Path, manifest: Dict[str, Any], filename: str = "manifest.json") -> Path:
    """Write a JSON manifest to the specified directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / filename
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    return manifest_path


def get_git_commit(root: Optional[Path] = None) -> Optional[str]:
    """Return the current git commit hash if available."""
    root = root or Path.cwd()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def collect_environment() -> Dict[str, Any]:
    """Collect basic environment metadata."""
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "cwd": os.getcwd(),
    }
