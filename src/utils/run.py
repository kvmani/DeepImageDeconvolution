"""Utilities for standardized run directory naming."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re


def _sanitize_tag(tag: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", tag).strip("-")
    return cleaned


def resolve_run_dir(base_dir: Path, run_tag: str | None) -> Path:
    """Return a run directory with timestamp + short tag appended to base."""
    if not run_tag:
        return base_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_tag = _sanitize_tag(run_tag)
    suffix = f"{timestamp}_{safe_tag}" if safe_tag else timestamp
    return base_dir / suffix
