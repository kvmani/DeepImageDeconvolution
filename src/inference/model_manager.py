"""Model caching utilities for inference."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import threading
from typing import Any, Callable, Dict, Optional

import torch

from src.models import build_model
from src.utils.logging import get_logger


def _hash_config(config: Dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
    """Load a checkpoint into a model."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])


@dataclass(frozen=True)
class ModelCacheKey:
    """Unique cache key for a model instance."""

    checkpoint_path: Path
    device: str
    model_signature: str


class ModelManager:
    """Cache and reuse inference models across runs."""

    def __init__(
        self,
        model_factory: Optional[Callable[[Dict[str, Any]], torch.nn.Module]] = None,
    ) -> None:
        self._model_factory = model_factory or build_model
        self._model: Optional[torch.nn.Module] = None
        self._cache_key: Optional[ModelCacheKey] = None
        self._lock = threading.Lock()
        self._logger = get_logger(__name__)

    def get_model(
        self,
        model_cfg: Dict[str, Any],
        checkpoint_path: Path,
        device: torch.device,
    ) -> torch.nn.Module:
        """Return a cached model or load a new one when parameters change."""
        checkpoint_path = checkpoint_path.resolve()
        signature = _hash_config(model_cfg)
        cache_key = ModelCacheKey(checkpoint_path, str(device), signature)

        with self._lock:
            if self._model is not None and self._cache_key == cache_key:
                return self._model

            self._logger.info(
                "Loading model (checkpoint=%s, device=%s)", checkpoint_path, device
            )
            model = self._model_factory(model_cfg)
            load_checkpoint(model, checkpoint_path)
            model = model.to(device)
            model.eval()
            self._model = model
            self._cache_key = cache_key
            return model

    def clear(self) -> None:
        """Clear the cached model."""
        with self._lock:
            self._model = None
            self._cache_key = None
