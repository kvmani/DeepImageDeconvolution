"""Model factory and registry."""
from __future__ import annotations

from typing import Any, Dict

from src.models.dual_unet import UNetConfig, UNetDual


def build_model(cfg: Dict[str, Any]) -> UNetDual:
    """Build a model from configuration.

    Parameters
    ----------
    cfg:
        Configuration dictionary with at least a `name` key.

    Returns
    -------
    torch.nn.Module
        Instantiated model.
    """
    name = str(cfg.get("name", "unet_dual")).lower()
    if name in {"unet_dual", "dual_unet"}:
        model_cfg = UNetConfig.from_dict(cfg)
        return UNetDual(model_cfg)
    raise ValueError(f"Unknown model name: {name}")


__all__ = ["build_model", "UNetConfig", "UNetDual"]
