import pytest

pytest.importorskip("torch")

import torch

from src.models.unet_dual import UNetConfig, UNetDual


def test_unet_dual_output_shapes() -> None:
    cfg = UNetConfig(
        in_channels=1,
        out_channels=1,
        base_channels=8,
        depth=3,
        up_mode="transpose",
        use_batchnorm=False,
        dropout=0.0,
        init="kaiming",
    )
    model = UNetDual(cfg)
    x = torch.randn(2, 1, 64, 64)
    out_a, out_b = model(x)
    assert out_a.shape == x.shape
    assert out_b.shape == x.shape
