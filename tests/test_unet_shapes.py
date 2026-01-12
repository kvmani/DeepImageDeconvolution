import pytest

pytest.importorskip("torch")

import torch

from src.models.dual_unet import UNetConfig, UNetDual


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
    out_a, out_b, x_hat = model(x)
    assert out_a.shape == x.shape
    assert out_b.shape == x.shape
    assert out_a.dtype == x.dtype
    assert out_b.dtype == x.dtype
    assert x_hat.shape == (2, 1)
    assert torch.all(x_hat >= 0.0)
    assert torch.all(x_hat <= 1.0)
    recon = x_hat.view(-1, 1, 1, 1) * out_a + (1.0 - x_hat.view(-1, 1, 1, 1)) * out_b
    assert recon.shape == x.shape
