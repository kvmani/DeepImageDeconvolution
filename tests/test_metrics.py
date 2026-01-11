import pytest

pytest.importorskip("torch")

import torch

from src.utils.metrics import mse, psnr, ssim


def test_psnr_higher_for_identical() -> None:
    x = torch.rand(1, 1, 16, 16)
    y = x.clone()
    psnr_same = psnr(x, y).item()
    psnr_diff = psnr(x, 1 - y).item()
    assert psnr_same > psnr_diff


def test_ssim_range() -> None:
    x = torch.rand(1, 1, 16, 16)
    y = x.clone()
    ssim_val = ssim(x, y).item()
    assert 0.9 <= ssim_val <= 1.0


def test_mse_zero_for_identical() -> None:
    x = torch.rand(1, 1, 8, 8)
    y = x.clone()
    assert mse(x, y).item() == pytest.approx(0.0, abs=1e-6)
