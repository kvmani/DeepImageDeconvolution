import pytest

pytest.importorskip("torch")

import torch

from src.utils.metrics import masked_mse, psnr, ssim, ssim_masked


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


def test_masked_mse_ignores_outside() -> None:
    x = torch.zeros(1, 1, 4, 4)
    y = torch.zeros(1, 1, 4, 4)
    y[:, :, 0, 0] = 1.0
    mask = torch.zeros(4, 4)
    mask[1:3, 1:3] = 1.0
    assert masked_mse(x, y, mask).item() == pytest.approx(0.0, abs=1e-6)


def test_ssim_masked_ignores_outside() -> None:
    x = torch.zeros(1, 1, 8, 8)
    y = x.clone()
    y[:, :, 0, 0] = 1.0
    mask = torch.zeros(8, 8)
    mask[2:6, 2:6] = 1.0
    ssim_val = ssim_masked(x, y, mask).item()
    assert 0.9 <= ssim_val <= 1.0
