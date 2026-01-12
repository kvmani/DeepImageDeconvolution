"""Reusable convolutional blocks for U-Net variants."""
from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """Two-layer convolutional block with optional batch norm and dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        bias = not use_batchnorm
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.insert(3, nn.Dropout2d(dropout))
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """Downsampling block with max pooling followed by ConvBlock."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = ConvBlock(
            in_channels,
            out_channels,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return self.conv(x)


class UpBlock(nn.Module):
    """Upsampling block with concatenated skip connection."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        up_mode: Literal["transpose", "bilinear"] = "transpose",
        use_batchnorm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if up_mode == "transpose":
            self.up = nn.ConvTranspose2d(
                in_channels,
                in_channels,
                kernel_size=2,
                stride=2,
            )
        elif up_mode == "bilinear":
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
            )
        else:
            raise ValueError(f"Unknown up_mode: {up_mode}")

        self.conv = ConvBlock(
            in_channels + skip_channels,
            out_channels,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            diff_y = skip.size(2) - x.size(2)
            diff_x = skip.size(3) - x.size(3)
            x = F.pad(
                x,
                [
                    diff_x // 2,
                    diff_x - diff_x // 2,
                    diff_y // 2,
                    diff_y - diff_y // 2,
                ],
            )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)
