"""Dual-output U-Net with shared encoder and two decoders."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import nn

from src.models.blocks import ConvBlock, DownBlock, UpBlock


@dataclass
class UNetConfig:
    """Configuration for the dual-output U-Net."""

    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = 32
    depth: int = 4
    up_mode: str = "transpose"
    use_batchnorm: bool = True
    dropout: float = 0.0
    init: str = "kaiming"

    @classmethod
    def from_dict(cls, cfg: Dict[str, object]) -> "UNetConfig":
        return cls(
            in_channels=int(cfg.get("in_channels", 1)),
            out_channels=int(cfg.get("out_channels", 1)),
            base_channels=int(cfg.get("base_channels", 32)),
            depth=int(cfg.get("depth", 4)),
            up_mode=str(cfg.get("up_mode", "transpose")),
            use_batchnorm=bool(cfg.get("use_batchnorm", True)),
            dropout=float(cfg.get("dropout", 0.0)),
            init=str(cfg.get("init", "kaiming")),
        )


class UNetDual(nn.Module):
    """Shared-encoder dual-decoder U-Net."""

    def __init__(self, cfg: UNetConfig) -> None:
        super().__init__()
        if cfg.depth < 1:
            raise ValueError("depth must be >= 1")

        self.cfg = cfg
        self.stem = ConvBlock(
            cfg.in_channels,
            cfg.base_channels,
            use_batchnorm=cfg.use_batchnorm,
            dropout=cfg.dropout,
        )

        encoder_channels: List[int] = [cfg.base_channels]
        self.down_blocks = nn.ModuleList()
        in_channels = cfg.base_channels
        for _ in range(cfg.depth - 1):
            out_channels = in_channels * 2
            self.down_blocks.append(
                DownBlock(
                    in_channels,
                    out_channels,
                    use_batchnorm=cfg.use_batchnorm,
                    dropout=cfg.dropout,
                )
            )
            encoder_channels.append(out_channels)
            in_channels = out_channels

        self.bottleneck = ConvBlock(
            in_channels,
            in_channels * 2,
            use_batchnorm=cfg.use_batchnorm,
            dropout=cfg.dropout,
        )
        bottleneck_channels = in_channels * 2

        self.weight_pool = nn.AdaptiveAvgPool2d(1)
        self.weight_head = nn.Linear(bottleneck_channels, 1)

        self.decoder_a = self._build_decoder(bottleneck_channels, encoder_channels)
        self.decoder_b = self._build_decoder(bottleneck_channels, encoder_channels)

        self.head_a = nn.Conv2d(cfg.base_channels, cfg.out_channels, kernel_size=1)
        self.head_b = nn.Conv2d(cfg.base_channels, cfg.out_channels, kernel_size=1)

        self._init_weights(cfg.init)

    def _build_decoder(
        self,
        bottleneck_channels: int,
        encoder_channels: List[int],
    ) -> nn.ModuleList:
        up_blocks = nn.ModuleList()
        in_channels = bottleneck_channels
        for skip_channels in reversed(encoder_channels):
            out_channels = skip_channels
            up_blocks.append(
                UpBlock(
                    in_channels,
                    skip_channels,
                    out_channels,
                    up_mode=self.cfg.up_mode,
                    use_batchnorm=self.cfg.use_batchnorm,
                    dropout=self.cfg.dropout,
                )
            )
            in_channels = out_channels
        return up_blocks

    def _init_weights(self, init_type: str) -> None:
        init_type = init_type.lower()
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                if init_type == "kaiming":
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                elif init_type == "xavier":
                    nn.init.xavier_normal_(module.weight)
                else:
                    raise ValueError(f"Unknown init: {init_type}")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        skips: List[torch.Tensor] = []
        x = self.stem(x)
        skips.append(x)

        for down in self.down_blocks:
            x = down(x)
            skips.append(x)

        x = self.bottleneck(x)

        pooled = self.weight_pool(x).flatten(1)
        x_hat = torch.sigmoid(self.weight_head(pooled))

        a = x
        for up, skip in zip(self.decoder_a, reversed(skips)):
            a = up(a, skip)
        a = self.head_a(a)

        b = x
        for up, skip in zip(self.decoder_b, reversed(skips)):
            b = up(b, skip)
        b = self.head_b(b)

        return a, b, x_hat
