from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class CNNBackbone(nn.Module):
    """Simple 2D CNN encoder for log-mel spectrograms."""

    def __init__(self, in_channels: int = 1, d_model: int = 256) -> None:
        super().__init__()
        c = int(d_model)
        self.conv = nn.Sequential(
            self._block(in_channels, c // 16, kernel_size=5, batch_norm=False),
            self._block(c // 16, c // 8, kernel_size=4),
            self._block(c // 8, c // 4, kernel_size=4),
            self._block(c // 4, c // 2, kernel_size=4),
            self._block(c // 2, c, kernel_size=4),
        )

    @staticmethod
    def _block(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 2,
        padding: int = 2,
        batch_norm: bool = True,
    ) -> nn.Sequential:
        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.LeakyReLU(0.1),
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
