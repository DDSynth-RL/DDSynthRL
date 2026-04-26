"""Probability utilities (Gaussian smoothing)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianKernelConv(nn.Module):
    """Smooth categorical distributions with a Gaussian kernel over the class axis.

    This matches Synth-Matching's implementation: `sigma` is interpreted *relative*
    to the number of classes, i.e. the effective standard deviation in "bins" is
    approximately `num_classes * sigma`.
    """

    def __init__(self, k: int = 5, sigma: float = 0.02) -> None:
        super().__init__()
        self.k = int(k)
        self.sigma = sigma
        self._cache: dict[int, torch.Tensor] = {}

    def kernel(self, length: int) -> torch.Tensor:
        if length not in self._cache:
            grid = torch.arange(-self.k, self.k + 1)
            kernel = torch.exp(-(grid / length / self.sigma) ** 2 / 2)
            kernel /= kernel.sum()
            self._cache[length] = kernel
        return self._cache[length]

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        # target: (..., num_classes)
        orig_shape = target.shape
        length = int(orig_shape[-1])
        kernel = self.kernel(length).to(device=target.device, dtype=target.dtype).view(1, 1, -1)

        flat = target.reshape(-1, 1, length)
        with torch.no_grad():
            weights = F.conv1d(flat, kernel, padding=self.k)
            weights = F.normalize(weights.squeeze(1), p=1, dim=-1)
        return weights.reshape(*orig_shape)
