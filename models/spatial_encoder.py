"""Shared spatial CNN applied independently to every daily input map."""

from __future__ import annotations

import torch
from torch import nn


class SpatialEncoder(nn.Module):
    """Map `(B, T, C, H, W)` to `(B, T, 32, 26, 61)` for the PoC grid."""

    def __init__(self, in_channels: int = 7, channels: tuple[int, int] = (16, 32)) -> None:
        super().__init__()
        first, second = channels
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, first, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(first),
            nn.ReLU(inplace=True),
            nn.Conv2d(first, second, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(second),
            nn.ReLU(inplace=True),
        )
        self.out_channels = second

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode each day with the same CNN weights.

        Args:
            inputs: `(B, T, C, H, W)`.
        Returns:
            `(B, T, 32, ceil(H/4), ceil(W/4))`.
        """
        if inputs.ndim != 5:
            raise ValueError(f"Expected inputs (B, T, C, H, W), got {tuple(inputs.shape)}")
        batch, days, channels, height, width = inputs.shape
        encoded = self.network(inputs.reshape(batch * days, channels, height, width))
        _, latent_channels, latent_h, latent_w = encoded.shape
        return encoded.reshape(batch, days, latent_channels, latent_h, latent_w)
