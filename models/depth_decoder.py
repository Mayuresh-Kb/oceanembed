"""One depth-conditioned decoder shared by every requested profile depth."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DepthConditionedDecoder(nn.Module):
    """Apply FiLM conditioning to a latent map and decode all requested depths."""

    def __init__(
        self,
        depths_metres: list[float],
        latent_channels: int = 32,
        embedding_channels: int = 16,
    ) -> None:
        super().__init__()
        depths = torch.tensor(depths_metres, dtype=torch.float32)
        if depths.ndim != 1 or depths.numel() == 0:
            raise ValueError("depths_metres must be a non-empty one-dimensional list.")
        self.register_buffer("depths_metres", depths, persistent=True)
        max_depth = float(depths.max())
        self.depth_embedding = nn.Sequential(
            nn.Linear(1, embedding_channels), nn.ReLU(inplace=True), nn.Linear(embedding_channels, embedding_channels), nn.ReLU(inplace=True)
        )
        self.film = nn.Linear(embedding_channels, 2 * latent_channels)
        self.decode = nn.Sequential(
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )
        self.max_depth = max(max_depth, 1.0)

    def forward(self, latent: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        """Decode `(B, C, h, w)` into `(B, D, H, W)` using shared weights."""
        if latent.ndim != 4:
            raise ValueError(f"Expected latent (B, C, h, w), got {tuple(latent.shape)}")
        batch, channels, height, width = latent.shape
        depth_values = (self.depths_metres / self.max_depth).to(dtype=latent.dtype).view(-1, 1)
        embedding = self.depth_embedding(depth_values)
        gamma, beta = self.film(embedding).chunk(2, dim=-1)  # (D, C), (D, C)
        conditioned = latent[:, None] * (1.0 + gamma[None, :, :, None, None]) + beta[None, :, :, None, None]
        conditioned = conditioned.reshape(batch * len(self.depths_metres), channels, height, width)
        decoded = F.interpolate(conditioned, size=output_size, mode="bilinear", align_corners=False)
        decoded = self.decode(decoded)
        return decoded.reshape(batch, len(self.depths_metres), output_size[0], output_size[1])
