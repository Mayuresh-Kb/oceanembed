"""Complete spatiotemporal, depth-conditioned OceanEmbed model."""

from __future__ import annotations

import torch
from torch import nn

from models.depth_decoder import DepthConditionedDecoder
from models.spatial_encoder import SpatialEncoder
from models.temporal_encoder import ConvLSTMEncoder


class OceanEmbed(nn.Module):
    """Surface sequence → latent ocean state → subsurface temperature profile."""

    def __init__(
        self,
        depths_metres: list[float],
        in_channels: int = 7,
        spatial_channels: tuple[int, int] = (16, 32),
        convlstm_hidden_channels: int = 32,
        depth_embedding_channels: int = 16,
        sst_mean: float = 0.0,
        sst_std: float = 1.0,
        residual_to_sst: bool = True,
    ) -> None:
        super().__init__()
        self.spatial_encoder = SpatialEncoder(in_channels=in_channels, channels=spatial_channels)
        self.temporal_encoder = ConvLSTMEncoder(
            input_channels=self.spatial_encoder.out_channels,
            hidden_channels=convlstm_hidden_channels,
        )
        self.depth_decoder = DepthConditionedDecoder(
            depths_metres,
            latent_channels=convlstm_hidden_channels,
            embedding_channels=depth_embedding_channels,
        )
        self.residual_to_sst = residual_to_sst
        self.register_buffer("sst_mean", torch.tensor(float(sst_mean)), persistent=True)
        self.register_buffer("sst_std", torch.tensor(float(sst_std)), persistent=True)

    def forward(self, inputs: torch.Tensor, input_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Predict temperature `(B, D, H, W)` from inputs `(B, T, 7, H, W)`.

        `input_mask`, when supplied, is `(B, T, 1, H, W)` and makes invalid
        input cells zero before spatial encoding. Target masks remain the sole
        authority for which predictions contribute to loss.
        """
        if input_mask is not None:
            if input_mask.shape[:2] != inputs.shape[:2] or input_mask.shape[-2:] != inputs.shape[-2:]:
                raise ValueError("input_mask must align with inputs as (B, T, 1, H, W).")
            inputs = inputs * input_mask.to(dtype=inputs.dtype)
        # SST is z-scored in the input tensor. Reconstruct physical SST before
        # masking so the shared decoder can learn depth-dependent departures
        # from a meaningful surface-temperature baseline rather than absolute
        # temperature entirely from random initialization.
        last_day_sst = inputs[:, -1, 0] * self.sst_std + self.sst_mean  # (B, H, W), °C
        sequence = self.spatial_encoder(inputs)
        latent = self.temporal_encoder(sequence)
        residual = self.depth_decoder(latent, output_size=tuple(inputs.shape[-2:]))
        return residual + last_day_sst[:, None] if self.residual_to_sst else residual
