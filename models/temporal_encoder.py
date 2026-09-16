"""ConvLSTM temporal encoder for ordered daily spatial embeddings."""

from __future__ import annotations

import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    """One convolutional LSTM cell operating on spatial feature maps."""

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(
        self, input_t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, cell = state
        gates = self.gates(torch.cat([input_t, hidden], dim=1))
        input_gate, forget_gate, output_gate, candidate = gates.chunk(4, dim=1)
        cell = torch.sigmoid(forget_gate) * cell + torch.sigmoid(input_gate) * torch.tanh(candidate)
        hidden = torch.sigmoid(output_gate) * torch.tanh(cell)
        return hidden, cell


class ConvLSTMEncoder(nn.Module):
    """Encode ordered `(B, T, C, H, W)` maps into the final hidden state."""

    def __init__(self, input_channels: int = 32, hidden_channels: int = 32) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.cell = ConvLSTMCell(input_channels, hidden_channels)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Return the final temporal latent state `(B, hidden, H, W)`."""
        if sequence.ndim != 5:
            raise ValueError(f"Expected sequence (B, T, C, H, W), got {tuple(sequence.shape)}")
        batch, days, _, height, width = sequence.shape
        hidden = sequence.new_zeros((batch, self.hidden_channels, height, width))
        cell = torch.zeros_like(hidden)
        for day in range(days):
            hidden, cell = self.cell(sequence[:, day], (hidden, cell))
        return hidden
