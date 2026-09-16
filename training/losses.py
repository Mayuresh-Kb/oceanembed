"""Masked scientific losses for OceanEmbed."""

from __future__ import annotations

import torch


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE over valid target cells only; invalid filled values never contribute."""
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError("prediction, target, and mask must have identical shapes.")
    valid = mask.to(dtype=prediction.dtype)
    count = valid.sum()
    if count.item() == 0:
        raise ValueError("masked_mse received an empty valid-target mask.")
    return ((prediction - target).square() * valid).sum() / count
