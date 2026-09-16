#!/usr/bin/env python3
"""Build a real OceanDataset and print one explicit tensor sample."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.ocean_dataset import OceanDataset, fit_input_normalization, load_poc_bundle  # noqa: E402
from preprocessing.config import load_config  # noqa: E402


def main() -> None:
    config = load_config()
    bundle = load_poc_bundle(
        ROOT / "data" / "glorys_nio_20240101_20240130_z1200m.nc",
        ROOT / "data" / "era5_winds_nio_20240101_20240130_merged.nc",
        config,
    )
    window = int(config["time"]["window"])
    # Demonstration split: first 21 days fit normalization; model training split
    # policy will be formalized in the training stage.
    stats = fit_input_normalization(bundle, range(21))
    dataset = OceanDataset(
        bundle,
        end_day_indices=range(window - 1, bundle.surface.shape[0]),
        normalization=stats,
        temporal_window=window,
        fill_value=float(config["dataset"]["tensor_fill_value"]),
    )
    sample = dataset[0]
    print("bundle surface (N, C, H, W):", bundle.surface.shape)
    print("bundle temperature (N, D, H, W):", bundle.temperature.shape)
    print("dataset samples:", len(dataset))
    print("sample inputs (T, C, H, W):", tuple(sample["inputs"].shape))
    print("sample target (D, H, W):", tuple(sample["target"].shape))
    print("sample input_mask (T, 1, H, W):", tuple(sample["input_mask"].shape))
    print("sample target_mask (D, H, W):", tuple(sample["target_mask"].shape))
    print("sample final day:", sample["end_time"])


if __name__ == "__main__":
    main()
