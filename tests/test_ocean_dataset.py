"""Tensor-shape tests for the OceanDataset, using small synthetic arrays."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from datasets.ocean_dataset import (
    OceanDataBundle,
    OceanDataset,
    _interpolate_horizontal,
    fit_input_normalization,
)


def _bundle() -> OceanDataBundle:
    surface = np.arange(8 * 7 * 2 * 3, dtype=np.float32).reshape(8, 7, 2, 3) + 1.0
    surface[:, :, 0, 0] = np.nan
    temperature = np.ones((8, 15, 2, 3), dtype=np.float32) * 20.0
    temperature[:, :, 0, 0] = np.nan
    return OceanDataBundle(
        surface=surface,
        temperature=temperature,
        input_mask=np.isfinite(surface).all(axis=1),
        target_mask=np.isfinite(temperature),
        times=np.arange("2024-01-01", "2024-01-09", dtype="datetime64[D]"),
        channel_names=["sst", "sss", "ssh", "current_u", "current_v", "wind_u", "wind_v"],
        target_depths=np.arange(15, dtype=np.float32),
        latitude=np.array([5.0, 5.25], dtype=np.float32),
        longitude=np.array([45.0, 45.25, 45.5], dtype=np.float32),
    )


def test_dataset_returns_explicit_temporal_and_depth_shapes() -> None:
    bundle = _bundle()
    stats = fit_input_normalization(bundle, range(5))
    dataset = OceanDataset(bundle, end_day_indices=[4, 5, 6, 7], normalization=stats, temporal_window=5)
    sample = dataset[0]
    assert len(dataset) == 4
    assert tuple(sample["inputs"].shape) == (5, 7, 2, 3)
    assert tuple(sample["target"].shape) == (15, 2, 3)
    assert tuple(sample["input_mask"].shape) == (5, 1, 2, 3)
    assert tuple(sample["target_mask"].shape) == (15, 2, 3)
    assert not bool(sample["input_mask"][:, :, 0, 0].any())
    assert float(sample["inputs"][:, :, 0, 0].abs().sum()) == 0.0


def test_dataset_rejects_partial_temporal_window() -> None:
    bundle = _bundle()
    stats = fit_input_normalization(bundle, range(5))
    with pytest.raises(ValueError, match="complete temporal window"):
        OceanDataset(bundle, end_day_indices=[3], normalization=stats, temporal_window=5)


def test_normalization_rejects_empty_training_split() -> None:
    with pytest.raises(ValueError, match="At least one training day"):
        fit_input_normalization(_bundle(), [])


def test_horizontal_interpolation_handles_descending_era5_latitude() -> None:
    source = xr.DataArray(
        np.array([[[30.0, 30.0], [5.0, 5.0]]], dtype=np.float32),
        dims=("time", "latitude", "longitude"),
        coords={"time": [np.datetime64("2024-01-01")], "latitude": [30.0, 5.0], "longitude": [45.0, 70.0]},
    )
    config = {"region": {"lat_min": 5.0, "lat_max": 30.0, "lon_min": 45.0, "lon_max": 70.0}, "target_grid": {"resolution_deg": 25.0, "interpolation": "linear"}}
    result = _interpolate_horizontal(source, config)
    assert float(result.sel(latitude=5.0, longitude=45.0).values.squeeze()) == pytest.approx(5.0)
    assert float(result.sel(latitude=30.0, longitude=45.0).values.squeeze()) == pytest.approx(30.0)
