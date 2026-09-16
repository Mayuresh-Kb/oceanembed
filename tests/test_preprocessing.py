"""Tests for inspection, region subset, regrid shapes, and land-mask handling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from preprocessing.config import load_config
from preprocessing.loader import load_ocean_dataset, open_and_inspect, rename_to_canonical
from preprocessing.preprocessing import (
    apply_land_mask,
    ocean_mask_from_nans,
    preprocess_file,
    squeeze_singleton_depth,
    stack_channels,
    subset_region,
)
from preprocessing.regridding import build_target_grid, regrid_dataset

SAMPLE_NC = Path(__file__).resolve().parents[1] / "data" / "cmems_mod_glo_phy_my_0.083deg_P1D-m_1788172075365.nc"


pytestmark = pytest.mark.skipif(
    not SAMPLE_NC.exists(),
    reason="Sample NetCDF is not in data/; tests against real file are skipped.",
)


def test_inspect_discovers_real_variable_names() -> None:
    ds, report = open_and_inspect(SAMPLE_NC, print_report=False)
    try:
        assert "time" in report.dimensions
        assert "latitude" in report.dimensions
        assert "longitude" in report.dimensions
        # Names must come from the file, not from our guesses in isolation.
        assert "thetao" in report.variables
        assert "so" in report.variables
        assert "zos" in report.variables
        assert "uo" in report.variables
        assert "vo" in report.variables
        # Winds are not in this GLORYS physics extract.
        assert "eastward_wind" not in report.variables
        assert "u10" not in report.variables
    finally:
        ds.close()


def test_variable_mapping_uses_config_candidates() -> None:
    loaded = load_ocean_dataset(SAMPLE_NC, print_report=False)
    try:
        mapped = loaded.mapped_names()
        assert mapped["sst"] == "thetao"
        assert mapped["sss"] == "so"
        assert mapped["ssh"] == "zos"
        assert mapped["current_u"] == "uo"
        assert mapped["current_v"] == "vo"
        assert "wind_u" not in mapped
        assert "wind_v" not in mapped
    finally:
        loaded.dataset.close()


def test_nio_subset_is_nonempty_and_in_region() -> None:
    config = load_config()
    loaded = load_ocean_dataset(SAMPLE_NC, print_report=False)
    try:
        canonical = rename_to_canonical(loaded.dataset, loaded.mapping)
        regional = subset_region(canonical, config)
        assert regional.sizes["latitude"] > 0
        assert regional.sizes["longitude"] > 0
        assert float(regional.latitude.min()) <= 5.0
        assert float(regional.latitude.max()) >= 30.0
        assert float(regional.longitude.min()) <= 45.0
        assert float(regional.longitude.max()) >= 105.0 - 0.2
    finally:
        loaded.dataset.close()


def test_target_grid_shape_is_101_by_241() -> None:
    lat, lon = build_target_grid()
    assert lat.shape == (101,)
    assert lon.shape == (241,)
    assert lat[0] == pytest.approx(5.0)
    assert lat[-1] == pytest.approx(30.0)
    assert lon[0] == pytest.approx(45.0)
    assert lon[-1] == pytest.approx(105.0)


def test_regrid_and_mask_preserve_nans_on_land() -> None:
    config = load_config()
    loaded = load_ocean_dataset(SAMPLE_NC, print_report=False)
    try:
        canonical = rename_to_canonical(loaded.dataset, loaded.mapping)
        surface = squeeze_singleton_depth(subset_region(canonical, config))[["sst", "sss", "ssh"]]
        regridded = regrid_dataset(surface, config)
        mask = ocean_mask_from_nans(regridded)
        masked = apply_land_mask(regridded, mask)

        assert tuple(regridded.sizes["latitude"] for _ in [0])[0] == 101
        assert regridded.sizes["longitude"] == 241
        assert bool(mask.any())
        assert bool((~mask).any()), "Expected some land/NaN points in the NIO box"
        land_sst = masked["sst"].values[..., ~mask.values]
        assert np.isnan(land_sst).all()
    finally:
        loaded.dataset.close()


def test_preprocess_tensor_shape_tchw() -> None:
    result = preprocess_file(str(SAMPLE_NC), print_report=False)
    t, c, h, w = result.channels.shape
    assert t == 1, "This sample file has a single daily snapshot"
    assert h == 101
    assert w == 241
    assert c == 5
    assert result.channel_names == ["sst", "sss", "ssh", "current_u", "current_v"]
    # Land remains NaN in the stacked tensor (not silently zeroed).
    assert np.isnan(result.channels).any()
    # Ocean points are finite after z-score.
    ocean = np.asarray(result.ocean_mask.values, dtype=bool)
    assert np.isfinite(result.channels[:, :, ocean]).all()
    result.dataset.close()


def test_stack_channels_rejects_missing_names() -> None:
    time = np.array(["2020-01-01"], dtype="datetime64[ns]")
    lat = np.array([5.0, 5.25], dtype=np.float32)
    lon = np.array([45.0, 45.25], dtype=np.float32)
    ds = xr.Dataset(
        {"sst": (("time", "latitude", "longitude"), np.ones((1, 2, 2), dtype=np.float32))},
        coords={"time": time, "latitude": lat, "longitude": lon},
    )
    with pytest.raises(KeyError):
        stack_channels(ds, ["sst", "wind_u"])
