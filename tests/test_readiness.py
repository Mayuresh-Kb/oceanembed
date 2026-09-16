"""Tests for the no-fabrication data-readiness gate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from preprocessing.config import load_config
from preprocessing.readiness import assess_readiness, audit_dataset


def _synthetic_glorys(days: int = 30) -> xr.Dataset:
    time = np.arange("2020-01-01", f"2020-01-{days + 1:02d}", dtype="datetime64[D]")
    depth = np.array([0.5, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 700, 1000], dtype=np.float32)
    lat = np.array([5.0, 30.0], dtype=np.float32)
    lon = np.array([45.0, 105.0], dtype=np.float32)
    three_d = np.ones((days, depth.size, lat.size, lon.size), dtype=np.float32)
    two_d = np.ones((days, lat.size, lon.size), dtype=np.float32)
    return xr.Dataset(
        {
            "thetao": (("time", "depth", "latitude", "longitude"), three_d),
            "so": (("time", "depth", "latitude", "longitude"), three_d),
            "uo": (("time", "depth", "latitude", "longitude"), three_d),
            "vo": (("time", "depth", "latitude", "longitude"), three_d),
            "zos": (("time", "latitude", "longitude"), two_d),
        },
        coords={"time": time, "depth": depth, "latitude": lat, "longitude": lon},
    )


def _synthetic_winds(days: int = 30) -> xr.Dataset:
    time = np.arange("2020-01-01", f"2020-01-{days + 1:02d}", dtype="datetime64[D]")
    lat = np.array([5.0, 30.0], dtype=np.float32)
    lon = np.array([45.0, 105.0], dtype=np.float32)
    values = np.ones((days, lat.size, lon.size), dtype=np.float32)
    return xr.Dataset(
        {"u10": (("time", "latitude", "longitude"), values), "v10": (("time", "latitude", "longitude"), values)},
        coords={"time": time, "latitude": lat, "longitude": lon},
    )


def test_full_synthetic_glorys_audit_has_profile_coverage() -> None:
    audit = audit_dataset(_synthetic_glorys(), "glorys", load_config())
    assert audit.depth_count == 14
    assert audit.depth_range == (0.5, 1000.0)
    assert not audit.issues


def test_surface_only_real_sample_fails_model_readiness() -> None:
    root = Path(__file__).resolve().parents[1]
    sample = root / "data" / "cmems_mod_glo_phy_my_0.083deg_P1D-m_1788172075365.nc"
    if not sample.exists():
        return
    report = assess_readiness(glorys_files=[sample], wind_files=[])
    codes = {issue.code for issue in report.model_issues}
    assert not report.model_ready
    assert "no_wind_files" in codes
    assert "too_few_depth_levels" in codes
    assert "insufficient_depth_coverage" in codes
    assert "too_few_aligned_days" in codes
