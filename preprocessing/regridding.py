"""Regrid ocean fields onto the project 0.25° North Indian Ocean grid.

xESMF is the usual tool for conservative climate remapping. It is not
used in this prototype because it is extra native-library complexity on
macOS. We use xarray linear interpolation instead, then re-apply a
nearest-neighbour ocean mask so land is not treated as interpolated ocean.

This is appropriate for a student pipeline prototype, not for a final
mass-conserving climate product.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from preprocessing.config import load_config


def build_target_grid(config: dict[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Inclusive latitude/longitude vectors at the configured resolution.

    For 5–30N and 45–105E at 0.25°:
      H = 101, W = 241
    """
    if config is None:
        config = load_config()
    region = config["region"]
    step = float(config["target_grid"]["resolution_deg"])
    lat = np.arange(region["lat_min"], region["lat_max"] + step * 0.5, step, dtype=np.float32)
    lon = np.arange(region["lon_min"], region["lon_max"] + step * 0.5, step, dtype=np.float32)
    return lat, lon


def regrid_dataset(ds: xr.Dataset, config: dict[str, Any] | None = None) -> xr.Dataset:
    """Interpolate data variables onto the target lat/lon grid.

    Mask handling:
      1. Interpolate fields (NaNs stay NaN along land interiors).
      2. Rebuild ocean mask on the new grid with nearest-neighbour
         validity so coastal interpolation does not silently fill land.
    """
    if config is None:
        config = load_config()
    lat_t, lon_t = build_target_grid(config)
    lat_name = _coord_name(ds, ("latitude", "lat"))
    lon_name = _coord_name(ds, ("longitude", "lon"))
    method = config["target_grid"].get("interpolation", "linear")

    interpolated = ds.interp(
        {lat_name: lat_t, lon_name: lon_t},
        method=method,
        kwargs={"fill_value": np.nan},
    )

    # Validity mask from original finite values, nearest on the new grid.
    validity = None
    for name, da in ds.data_vars.items():
        finite = np.isfinite(da)
        collapse = [d for d in da.dims if d not in (lat_name, lon_name)]
        if collapse:
            finite = finite.all(dim=collapse)
        validity = finite if validity is None else (validity & finite)

    if validity is None:
        return interpolated

    mask_method = config["target_grid"].get("mask_interpolation", "nearest")
    validity_t = validity.astype("float32").interp(
        {lat_name: lat_t, lon_name: lon_t},
        method=mask_method,
        kwargs={"fill_value": 0.0},
    )
    ocean = validity_t >= 0.5
    out = interpolated.copy()
    for name in list(out.data_vars):
        out[name] = out[name].where(ocean)
    return out


def _coord_name(ds: xr.Dataset, names: tuple[str, ...]) -> str:
    for name in names:
        if name in ds.coords or name in ds.variables:
            return name
    raise KeyError(f"No coordinate in {names}; have {list(ds.coords)}")
