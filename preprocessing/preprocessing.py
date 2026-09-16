"""Region subset, daily alignment, land/NaN mask, and normalization.

These steps operate on xarray objects. Tensor conversion happens later
in the dataset class so shapes stay explicit and testable here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from preprocessing.config import load_config
from preprocessing.loader import (
    LoadedDataset,
    load_ocean_dataset,
    rename_to_canonical,
)
from preprocessing.regridding import build_target_grid, regrid_dataset


@dataclass
class ProcessedFields:
    """Preprocessed surface fields on the project grid.

    Arrays use the NetCDF dimension order from xarray, typically:
      (time, latitude, longitude) or (time, depth, latitude, longitude)

    Channel stack `channels` has shape (T, C, H, W) with NaNs on land.
    """

    dataset: xr.Dataset
    ocean_mask: xr.DataArray  # True over ocean; shape (H, W) or broadcastable
    channel_names: list[str]
    channels: np.ndarray  # (T, C, H, W)
    norm_stats: dict[str, dict[str, float]]
    notes: list[str]


def subset_region(ds: xr.Dataset, config: dict[str, Any] | None = None) -> xr.Dataset:
    """Select the North Indian Ocean box using latitude/longitude coordinates.

    Coordinate names are discovered from the dataset, not hard-coded as the
    only possible names, but `latitude`/`longitude` and `lat`/`lon` are tried.
    """
    if config is None:
        config = load_config()
    region = config["region"]
    lat_name = _find_coord(ds, ("latitude", "lat", "nav_lat"))
    lon_name = _find_coord(ds, ("longitude", "lon", "nav_lon"))

    pad = float(region.get("subset_pad_deg", 0.0))
    lat_min = float(region["lat_min"]) - pad
    lat_max = float(region["lat_max"]) + pad
    lon_min = float(region["lon_min"]) - pad
    lon_max = float(region["lon_max"]) + pad

    lat_values = ds[lat_name]
    if lat_values.ndim != 1:
        raise ValueError(f"Expected 1-D latitude coordinate, got dims={lat_values.dims}")
    # Slice direction must follow the coordinate's stored order.
    if lat_values[0] > lat_values[-1]:
        lat_slice = slice(lat_max, lat_min)
    else:
        lat_slice = slice(lat_min, lat_max)

    lon_values = ds[lon_name]
    if lon_values.ndim != 1:
        raise ValueError(f"Expected 1-D longitude coordinate, got dims={lon_values.dims}")

    subset = ds.sel({lat_name: lat_slice, lon_name: lon_slice_1d(lon_values, lon_min, lon_max)})
    if subset.sizes[lat_name] == 0 or subset.sizes[lon_name] == 0:
        raise ValueError(
            f"Region subset is empty. File lat/lon ranges may not overlap "
            f"[{lat_min}, {lat_max}] x [{lon_min}, {lon_max}]."
        )
    return subset


def lon_slice_1d(lon: xr.DataArray, lon_min: float, lon_max: float) -> slice:
    """Build a slice for a 1-D longitude coordinate in increasing order.

    This prototype assumes the requested box does not wrap the dateline.
    The North Indian Ocean box (45E–105E) does not wrap.
    """
    if lon_min > lon_max:
        raise ValueError("Wrapping longitude boxes are not implemented in this prototype.")
    if lon[0] > lon[-1]:
        return slice(lon_max, lon_min)
    return slice(lon_min, lon_max)


def align_daily(ds: xr.Dataset) -> xr.Dataset:
    """Ensure a time coordinate exists and is daily.

    If time is already daily (including a single daily snapshot), this is a no-op
    besides sorting. Sub-daily data would be averaged to calendar days.
    """
    time_name = _find_time_coord(ds)
    if time_name is None:
        raise ValueError("No time coordinate found; cannot align to daily resolution.")

    time = ds[time_name]
    ds = ds.sortby(time_name)
    if time.size <= 1:
        return ds

    # Infer typical step in nanoseconds if datetime-like.
    if np.issubdtype(time.dtype, np.datetime64):
        deltas = np.diff(time.values.astype("datetime64[ns]").astype(np.int64))
        if deltas.size and np.all(deltas >= np.timedelta64(1, "D").astype("timedelta64[ns]").astype(np.int64)):
            return ds
        return ds.resample({time_name: "1D"}).mean()
    return ds


def squeeze_singleton_depth(ds: xr.Dataset) -> xr.Dataset:
    """Drop a size-1 depth axis so surface fields are (time, lat, lon).

    Does not invent extra depth levels. If multiple depths exist, they are kept.
    """
    for name in ("depth", "deptht", "z", "lev"):
        if name in ds.dims and ds.sizes[name] == 1:
            return ds.squeeze(name, drop=True)
    return ds


def ocean_mask_from_nans(ds: xr.Dataset) -> xr.DataArray:
    """Ocean = True where every mapped variable is finite.

    Land/missing points stay NaN. They are not filled.
    """
    mask = None
    for name, da in ds.data_vars.items():
        finite = np.isfinite(da)
        # Collapse non-horizontal dims with AND so a point is ocean only if
        # it is valid for all times/depths present in this array.
        collapse = [d for d in da.dims if d not in ("latitude", "lat", "longitude", "lon")]
        if collapse:
            finite = finite.all(dim=collapse)
        mask = finite if mask is None else (mask & finite)
    if mask is None:
        raise ValueError("Cannot build a land mask from an empty dataset.")
    mask.name = "ocean_mask"
    mask.attrs["description"] = "True where all selected variables are finite (treated as ocean)."
    return mask


def apply_land_mask(ds: xr.Dataset, mask: xr.DataArray) -> xr.Dataset:
    """Set non-ocean points to NaN on every data variable."""
    masked = ds.copy()
    for name in list(masked.data_vars):
        masked[name] = masked[name].where(mask)
    masked["ocean_mask"] = mask
    return masked


def compute_zscore_stats(
    ds: xr.Dataset,
    mask: xr.DataArray,
    *,
    eps: float = 1e-6,
) -> dict[str, dict[str, float]]:
    """Variable-wise mean/std over ocean points only."""
    stats: dict[str, dict[str, float]] = {}
    for name, da in ds.data_vars.items():
        if name == "ocean_mask":
            continue
        ocean_values = da.where(mask)
        mean = float(ocean_values.mean(skipna=True).values)
        std = float(ocean_values.std(skipna=True).values)
        if not np.isfinite(std) or std < eps:
            raise ValueError(
                f"Standard deviation for {name} is too small ({std}). "
                "Check that the ocean mask leaves enough valid points."
            )
        stats[name] = {"mean": mean, "std": std, "eps": eps}
    return stats


def apply_zscore(
    ds: xr.Dataset,
    stats: dict[str, dict[str, float]],
) -> xr.Dataset:
    """Apply stored mean/std. Land NaNs remain NaN (NaN arithmetic)."""
    out = ds.copy()
    for name, var_stats in stats.items():
        if name not in out:
            continue
        out[name] = (out[name] - var_stats["mean"]) / var_stats["std"]
    return out


def stack_channels(
    ds: xr.Dataset,
    channel_names: list[str],
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    time_name: str = "time",
) -> np.ndarray:
    """Stack canonical variables to shape (T, C, H, W).

    Missing names are not silently invented. `channel_names` must already
    be the variables present after optional-wind omission.
    """
    missing = [name for name in channel_names if name not in ds]
    if missing:
        raise KeyError(f"Cannot stack missing variables: {missing}")

    arrays = []
    for name in channel_names:
        da = ds[name]
        extra = [d for d in da.dims if d not in (time_name, lat_name, lon_name)]
        if extra:
            raise ValueError(
                f"{name} still has extra dims {extra}. Squeeze singleton depth "
                "before stacking, or keep profiles in a separate target array."
            )
        da = da.transpose(time_name, lat_name, lon_name)
        arrays.append(np.asarray(da.values, dtype=np.float32))

    # stacked: (C, T, H, W) -> (T, C, H, W)
    stacked = np.stack(arrays, axis=0)
    stacked = np.transpose(stacked, (1, 0, 2, 3))
    return stacked


def preprocess_file(
    path: str,
    config: dict[str, Any] | None = None,
    *,
    print_report: bool = True,
    normalize: bool = True,
) -> ProcessedFields:
    """Full preprocessing chain for one NetCDF file.

    Steps:
      raw NetCDF -> inspect -> map variables -> region subset ->
      daily align -> squeeze surface depth -> regrid 0.25° ->
      land mask from NaNs -> ocean-only z-score -> (T, C, H, W)
    """
    if config is None:
        config = load_config()

    notes: list[str] = []
    loaded = load_ocean_dataset(path, config, print_report=print_report)
    notes.append(_source_note(loaded))

    canonical = rename_to_canonical(loaded.dataset, loaded.mapping)
    regional = subset_region(canonical, config)
    daily = align_daily(regional)
    surface = squeeze_singleton_depth(daily)

    present = [name for name in config["channel_order"] if name in surface]
    omitted = [name for name in config["channel_order"] if name not in surface]
    if omitted:
        notes.append(
            "Optional/missing surface channels omitted (not zero-filled): "
            + ", ".join(omitted)
        )
    surface = surface[present]

    regridded = regrid_dataset(surface, config)
    mask = ocean_mask_from_nans(regridded)
    masked = apply_land_mask(regridded, mask)

    stats: dict[str, dict[str, float]] = {}
    working = masked
    if normalize:
        stats = compute_zscore_stats(
            masked,
            mask,
            eps=float(config["normalization"]["eps"]),
        )
        working = apply_zscore(masked, stats)
        working["ocean_mask"] = mask

    lat_name = _find_coord(working, ("latitude", "lat"))
    lon_name = _find_coord(working, ("longitude", "lon"))
    time_name = _find_time_coord(working)
    if time_name is None:
        raise ValueError("Processed dataset lost its time coordinate.")

    channels = stack_channels(
        working,
        present,
        lat_name=lat_name,
        lon_name=lon_name,
        time_name=time_name,
    )
    # Re-apply mask on the stacked array so land stays NaN after z-score.
    mask_np = np.asarray(mask.values, dtype=bool)
    channels[:, :, ~mask_np] = np.nan

    notes.append(
        f"channel tensor shape (T, C, H, W) = {tuple(channels.shape)} "
        f"with C names {present}"
    )
    return ProcessedFields(
        dataset=working,
        ocean_mask=mask,
        channel_names=present,
        channels=channels,
        norm_stats=stats,
        notes=notes,
    )


def _source_note(loaded: LoadedDataset) -> str:
    source = loaded.dataset.attrs.get("source", "unknown")
    title = loaded.dataset.attrs.get("title", "")
    return (
        f"Training-reference candidate source={source!r}. "
        "Treat GLORYS as reanalysis, not perfect ground truth. "
        f"title={title!r}"
    )


def _find_coord(ds: xr.Dataset, names: tuple[str, ...]) -> str:
    for name in names:
        if name in ds.coords or name in ds.variables:
            return name
    raise KeyError(
        f"Could not find a coordinate named any of {names}. "
        f"Available: {list(ds.coords)}"
    )


def _find_time_coord(ds: xr.Dataset) -> str | None:
    for name in ("time", "t", "date"):
        if name in ds.coords or name in ds.dims:
            return name
    for name, coord in ds.coords.items():
        if np.issubdtype(coord.dtype, np.datetime64):
            return str(name)
    return None
