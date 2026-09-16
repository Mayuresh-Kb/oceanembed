"""Build explicit OceanEmbed training tensors from GLORYS and ERA5 files.

The data contract is deliberately separate from neural-network code:

  surface:      (N_days, 7, H, W)
  temperature:  (N_days, 15, H, W)
  input_mask:   (N_days, H, W)
  target_mask:  (N_days, 15, H, W)

`OceanDataset` turns the above into temporal samples:

  inputs:       (T, 7, H, W)
  target:       (15, H, W) for the final day of the T-day window
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import ExitStack
import gc
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr
import dask

from preprocessing.config import load_config
from preprocessing.regridding import build_target_grid


@dataclass
class OceanDataBundle:
    """Aligned raw arrays and masks before input normalization.

    GLORYS remains a training reference/reanalysis. It is not represented here
    as perfect ground truth.
    """

    surface: np.ndarray  # (N, 7, H, W), raw physical units, NaN on invalid cells
    temperature: np.ndarray  # (N, 15, H, W), degrees C, NaN on invalid cells
    input_mask: np.ndarray  # (N, H, W), all seven input channels finite
    target_mask: np.ndarray  # (N, 15, H, W), valid reference-temperature cells
    times: np.ndarray  # (N,), datetime64[D]
    channel_names: list[str]
    target_depths: np.ndarray  # (15,), metres
    latitude: np.ndarray  # (H,)
    longitude: np.ndarray  # (W,)

    def __post_init__(self) -> None:
        n, channels, height, width = self.surface.shape
        if self.temperature.shape != (n, len(self.target_depths), height, width):
            raise ValueError("temperature shape must be (N, D, H, W) aligned to surface.")
        if self.input_mask.shape != (n, height, width):
            raise ValueError("input_mask shape must be (N, H, W).")
        if self.target_mask.shape != self.temperature.shape:
            raise ValueError("target_mask shape must match temperature.")
        if channels != len(self.channel_names):
            raise ValueError("channel_names length must match the surface channel axis.")
        if len(self.times) != n:
            raise ValueError("times length must match the day axis.")


def load_poc_bundle(
    glorys_path: str | Path | Sequence[str | Path],
    wind_path: str | Path | Sequence[str | Path],
    config: dict[str, Any] | None = None,
    *,
    cache_dir: str | Path | None = None,
) -> OceanDataBundle:
    """Load the real paired PoC sources and build raw aligned arrays.

    This function discovers configured variable names from the actual files,
    accepts `time` or `valid_time`, and requires their common daily dates.
    It does not normalize or fill NaNs.
    """
    config = config or load_config()
    with ExitStack() as stack:
        # Disk-backed preparation reads one hyperslab at a time from NetCDF.
        # Dask's all-days task graph is useful for small in-memory runs but
        # causes excessive overhead and retained buffers for 180-day GLORYS.
        stream_from_netcdf = cache_dir is not None
        glorys = _open_time_sequence(glorys_path, config, stack, chunk_by_day=not stream_from_netcdf)
        winds = _open_time_sequence(wind_path, config, stack, chunk_by_day=not stream_from_netcdf)
        common_times = np.intersect1d(glorys.time.values, winds.time.values)
        if common_times.size == 0:
            raise ValueError("GLORYS and ERA5 files have no common timestamps.")
        # ``sel(time=<all dates>)`` is advanced indexing. On a large,
        # unchunked NetCDF it can read every date before a later one-day
        # ``isel`` is applied. Preserve direct hyperslab reads when already
        # aligned (the normal regional-patch case).
        if not np.array_equal(glorys.time.values, common_times):
            glorys = glorys.sel(time=common_times)
        if not np.array_equal(winds.time.values, common_times):
            winds = winds.sel(time=common_times)

        if cache_dir is not None:
            return _build_disk_backed_bundle(
                glorys, winds, common_times, config, Path(cache_dir)
            )

        surface_fields = _build_surface_fields(glorys, winds, config)
        target = _build_temperature_target(glorys, config)
        # Regrid each source field independently before combining them. Joining
        # GLORYS (~0.083°) and ERA5 (0.25°) first can create a near-duplicate
        # floating-point coordinate union and mask valid ERA5 values.
        surface_fields = {
            name: _interpolate_horizontal(field, config)
            for name, field in surface_fields.items()
        }
        target = _interpolate_horizontal(target, config)

        # Force one documented order before converting to NumPy/PyTorch.
        channel_names = list(config["channel_order"])
        surface_data = xr.concat(
            # Drop only non-dimension metadata (for example the scalar depth
            # left after selecting GLORYS's surface layer, or ERA5's number).
            # Time/latitude/longitude remain coordinate axes.
            [surface_fields[name].reset_coords(drop=True) for name in channel_names],
            dim=xr.IndexVariable("channel", channel_names),
            coords="minimal",
            compat="override",
        ).transpose("time", "channel", "latitude", "longitude")
        # Do not materialize a 180-day high-resolution Dask graph in one go.
        # Chunked conversion retains the identical final NumPy contract while
        # keeping peak memory practical on student hardware.
        surface = _to_float32_time_chunks(surface_data)
        temperature = _to_float32_time_chunks(target.transpose("time", "depth", "latitude", "longitude"))

        input_mask = np.isfinite(surface).all(axis=1)
        target_mask = np.isfinite(temperature)
        lat, lon = build_target_grid(config)
        return OceanDataBundle(
            surface=surface,
            temperature=temperature,
            input_mask=input_mask,
            target_mask=target_mask,
            times=np.asarray(common_times, dtype="datetime64[D]"),
            channel_names=channel_names,
            target_depths=np.asarray(config["depth"]["target_metres"], dtype=np.float32),
            latitude=lat,
            longitude=lon,
        )


def _build_disk_backed_bundle(
    glorys: xr.Dataset,
    winds: xr.Dataset,
    common_times: np.ndarray,
    config: dict[str, Any],
    cache_dir: Path,
) -> OceanDataBundle:
    """Create or reopen day-wise regridded arrays stored as NumPy memmaps.

    A 180-day, 15-depth cube is too large to hold alongside xarray, PyTorch,
    and MPS tensors on an 8 GB student laptop.  This keeps the same array
    contract, but pages data from disk; ``OceanDataset`` still copies only its
    five-day input window into RAM for each sample.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    channel_names = list(config["channel_order"])
    depths = np.asarray(config["depth"]["target_metres"], dtype=np.float32)
    latitude, longitude = build_target_grid(config)
    n_days, height, width = len(common_times), len(latitude), len(longitude)
    paths = {
        "surface": cache_dir / "surface.npy",
        "temperature": cache_dir / "temperature.npy",
        "input_mask": cache_dir / "input_mask.npy",
        "target_mask": cache_dir / "target_mask.npy",
        "metadata": cache_dir / "metadata.json",
    }
    signature = {
        "times": [str(value) for value in np.asarray(common_times, dtype="datetime64[D]")],
        "shape": [n_days, len(channel_names), len(depths), height, width],
        "channels": channel_names,
        "depths_metres": depths.tolist(),
    }
    if paths["metadata"].exists() and all(path.exists() for key, path in paths.items() if key != "metadata"):
        saved = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        if saved == signature:
            return _open_cached_bundle(paths, signature, latitude, longitude)

    surface = np.lib.format.open_memmap(
        paths["surface"], mode="w+", dtype=np.float32,
        shape=(n_days, len(channel_names), height, width),
    )
    temperature = np.lib.format.open_memmap(
        paths["temperature"], mode="w+", dtype=np.float32,
        shape=(n_days, len(depths), height, width),
    )
    input_mask = np.lib.format.open_memmap(
        paths["input_mask"], mode="w+", dtype=np.bool_, shape=(n_days, height, width)
    )
    target_mask = np.lib.format.open_memmap(
        paths["target_mask"], mode="w+", dtype=np.bool_, shape=(n_days, len(depths), height, width)
    )
    # A serial scheduler avoids retaining parallel interpolation buffers on
    # memory-constrained laptops. Each iteration evaluates exactly one day.
    with dask.config.set(scheduler="single-threaded"):
        for day in range(n_days):
            # The one-day slice is intentional: it bounds regridding memory.
            one_day_glorys = glorys.isel(time=slice(day, day + 1))
            one_day_winds = winds.isel(time=slice(day, day + 1))
            fields = {
                name: _interpolate_horizontal(field, config)
                for name, field in _build_surface_fields(one_day_glorys, one_day_winds, config).items()
            }
            stacked = xr.concat(
                [fields[name].reset_coords(drop=True) for name in channel_names],
                dim=xr.IndexVariable("channel", channel_names), coords="minimal", compat="override",
            ).transpose("time", "channel", "latitude", "longitude")
            day_surface = np.asarray(stacked.values, dtype=np.float32)[0]
            day_target = _interpolate_horizontal(
                _build_temperature_target(one_day_glorys, config), config
            ).transpose("time", "depth", "latitude", "longitude")
            day_temperature = np.asarray(day_target.values, dtype=np.float32)[0]
            surface[day] = day_surface
            temperature[day] = day_temperature
            input_mask[day] = np.isfinite(day_surface).all(axis=0)
            target_mask[day] = np.isfinite(day_temperature)
            # xarray's delayed interpolation objects can retain task graphs if
            # references survive an iteration. Drop them before the next day.
            del one_day_glorys, one_day_winds, fields, stacked, day_target
            del day_surface, day_temperature
            gc.collect()
            if (day + 1) % 10 == 0 or day + 1 == n_days:
                print(f"Prepared disk-backed training data: {day + 1}/{n_days} days", flush=True)
    for array in (surface, temperature, input_mask, target_mask):
        array.flush()
    paths["metadata"].write_text(json.dumps(signature, indent=2), encoding="utf-8")
    return _open_cached_bundle(paths, signature, latitude, longitude)


def _open_cached_bundle(
    paths: dict[str, Path], signature: dict[str, Any], latitude: np.ndarray, longitude: np.ndarray
) -> OceanDataBundle:
    """Open a validated cache read-only so training cannot mutate raw tensors."""
    return OceanDataBundle(
        surface=np.load(paths["surface"], mmap_mode="r"),
        temperature=np.load(paths["temperature"], mmap_mode="r"),
        input_mask=np.load(paths["input_mask"], mmap_mode="r"),
        target_mask=np.load(paths["target_mask"], mmap_mode="r"),
        times=np.asarray(signature["times"], dtype="datetime64[D]"),
        channel_names=list(signature["channels"]),
        target_depths=np.asarray(signature["depths_metres"], dtype=np.float32),
        latitude=latitude,
        longitude=longitude,
    )


def _open_time_sequence(
    paths: str | Path | Sequence[str | Path],
    config: dict[str, Any],
    stack: ExitStack,
    *,
    chunk_by_day: bool = True,
) -> xr.Dataset:
    """Open one or more same-schema files and concatenate only along time.

    This avoids writing a large duplicate combined GLORYS NetCDF. Every source
    is standardized first; spatial-coordinate mismatches are rejected by the
    exact join rather than silently interpolated during concatenation.
    """
    items = [paths] if isinstance(paths, (str, Path)) else list(paths)
    if not items:
        raise ValueError("At least one input NetCDF path is required.")
    datasets = []
    for path in items:
        raw = stack.enter_context(
            xr.open_dataset(path, chunks={} if chunk_by_day else None, cache=False)
        )
        standardized = _standardize_coordinates(raw, config)
        if chunk_by_day:
            # Source files can store all 150 days in a single on-disk chunk.
            # Force day-sized chunks for the small in-memory pathway.
            datasets.append(standardized.chunk({"time": 1}))
        else:
            # Keep direct netCDF4 indexing for disk-backed streaming. It reads
            # only the requested time/depth/lat/lon hyperslab per iteration.
            datasets.append(standardized)
    # Avoid xarray concatenation for the common one-file case. On unchunked
    # NetCDF, concatenation can turn later daily indexing into a full variable
    # read. Multi-file runs retain the strict coordinate validation below.
    sequence = datasets[0] if len(datasets) == 1 else xr.concat(
        datasets, dim="time", join="exact", compat="equals"
    ).sortby("time")
    values = np.asarray(sequence.time.values)
    if np.unique(values).size != values.size:
        raise ValueError("Input files contain duplicate timestamps; refusing to concatenate them.")
    daily_values = values.astype("datetime64[D]")
    expected = daily_values[0] + np.arange(daily_values.size).astype("timedelta64[D]")
    if not np.array_equal(daily_values, expected):
        missing = np.setdiff1d(expected, daily_values)
        preview = ", ".join(str(day) for day in missing[:5])
        raise ValueError(
            "Input timestamps are not consecutive daily observations; ConvLSTM windows "
            f"would be scientifically invalid. Missing day(s): {preview}."
        )
    return sequence


def _to_float32_time_chunks(data: xr.DataArray, *, days_per_chunk: int = 8) -> np.ndarray:
    """Materialize a time-major DataArray in bounded-memory chunks."""
    if "time" not in data.dims:
        raise ValueError("Expected a time dimension for chunked conversion.")
    shape = tuple(data.sizes[dimension] for dimension in data.dims)
    result = np.empty(shape, dtype=np.float32)
    for start in range(0, shape[0], days_per_chunk):
        stop = min(start + days_per_chunk, shape[0])
        result[start:stop] = data.isel(time=slice(start, stop)).values.astype(np.float32)
    return result


def fit_input_normalization(
    bundle: OceanDataBundle,
    fit_day_indices: Iterable[int],
    *,
    eps: float = 1e-6,
) -> dict[str, dict[str, float]]:
    """Fit channel statistics on declared training days only.

    Passing validation/test days here would leak information. The caller must
    explicitly provide training day indices.
    """
    indices = np.asarray(list(fit_day_indices), dtype=int)
    if indices.size == 0:
        raise ValueError("At least one training day index is required for normalization.")
    if indices.min() < 0 or indices.max() >= bundle.surface.shape[0]:
        raise IndexError("Normalization indices are outside the available time axis.")
    stats: dict[str, dict[str, float]] = {}
    for channel_index, name in enumerate(bundle.channel_names):
        values = bundle.surface[indices, channel_index]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError(f"No finite training values available for {name}.")
        mean = float(finite.mean())
        std = float(finite.std())
        if not np.isfinite(std) or std < eps:
            raise ValueError(f"Training standard deviation for {name} is invalid: {std}.")
        stats[name] = {"mean": mean, "std": std, "eps": eps}
    return stats


class OceanDataset(Dataset[dict[str, torch.Tensor | str]]):
    """Windowed PyTorch dataset with explicit masks and target depth axis."""

    def __init__(
        self,
        bundle: OceanDataBundle,
        *,
        end_day_indices: Iterable[int],
        normalization: dict[str, dict[str, float]],
        temporal_window: int,
        fill_value: float = 0.0,
    ) -> None:
        if temporal_window < 1:
            raise ValueError("temporal_window must be at least one day.")
        self.bundle = bundle
        self.temporal_window = temporal_window
        self.fill_value = float(fill_value)
        self.end_day_indices = np.asarray(list(end_day_indices), dtype=int)
        if self.end_day_indices.size == 0:
            raise ValueError("At least one sample end-day index is required.")
        if self.end_day_indices.min() < temporal_window - 1:
            raise ValueError("Every sample needs a complete temporal window before its end day.")
        if self.end_day_indices.max() >= bundle.surface.shape[0]:
            raise IndexError("Sample end-day index is outside the available time axis.")
        self.normalization = normalization
        missing_stats = [name for name in bundle.channel_names if name not in normalization]
        if missing_stats:
            raise KeyError(f"Missing normalization statistics for channels: {missing_stats}")

    def __len__(self) -> int:
        return int(self.end_day_indices.size)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        end = int(self.end_day_indices[index])
        start = end - self.temporal_window + 1
        raw_inputs = self.bundle.surface[start : end + 1].copy()  # (T, C, H, W)
        for channel_index, name in enumerate(self.bundle.channel_names):
            stats = self.normalization[name]
            raw_inputs[:, channel_index] = (raw_inputs[:, channel_index] - stats["mean"]) / stats["std"]

        # Zero is only a transport value; the paired masks define valid cells.
        inputs = np.nan_to_num(raw_inputs, nan=self.fill_value, posinf=self.fill_value, neginf=self.fill_value)
        target = np.nan_to_num(self.bundle.temperature[end], nan=self.fill_value, posinf=self.fill_value, neginf=self.fill_value)
        input_mask = self.bundle.input_mask[start : end + 1, None, :, :]
        target_mask = self.bundle.target_mask[end]
        return {
            "inputs": torch.from_numpy(inputs.astype(np.float32)),  # (T, C, H, W)
            "target": torch.from_numpy(target.astype(np.float32)),  # (D, H, W)
            "input_mask": torch.from_numpy(input_mask.astype(bool)),  # (T, 1, H, W)
            "target_mask": torch.from_numpy(target_mask.astype(bool)),  # (D, H, W)
            "end_time": str(self.bundle.times[end]),
        }


def _standardize_coordinates(dataset: xr.Dataset, config: dict[str, Any]) -> xr.Dataset:
    """Rename configured coordinate aliases to the internal names once."""
    candidates = config["poc_readiness"]["coordinate_candidates"]
    rename: dict[str, str] = {}
    for canonical in ("time", "latitude", "longitude", "depth"):
        for name in candidates[canonical]:
            if name in dataset.dims or name in dataset.coords:
                if name != canonical:
                    rename[name] = canonical
                break
    standardized = dataset.rename(rename) if rename else dataset
    required = ("time", "latitude", "longitude")
    missing = [name for name in required if name not in standardized.coords]
    if missing:
        raise KeyError(f"Dataset is missing required coordinates after standardization: {missing}")
    return standardized.sortby("time")


def _build_surface_fields(
    glorys: xr.Dataset, winds: xr.Dataset, config: dict[str, Any]
) -> dict[str, xr.DataArray]:
    fields: dict[str, xr.DataArray] = {}
    for canonical in ("sst", "sss", "current_u", "current_v"):
        source = _find_variable(glorys, config["surface_inputs"][canonical])
        fields[canonical] = _surface_level(glorys[source])
    ssh_source = _find_variable(glorys, config["surface_inputs"]["ssh"])
    fields["ssh"] = glorys[ssh_source]
    for canonical in ("wind_u", "wind_v"):
        source = _find_variable(winds, config["surface_inputs"][canonical])
        fields[canonical] = winds[source]
    return fields


def _build_temperature_target(glorys: xr.Dataset, config: dict[str, Any]) -> xr.DataArray:
    source = _find_variable(glorys, config["targets"]["temperature"])
    temperature = glorys[source]
    if "depth" not in temperature.dims:
        raise ValueError("GLORYS temperature target has no depth dimension.")
    native_depths = np.asarray(temperature.depth.values, dtype=float)
    requested = np.asarray(config["depth"]["target_metres"], dtype=float)
    tolerance = float(config["depth"]["surface_nearest_tolerance_metres"])
    outputs: list[xr.DataArray] = []
    for depth in requested:
        if depth == 0:
            nearest_index = int(np.argmin(np.abs(native_depths)))
            nearest = native_depths[nearest_index]
            if abs(nearest) > tolerance:
                raise ValueError(f"Nearest native surface level ({nearest} m) exceeds {tolerance} m tolerance.")
            layer = temperature.isel(depth=nearest_index).expand_dims(depth=[depth])
        else:
            if depth < native_depths.min() or depth > native_depths.max():
                raise ValueError(f"Requested depth {depth} m is outside native GLORYS coverage; refusing extrapolation.")
            layer = temperature.interp(depth=[depth])
        outputs.append(layer)
    return xr.concat(outputs, dim="depth").rename("temperature")


def _interpolate_horizontal(data: xr.Dataset | xr.DataArray, config: dict[str, Any]) -> xr.Dataset | xr.DataArray:
    lat, lon = build_target_grid(config)
    # ERA5 commonly stores latitude north-to-south while GLORYS commonly uses
    # south-to-north. xarray interpolation needs monotonic source coordinates.
    ordered = data.sortby("latitude").sortby("longitude")
    # Read only a narrow native-grid halo around the configured target area.
    # This is essential for the documented regional-patch experiment and also
    # avoids silently using observations outside the requested study box.
    pad = float(config["region"].get("subset_pad_deg", 0.2))
    ordered = ordered.sel(
        latitude=slice(float(lat.min()) - pad, float(lat.max()) + pad),
        longitude=slice(float(lon.min()) - pad, float(lon.max()) + pad),
    )
    return ordered.interp(
        latitude=lat,
        longitude=lon,
        method=config["target_grid"].get("interpolation", "linear"),
        kwargs={"fill_value": np.nan},
    )


def _surface_level(data: xr.DataArray) -> xr.DataArray:
    return data.isel(depth=0, drop=True) if "depth" in data.dims else data


def _find_variable(dataset: xr.Dataset, spec: dict[str, Any]) -> str:
    for name in spec.get("candidates", []):
        if name in dataset.data_vars:
            return str(name)
    wanted = {str(value).lower() for value in spec.get("standard_names", [])}
    for name, variable in dataset.data_vars.items():
        if str(variable.attrs.get("standard_name", "")).lower() in wanted:
            return str(name)
    raise KeyError(f"Could not find variable for role {spec.get('role', 'unknown')} in {list(dataset.data_vars)}")
