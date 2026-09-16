"""Safely unpack and combine ERA5 wind files returned by CDS.

CDS may deliver one NetCDF per requested variable inside a ZIP archive, even
when the requested data format is NetCDF. This module validates archive paths
and requires exactly matching coordinates before merging fields.
"""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import xarray as xr

from preprocessing.loader import inspect_dataset


def extract_era5_archive(archive_path: str | Path, output_dir: str | Path) -> list[Path]:
    """Extract NetCDF members without permitting path traversal."""
    archive_path = Path(archive_path)
    output_dir = Path(output_dir).resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"ERA5 archive not found: {archive_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with ZipFile(archive_path) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if not members:
            raise ValueError(f"ERA5 archive {archive_path} contains no files.")
        for member in members:
            destination = (output_dir / member.filename).resolve()
            if output_dir not in destination.parents:
                raise ValueError(f"Unsafe archive member path: {member.filename!r}")
            if destination.suffix != ".nc":
                raise ValueError(f"Expected a NetCDF member, found: {member.filename!r}")
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite existing file: {destination}")
            with archive.open(member) as source, destination.open("wb") as target:
                target.write(source.read())
            extracted.append(destination)
    return extracted


def merge_wind_netcdfs(input_paths: list[str | Path], output_path: str | Path) -> xr.Dataset:
    """Merge separate wind fields after enforcing coordinate equality.

    `join="exact"` rejects mismatched time or spatial grids rather than
    interpolating silently. The returned dataset is already written to disk.
    """
    if len(input_paths) < 2:
        raise ValueError("At least two ERA5 NetCDF files are needed to merge wind U and V.")
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {output_path}")

    datasets = [xr.open_dataset(path) for path in input_paths]
    try:
        for path, dataset in zip(input_paths, datasets):
            print(inspect_dataset(dataset, path=path).format())
        merged = xr.merge(datasets, join="exact", compat="no_conflicts", combine_attrs="drop_conflicts")
        merged.to_netcdf(output_path)
        return merged.load()
    finally:
        for dataset in datasets:
            dataset.close()


def concatenate_monthly_winds(input_paths: list[str | Path], output_path: str | Path) -> xr.Dataset:
    """Concatenate prepared monthly U/V files along time after grid checks."""
    if len(input_paths) < 2:
        raise ValueError("At least two prepared monthly wind files are required.")
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {output_path}")
    datasets = [xr.open_dataset(path) for path in input_paths]
    try:
        first = datasets[0]
        for dataset in datasets[1:]:
            if set(dataset.data_vars) != set(first.data_vars):
                raise ValueError("Monthly wind files do not contain the same variables.")
            for coordinate in ("latitude", "longitude"):
                if coordinate not in dataset.coords or not dataset[coordinate].identical(first[coordinate]):
                    raise ValueError(f"Monthly wind files have incompatible {coordinate} coordinates.")
        combined = xr.concat(datasets, dim="valid_time" if "valid_time" in first.coords else "time").sortby("valid_time" if "valid_time" in first.coords else "time")
        time_name = "valid_time" if "valid_time" in combined.coords else "time"
        if len(set(combined[time_name].values.tolist())) != combined.sizes[time_name]:
            raise ValueError("Monthly wind files contain duplicate timestamps.")
        combined.to_netcdf(output_path)
        return combined.load()
    finally:
        for dataset in datasets:
            dataset.close()
