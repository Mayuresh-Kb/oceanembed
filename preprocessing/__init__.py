"""Preprocessing package for OceanEmbed."""

from preprocessing.config import load_config
from preprocessing.loader import inspect_dataset, load_ocean_dataset, open_and_inspect
from preprocessing.preprocessing import preprocess_file
from preprocessing.regridding import build_target_grid, regrid_dataset
from preprocessing.readiness import assess_readiness, audit_file
from preprocessing.era5 import extract_era5_archive, merge_wind_netcdfs

__all__ = [
    "load_config",
    "inspect_dataset",
    "open_and_inspect",
    "load_ocean_dataset",
    "preprocess_file",
    "build_target_grid",
    "regrid_dataset",
    "assess_readiness",
    "audit_file",
    "extract_era5_archive",
    "merge_wind_netcdfs",
]
