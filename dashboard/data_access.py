"""Read-only adapters from OceanEmbed artifacts to dashboard-ready objects."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import xarray as xr

from datasets.ocean_dataset import OceanDataBundle, load_poc_bundle
from preprocessing.config import load_config
from preprocessing.regridding import build_target_grid


ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "Full NIO: 30-day demo": {
        "region_title": "North Indian Ocean",
        "region_bounds": (5.0, 30.0, 45.0, 105.0),
        "context_bounds": (0.0, 35.0, 40.0, 110.0),
        "predictions": ROOT / "data" / "processed" / "smoke_test" / "validation_predictions.nc",
        "metrics": ROOT / "data" / "processed" / "argo_validation" / "argo_validation_metrics.json",
        "collocations": ROOT / "data" / "processed" / "argo_validation" / "argo_collocations.csv",
        "glorys": ROOT / "data" / "glorys_nio_20240101_20240130_z1200m.nc",
        "winds": ROOT / "data" / "era5_winds_nio_20240101_20240130_merged.nc",
        "config": None,
        "cache_dir": None,
    },
    "Bay of Bengal: 150-day experiment": {
        "region_title": "Bay of Bengal",
        "dataset_name": "CMEMS GLORYS12V1 Global Ocean Physics Reanalysis",
        "region_bounds": (10.0, 20.0, 80.0, 90.0),
        "context_bounds": (0.0, 25.0, 70.0, 100.0),
        "predictions": ROOT / "dashboard_assets" / "bay_of_bengal" / "validation_predictions.nc",
        "metrics": ROOT / "dashboard_assets" / "bay_of_bengal" / "argo_validation_metrics.json",
        "collocations": ROOT / "dashboard_assets" / "bay_of_bengal" / "argo_collocations.csv",
        "dashboard_cache": ROOT / "dashboard_assets" / "bay_of_bengal" / "dataset_cache",
        "glorys": ROOT / "data" / "glorys_nio_20240201_20240629_z1200m.nc",
        "winds": ROOT / "data" / "era5_winds_nio_20240201_20240629_merged.nc",
        "config": ROOT / "configs" / "hackathon_patch_150d.yaml",
        "cache_dir": ROOT / "data" / "processed" / "hackathon_patch_150d" / "dataset_cache",
    },
}


def _run(name: str) -> dict[str, object]:
    if name not in RUNS:
        raise KeyError(f"Unknown dashboard run: {name}")
    return RUNS[name]


def load_prediction_cube(name: str = "Bay of Bengal: 150-day experiment") -> xr.Dataset:
    """Load saved model output; callers should not use this for training."""
    return xr.open_dataset(_run(name)["predictions"])


def load_metrics(name: str = "Bay of Bengal: 150-day experiment") -> dict[str, object]:
    return json.loads(_run(name)["metrics"].read_text(encoding="utf-8"))


def load_collocations(name: str = "Bay of Bengal: 150-day experiment") -> list[dict[str, str]]:
    with _run(name)["collocations"].open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_surface_fields(name: str = "Bay of Bengal: 150-day experiment") -> tuple[object, dict[str, np.ndarray]]:
    """Return raw physical-unit fields for the five prediction dates.

    `bundle.surface` is `(N, 7, H, W)` and represents reanalysis/ERA5 inputs,
    not direct satellite observations in this PoC.
    """
    run = _run(name)
    config = load_config(run["config"])
    dashboard_cache = run.get("dashboard_cache")
    if dashboard_cache is not None and Path(dashboard_cache).exists():
        bundle = _load_dashboard_bundle(Path(dashboard_cache), config)
    else:
        bundle = load_poc_bundle(
            run["glorys"], run["winds"], config, cache_dir=run["cache_dir"]
        )
    values = {name: bundle.surface[:, index] for index, name in enumerate(bundle.channel_names)}
    return bundle, values


def _load_dashboard_bundle(cache_dir: Path, config: dict[str, object]) -> OceanDataBundle:
    """Open the compact, read-only dashboard cache without raw NetCDF inputs."""
    metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    latitude, longitude = build_target_grid(config)
    return OceanDataBundle(
        surface=np.load(cache_dir / "surface.npy", mmap_mode="r"),
        temperature=np.load(cache_dir / "temperature.npy", mmap_mode="r"),
        input_mask=np.load(cache_dir / "input_mask.npy", mmap_mode="r"),
        target_mask=np.load(cache_dir / "target_mask.npy", mmap_mode="r"),
        times=np.asarray(metadata["times"], dtype="datetime64[D]"),
        channel_names=list(metadata["channels"]),
        target_depths=np.asarray(metadata["depths_metres"], dtype=np.float32),
        latitude=latitude,
        longitude=longitude,
    )
