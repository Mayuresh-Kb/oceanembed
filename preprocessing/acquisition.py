"""Build explicit, reviewable requests for the OceanEmbed PoC data subset.

Network clients are imported only by the execute functions. The request builders
are deterministic and testable without credentials or network access.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import yaml

from preprocessing.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACQUISITION_CONFIG = REPO_ROOT / "configs" / "poc_acquisition.yaml"


def load_acquisition_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load acquisition choices separate from scientific preprocessing config."""
    config_path = Path(path) if path else DEFAULT_ACQUISITION_CONFIG
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Acquisition config at {config_path} must be a mapping.")
    return config


def build_glorys_request(
    acquisition: dict[str, Any],
    scientific: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a Copernicus Marine subset request for multi-level GLORYS."""
    scientific = scientific or load_config()
    region = scientific["region"]
    period = acquisition["period"]
    glorys = acquisition["glorys"]
    return {
        "dataset_id": glorys["dataset_id"],
        "variables": list(glorys["variables"]),
        "minimum_longitude": float(region["lon_min"]),
        "maximum_longitude": float(region["lon_max"]),
        "minimum_latitude": float(region["lat_min"]),
        "maximum_latitude": float(region["lat_max"]),
        "minimum_depth": float(glorys["depth_min_metres"]),
        "maximum_depth": float(glorys["depth_max_metres"]),
        "start_datetime": period["start"],
        "end_datetime": period["end"],
        "file_format": "netcdf",
    }


def build_era5_request(
    acquisition: dict[str, Any],
    scientific: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Build a CDS daily-mean wind request.

    CDS daily-statistics requests use lists of years/months/days. To prevent a
    Cartesian-product request containing unintended dates, this PoC manifest
    accepts a period inside exactly one calendar month.
    """
    scientific = scientific or load_config()
    start, end = _parse_period(acquisition["period"])
    if (start.year, start.month) != (end.year, end.month):
        raise ValueError("Split a cross-month ERA5 request into separate monthly requests.")
    winds = acquisition["era5_winds"]
    region = scientific["region"]
    days = [f"{item.day:02d}" for item in _inclusive_dates(start, end)]
    request = {
        "product_type": ["reanalysis"],
        "variable": list(winds["variables"]),
        "year": [str(start.year)],
        "month": [f"{start.month:02d}"],
        "day": days,
        "daily_statistic": winds["daily_statistic"],
        "time_zone": winds["time_zone"],
        "frequency": winds["frequency"],
        # CDS order: north, west, south, east.
        "area": [
            float(region["lat_max"]),
            float(region["lon_min"]),
            float(region["lat_min"]),
            float(region["lon_max"]),
        ],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    return str(winds["dataset_id"]), request


def execute_glorys_download(
    request: dict[str, Any], *, output_dir: str | Path, output_filename: str
) -> Any:
    """Run an authenticated GLORYS request using stored Copernicus credentials."""
    import copernicusmarine

    return copernicusmarine.subset(
        **request,
        output_directory=Path(output_dir),
        output_filename=output_filename,
        overwrite=False,
    )


def execute_era5_download(
    dataset_id: str, request: dict[str, Any], *, output_path: str | Path
) -> Any:
    """Run an authenticated ERA5 request using a locally configured CDS API key."""
    cds_config = Path.home() / ".cdsapirc"
    if not _has_cds_credentials(cds_config):
        raise RuntimeError(
            "CDS API credentials are not configured. Create "
            f"{cds_config} using your personal token from "
            "https://cds.climate.copernicus.eu/how-to-api, then rerun this command. "
            "Do not add the token to this repository or paste it into chat."
        )
    import cdsapi

    return cdsapi.Client().retrieve(dataset_id, request, str(output_path))


def _parse_period(period: dict[str, str]) -> tuple[date, date]:
    start = date.fromisoformat(str(period["start"]))
    end = date.fromisoformat(str(period["end"]))
    if end < start:
        raise ValueError("Acquisition end date cannot be before start date.")
    return start, end


def _inclusive_dates(start: date, end: date) -> list[date]:
    days: list[date] = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


def _has_cds_credentials(path: Path) -> bool:
    """Check only for required keys; never read or print the token value."""
    if not path.is_file():
        return False
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition(":")
        if separator:
            values[key.strip()] = value.strip()
    return bool(values.get("url")) and bool(values.get("key"))
