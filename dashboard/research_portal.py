"""Research-oriented field selection, spatial subsetting, and CSV export."""

from __future__ import annotations

import csv
from io import StringIO

import numpy as np


RESEARCH_VARIABLES = (
    "Reconstructed subsurface temperature",
    "Surface temperature",
    "Surface salinity",
    "Sea-surface height",
    "Surface current speed",
    "Surface wind speed",
)


def research_layer(
    variable: str,
    prediction: np.ndarray,
    surface: dict[str, np.ndarray],
    date_index: int,
) -> tuple[np.ndarray, str, str, str, bool, tuple[np.ndarray, np.ndarray] | None, float | None, str]:
    """Return the selected field and its complete research metadata."""
    if variable == "Reconstructed subsurface temperature":
        return prediction, variable, "°C", "temperature", False, None, None, "OceanEmbed reconstruction"
    if variable == "Surface temperature":
        return surface["sst"][date_index], variable, "°C", "temperature", False, None, 0.0, "GLORYS reanalysis input"
    if variable == "Surface salinity":
        return surface["sss"][date_index], variable, "PSU", "salinity", False, None, 0.0, "GLORYS reanalysis input"
    if variable == "Sea-surface height":
        return surface["ssh"][date_index], variable, "m", "height", True, None, 0.0, "GLORYS reanalysis input"
    if variable == "Surface current speed":
        u, v = surface["current_u"][date_index], surface["current_v"][date_index]
        return np.hypot(u, v), variable, "m/s", "speed", False, (u, v), 0.0, "GLORYS reanalysis input"
    if variable == "Surface wind speed":
        u, v = surface["wind_u"][date_index], surface["wind_v"][date_index]
        return np.hypot(u, v), variable, "m/s", "speed", False, (u, v), 0.0, "ERA5 reanalysis input"
    raise ValueError(f"Unsupported research variable: {variable}")


def subset_indices(
    latitude: np.ndarray,
    longitude: np.ndarray,
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Find grid cells within inclusive researcher-selected bounds."""
    rows = np.where((latitude >= lat_bounds[0]) & (latitude <= lat_bounds[1]))[0]
    columns = np.where((longitude >= lon_bounds[0]) & (longitude <= lon_bounds[1]))[0]
    if rows.size == 0 or columns.size == 0:
        raise ValueError("Selected bounds contain no cells on this experiment grid.")
    return rows, columns


def subset_field(field: np.ndarray, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """Return a two-dimensional field cropped to selected grid rows/columns."""
    return field[np.ix_(rows, columns)]


def research_csv(
    *,
    field: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    variable: str,
    units: str,
    source: str,
    date: str,
    depth_metres: float | None,
    run_label: str,
) -> bytes:
    """Export a selected field with values, validity, and research provenance."""
    stream = StringIO()
    columns = [
        "date", "depth_m", "latitude", "longitude", "variable", "value", "units",
        "valid", "source", "run_label",
    ]
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for row, lat in enumerate(latitude):
        for column, lon in enumerate(longitude):
            raw_value = field[row, column]
            writer.writerow({
                "date": date,
                "depth_m": "" if depth_metres is None else depth_metres,
                "latitude": float(lat),
                "longitude": float(lon),
                "variable": variable,
                "value": float(raw_value) if np.isfinite(raw_value) else "",
                "units": units,
                "valid": bool(np.isfinite(raw_value)),
                "source": source,
                "run_label": run_label,
            })
    return stream.getvalue().encode("utf-8")
