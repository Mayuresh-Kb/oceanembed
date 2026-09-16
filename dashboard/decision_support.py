"""Derived decision-support products for the read-only OceanEmbed dashboard.

All outputs remain reconstruction/analysis products.  This module does not
create forecasts or official disaster warnings.
"""

from __future__ import annotations

import csv
from io import StringIO
from typing import Any

import numpy as np
from scipy import ndimage

from dashboard.alerts import short_reference_alert


def reference_mean(temperature: np.ndarray, depth_index: int, reference_days: list[int]) -> np.ndarray:
    """Return a masked mean reference for one depth from a raw bundle."""
    layers = temperature[reference_days, depth_index]
    finite_count = np.isfinite(layers).sum(axis=0)
    return np.divide(
        np.nansum(layers, axis=0), finite_count,
        out=np.full_like(layers[0], np.nan), where=finite_count > 0,
    )


def previous_difference(current: np.ndarray, previous: np.ndarray | None) -> np.ndarray | None:
    """Return current minus previous analysis, respecting invalid cells."""
    if previous is None:
        return None
    difference = current - previous
    return np.where(np.isfinite(current) & np.isfinite(previous), difference, np.nan)


def anomaly_regions(
    anomaly: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    config: dict[str, Any],
) -> list[dict[str, float | int | str]]:
    """Find connected moderate/high anomaly areas for map review markers."""
    moderate = float(config["moderate_abs_temperature_anomaly_c"])
    high = float(config["high_abs_temperature_anomaly_c"])
    minimum_cells = int(config.get("minimum_anomaly_region_cells", 1))
    active = np.isfinite(anomaly) & (np.abs(anomaly) >= moderate)
    labels, count = ndimage.label(active, structure=np.ones((3, 3), dtype=int))
    regions: list[dict[str, float | int | str]] = []
    for identifier in range(1, count + 1):
        rows, columns = np.where(labels == identifier)
        if rows.size < minimum_cells:
            continue
        values = anomaly[rows, columns]
        # Approximate each grid cell on a sphere. This is a dashboard review
        # aid, not a geodetic area calculation for a legal/operational product.
        lat_step = float(np.median(np.diff(latitude))) if len(latitude) > 1 else 0.25
        lon_step = float(np.median(np.diff(longitude))) if len(longitude) > 1 else 0.25
        area_km2 = float(np.sum(
            (111.32 * abs(lat_step)) * (111.32 * np.cos(np.deg2rad(latitude[rows])) * abs(lon_step))
        ))
        severity = "HIGH ANOMALY" if np.any(np.abs(values) >= high) else "MODERATE ANOMALY"
        regions.append({
            "id": int(identifier), "severity": severity, "cell_count": int(rows.size),
            "lat_min": float(latitude[rows].min()), "lat_max": float(latitude[rows].max()),
            "lon_min": float(longitude[columns].min()), "lon_max": float(longitude[columns].max()),
            "centroid_lat": float(latitude[rows].mean()), "centroid_lon": float(longitude[columns].mean()),
            "area_km2_approx": area_km2, "mean_anomaly_c": float(np.mean(values)),
            "maximum_absolute_anomaly_c": float(np.max(np.abs(values))),
        })
    return sorted(regions, key=lambda item: float(item["maximum_absolute_anomaly_c"]), reverse=True)


def build_decision_summary(
    prediction: np.ndarray,
    reference: np.ndarray,
    previous: np.ndarray | None,
    latitude: np.ndarray,
    longitude: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Create one shared operational summary for cards, map, and exports."""
    anomaly = prediction - reference
    change = previous_difference(prediction, previous)
    alert = short_reference_alert(prediction, reference, config)
    valid = np.isfinite(change) if change is not None else np.zeros_like(prediction, dtype=bool)
    threshold = float(config.get("day_to_day_change_threshold_c", 0.5))
    change_mean = float(np.mean(change[valid])) if valid.any() else None
    changed_fraction = float(np.mean(np.abs(change[valid]) >= threshold)) if valid.any() else None
    regions = anomaly_regions(anomaly, latitude, longitude, config)
    status = str(alert["status"])
    action = (
        "Review the highlighted area and compare the anomaly, change, and ARGO evidence."
        if status in {"MODERATE ANOMALY", "HIGH ANOMALY"}
        else "Continue routine monitoring; no temperature-anomaly area currently needs priority review."
    )
    return {
        "alert": alert, "anomaly": anomaly, "change": change, "regions": regions,
        "change_mean_c": change_mean, "changed_fraction": changed_fraction,
        "change_threshold_c": threshold, "action": action,
    }


def nearest_cell(latitude: np.ndarray, longitude: np.ndarray, point_lat: float, point_lon: float) -> tuple[int, int]:
    return int(np.argmin(np.abs(latitude - point_lat))), int(np.argmin(np.abs(longitude - point_lon)))


def layer_csv(
    *, layer: np.ndarray, latitude: np.ndarray, longitude: np.ndarray, layer_name: str,
    units: str, date: str, depth: float, run_label: str,
) -> bytes:
    """Export every selected map cell with explicit provenance and validity."""
    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=["date", "depth_m", "latitude", "longitude", "layer", "value", "units", "valid", "run_label"])
    writer.writeheader()
    for row, lat in enumerate(latitude):
        for column, lon in enumerate(longitude):
            value = float(layer[row, column]) if np.isfinite(layer[row, column]) else ""
            writer.writerow({"date": date, "depth_m": depth, "latitude": float(lat), "longitude": float(lon), "layer": layer_name, "value": value, "units": units, "valid": bool(np.isfinite(layer[row, column])), "run_label": run_label})
    return stream.getvalue().encode("utf-8")


def regions_csv(regions: list[dict[str, float | int | str]], *, date: str, depth: float, run_label: str) -> bytes:
    """Export detected anomaly regions, including their review-oriented metadata."""
    stream = StringIO()
    fields = ["date", "depth_m", "run_label", "id", "severity", "cell_count", "area_km2_approx", "lat_min", "lat_max", "lon_min", "lon_max", "centroid_lat", "centroid_lon", "mean_anomaly_c", "maximum_absolute_anomaly_c"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for region in regions:
        writer.writerow({"date": date, "depth_m": depth, "run_label": run_label, **region})
    return stream.getvalue().encode("utf-8")
