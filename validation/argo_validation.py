"""Guardrails for the separate, profile-based ARGO validation stage.

This module does not download data or make a skill claim. It verifies that an
incoming profile NetCDF can support an auditable temperature-depth comparison.
"""

from __future__ import annotations

from pathlib import Path
from datetime import date
import csv
import json
import math
from typing import Iterable

import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
import numpy as np
import xarray as xr


REQUIRED_CONCEPTS = {
    "profile date/time": ("JULD", "TIME", "time", "date"),
    "profile latitude": ("LATITUDE", "latitude", "lat"),
    "profile longitude": ("LONGITUDE", "longitude", "lon"),
    "pressure/depth": ("PRES_ADJUSTED", "PRES", "pressure", "depth"),
    "temperature": ("TEMP_ADJUSTED", "TEMP", "temperature", "temp"),
    "temperature quality control": ("TEMP_ADJUSTED_QC", "TEMP_QC", "temperature_qc"),
}


def inspect_argo_profile_file(path: str | Path) -> dict[str, object]:
    """Return an explicit readiness report without assuming a file schema.

    Variable candidates are reported, rather than silently selected. A caller
    must review the report before choosing adjusted/raw fields and QC policy.
    """
    input_path = Path(path)
    with xr.open_dataset(input_path, decode_times=False) as dataset:
        names = sorted(dataset.variables)
        matches = {
            concept: [candidate for candidate in candidates if candidate in dataset.variables]
            for concept, candidates in REQUIRED_CONCEPTS.items()
        }
        missing = [concept for concept, candidates in matches.items() if not candidates]
        return {
            "path": str(input_path),
            "dimensions": {name: int(size) for name, size in dataset.sizes.items()},
            "variables": names,
            "candidate_matches": matches,
            "ready_for_manual_mapping": not missing,
            "missing_concepts": missing,
            "warning": "Inspect QC flags and data modes before using ARGO observations. ARGO is never used in training.",
        }


def index_argo_profiles(
    profile_root: str | Path,
    output_csv: str | Path,
    *,
    candidate_dates: Iterable[date],
    max_time_difference_days: int,
    lat_min: float = 5.0,
    lat_max: float = 30.0,
    lon_min: float = 45.0,
    lon_max: float = 105.0,
) -> dict[str, int]:
    """Create a compact metadata index for candidate independent profiles.

    Every NetCDF is opened only to read profile-level metadata.  The raw Argo
    download is never modified, and temperature values are not read here.
    """
    if max_time_difference_days < 0:
        raise ValueError("max_time_difference_days must be non-negative.")
    candidate_dates = list(candidate_dates)
    if not candidate_dates:
        raise ValueError("At least one OceanEmbed prediction date is required.")
    root = Path(profile_root)
    output = Path(output_csv)
    files = sorted(root.rglob("*.nc"))
    rows: list[dict[str, object]] = []
    unreadable = 0
    for path in files:
        try:
            rows.extend(
                _candidate_rows_from_file(
                    path, root, candidate_dates, max_time_difference_days,
                    lat_min, lat_max, lon_min, lon_max,
                )
            )
        except (OSError, RuntimeError, KeyError, ValueError):
            unreadable += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["relative_path", "profile_index", "profile_date", "latitude", "longitude", "data_mode", "position_qc", "time_qc", "nearest_prediction_date", "time_difference_days"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {"files_scanned": len(files), "candidate_profiles": len(rows), "unreadable_files": unreadable}


def _candidate_rows_from_file(
    path: Path,
    root: Path,
    candidate_dates: list[date],
    maximum_difference: int,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> list[dict[str, object]]:
    with Dataset(path) as dataset:
        required = ("JULD", "LATITUDE", "LONGITUDE")
        if any(name not in dataset.variables for name in required):
            return []
        juld = dataset.variables["JULD"]
        dates = num2date(juld[:], units=juld.units, only_use_cftime_datetimes=False, only_use_python_datetimes=True)
        latitudes = dataset.variables["LATITUDE"][:]
        longitudes = dataset.variables["LONGITUDE"][:]
        modes = _string_values(dataset.variables.get("DATA_MODE"), len(dates))
        position_qc = _string_values(dataset.variables.get("POSITION_QC"), len(dates))
        time_qc = _string_values(dataset.variables.get("JULD_QC"), len(dates))
    rows: list[dict[str, object]] = []
    for index, (timestamp, latitude, longitude) in enumerate(zip(dates, latitudes, longitudes)):
        if np_masked_or_nonfinite(latitude) or np_masked_or_nonfinite(longitude):
            continue
        profile_date = timestamp.date()
        closest = min(candidate_dates, key=lambda value: abs((profile_date - value).days))
        difference = abs((profile_date - closest).days)
        if difference > maximum_difference or not (lat_min <= float(latitude) <= lat_max and lon_min <= float(longitude) <= lon_max):
            continue
        rows.append({"relative_path": str(path.relative_to(root)), "profile_index": index, "profile_date": profile_date.isoformat(), "latitude": float(latitude), "longitude": float(longitude), "data_mode": modes[index], "position_qc": position_qc[index], "time_qc": time_qc[index], "nearest_prediction_date": closest.isoformat(), "time_difference_days": difference})
    return rows


def _string_values(variable: object, size: int) -> list[str]:
    if variable is None:
        return [""] * size
    values = variable[:]
    return [bytes(value).decode("ascii", errors="ignore").strip() if isinstance(value, bytes) else str(value).strip() for value in values]


def np_masked_or_nonfinite(value: object) -> bool:
    """Avoid importing NumPy just to validate scalar NetCDF metadata."""
    return bool(getattr(value, "mask", False)) or not math.isfinite(float(value))


def run_independent_argo_validation(
    *,
    candidate_csv: str | Path,
    profile_root: str | Path,
    prediction_path: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    """Collocate QC-filtered ARGO temperatures with saved OceanEmbed output.

    ARGO is read only here. It is never used for training, normalization, or
    model selection. Primary metrics exclude real-time profiles and use only
    delayed/adjusted profiles. The prototype maps pressure in dbar to depth in metres;
    this approximation is recorded in every output and must be replaced with a
    TEOS-10 conversion for a scientific analysis.
    """
    candidates = list(csv.DictReader(Path(candidate_csv).open(encoding="utf-8")))
    root = Path(profile_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    with xr.open_dataset(prediction_path) as prediction_cube:
        run_label = str(prediction_cube.attrs.get("label", "DEMO / DATA-LIMITED"))
        target_depths = np.asarray(prediction_cube.depth.values, dtype=float)
        for candidate in candidates:
            rows.extend(_collocate_candidate(candidate, root, prediction_cube, target_depths))

    collocations_path = output / "argo_collocations.csv"
    fields = ["relative_path", "profile_index", "profile_date", "prediction_date", "time_difference_days", "latitude", "longitude", "data_mode", "depth_metres_prototype", "argo_temperature_c", "oceanembed_temperature_c"]
    with collocations_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    metrics = _metrics(rows)
    metrics.update({
        "label": run_label,
        "source": "Independent ARGO in-situ profile observations; not perfect ground truth.",
        "qc_policy": "TEMP/PRES adjusted QC flag 1 only; only D/A profiles are included. R profiles are excluded from primary metrics.",
        "vertical_coordinate_note": "Prototype uses pressure in dbar as approximate depth in metres; no extrapolation.",
        "collocations_csv": str(collocations_path),
    })
    (output / "argo_validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if rows:
        _plot_mean_profile(rows, output / "argo_vs_oceanembed_mean_profile.png")
    return metrics


def _collocate_candidate(candidate: dict[str, str], root: Path, cube: xr.Dataset, depths: np.ndarray) -> list[dict[str, object]]:
    path = root / candidate["relative_path"]
    index = int(candidate["profile_index"])
    mode = candidate["data_mode"]
    if mode not in {"D", "A"}:
        return []
    with Dataset(path) as dataset:
        temperature_name, pressure_name, qc_name = "TEMP_ADJUSTED", "PRES_ADJUSTED", "TEMP_ADJUSTED_QC"
        required = (temperature_name, pressure_name, qc_name)
        if any(name not in dataset.variables for name in required):
            return []
        temperature = np.ma.filled(dataset.variables[temperature_name][index], np.nan).astype(float)
        pressure = np.ma.filled(dataset.variables[pressure_name][index], np.nan).astype(float)
        # TEMP_QC is level-resolved: (N_PROF, N_LEVELS), unlike DATA_MODE.
        # Select the same profile row before comparing it with temperature.
        qc = _decode_characters(dataset.variables[qc_name][index])
    valid = np.isfinite(temperature) & np.isfinite(pressure) & (np.asarray(qc) == "1")
    pressure, temperature = pressure[valid], temperature[valid]
    if pressure.size < 2:
        return []
    order = np.argsort(pressure)
    pressure, temperature = pressure[order], temperature[order]
    unique_pressure, unique_index = np.unique(pressure, return_index=True)
    pressure, temperature = unique_pressure, temperature[unique_index]
    valid_depths = depths[(depths >= pressure.min()) & (depths <= pressure.max())]
    if valid_depths.size == 0:
        return []
    argo_temperature = np.interp(valid_depths, pressure, temperature)
    latitude, longitude = float(candidate["latitude"]), float(candidate["longitude"])
    prediction_date = np.datetime64(candidate["nearest_prediction_date"])
    map_prediction = cube["oceanembed_temperature"].sel(time=prediction_date).interp(latitude=latitude, longitude=longitude)
    reference_valid = cube["glorys_reference_mask"].sel(time=prediction_date).interp(latitude=latitude, longitude=longitude, method="nearest")
    predicted = np.asarray(map_prediction.sel(depth=valid_depths).values, dtype=float)
    ocean = np.asarray(reference_valid.sel(depth=valid_depths).values) >= 0.5
    rows: list[dict[str, object]] = []
    for depth, observed, estimate, is_ocean in zip(valid_depths, argo_temperature, predicted, ocean):
        if not (is_ocean and np.isfinite(observed) and np.isfinite(estimate)):
            continue
        rows.append({"relative_path": candidate["relative_path"], "profile_index": index, "profile_date": candidate["profile_date"], "prediction_date": candidate["nearest_prediction_date"], "time_difference_days": int(candidate["time_difference_days"]), "latitude": latitude, "longitude": longitude, "data_mode": mode, "depth_metres_prototype": float(depth), "argo_temperature_c": float(observed), "oceanembed_temperature_c": float(estimate)})
    return rows


def _metrics(rows: list[dict[str, object]]) -> dict[str, float | int]:
    if not rows:
        return {"valid_temperature_depth_pairs": 0, "rmse_c": float("nan"), "mae_c": float("nan"), "bias_c": float("nan")}
    observed = np.asarray([float(row["argo_temperature_c"]) for row in rows])
    predicted = np.asarray([float(row["oceanembed_temperature_c"]) for row in rows])
    error = predicted - observed
    return {"valid_temperature_depth_pairs": int(len(rows)), "rmse_c": float(np.sqrt(np.mean(error**2))), "mae_c": float(np.mean(np.abs(error))), "bias_c": float(np.mean(error))}


def _decode_characters(values: object) -> list[str]:
    """Decode one NetCDF QC character array into scalar flag strings."""
    flattened = np.ma.filled(values, b"").reshape(-1)
    return [bytes(value).decode("ascii", errors="ignore").strip() if isinstance(value, bytes) else str(value).strip() for value in flattened]


def _plot_mean_profile(rows: list[dict[str, object]], output_path: Path) -> None:
    depths = sorted({float(row["depth_metres_prototype"]) for row in rows})
    observed = [np.mean([float(row["argo_temperature_c"]) for row in rows if float(row["depth_metres_prototype"]) == depth]) for depth in depths]
    predicted = [np.mean([float(row["oceanembed_temperature_c"]) for row in rows if float(row["depth_metres_prototype"]) == depth]) for depth in depths]
    figure, axis = plt.subplots(figsize=(6, 7))
    axis.plot(observed, depths, marker="o", label="ARGO (QC-filtered)")
    axis.plot(predicted, depths, marker="o", label="OceanEmbed prediction")
    axis.invert_yaxis()
    axis.set(xlabel="Temperature (°C)", ylabel="Pressure used as depth (m; prototype)", title="DEMO / DATA-LIMITED: Mean ARGO comparison")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
