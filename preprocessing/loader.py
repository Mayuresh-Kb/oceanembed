"""NetCDF inspection and loading without assuming variable names.

The loader prints dimensions, coordinates, and variables whenever a new
dataset is opened. Canonical OceanEmbed names (sst, sss, ...) are resolved
from config candidate lists and CF standard_name attributes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import xarray as xr

from preprocessing.config import load_config


@dataclass
class InspectionReport:
    """Human-readable snapshot of a NetCDF file."""

    path: str
    dimensions: dict[str, int]
    coordinates: dict[str, dict[str, Any]]
    variables: dict[str, dict[str, Any]]
    global_attrs: dict[str, Any]

    def format(self) -> str:
        lines = [
            f"NetCDF: {self.path}",
            "Dimensions:",
        ]
        for name, size in self.dimensions.items():
            lines.append(f"  {name}: {size}")
        lines.append("Coordinates:")
        for name, info in self.coordinates.items():
            lines.append(
                f"  {name}: dims={info['dims']} shape={info['shape']} "
                f"dtype={info['dtype']} min={info.get('min')} max={info.get('max')}"
            )
        lines.append("Data variables:")
        for name, info in self.variables.items():
            lines.append(
                f"  {name}: dims={info['dims']} shape={info['shape']} "
                f"dtype={info['dtype']} standard_name={info.get('standard_name')!r} "
                f"long_name={info.get('long_name')!r} units={info.get('units')!r}"
            )
        return "\n".join(lines)


@dataclass
class VariableResolution:
    """How a canonical variable was (or was not) found in a file."""

    canonical: str
    required: bool
    found: bool
    source_name: str | None = None
    matched_by: str | None = None
    notes: str = ""


@dataclass
class LoadedDataset:
    """Opened dataset plus inspection and variable mapping."""

    dataset: xr.Dataset
    report: InspectionReport
    mapping: dict[str, VariableResolution] = field(default_factory=dict)

    def mapped_names(self) -> dict[str, str]:
        """canonical -> source variable name, only for variables that were found."""
        return {
            name: res.source_name
            for name, res in self.mapping.items()
            if res.found and res.source_name is not None
        }


def inspect_dataset(ds: xr.Dataset, path: str | Path | None = None) -> InspectionReport:
    """Build an inspection report. Does not guess scientific meaning."""
    dimensions = {str(k): int(v) for k, v in ds.sizes.items()}
    coordinates: dict[str, dict[str, Any]] = {}
    for name, da in ds.coords.items():
        info: dict[str, Any] = {
            "dims": tuple(da.dims),
            "shape": tuple(int(s) for s in da.shape),
            "dtype": str(da.dtype),
        }
        if np.issubdtype(da.dtype, np.number) and da.size > 0:
            info["min"] = _to_builtin(da.min().values)
            info["max"] = _to_builtin(da.max().values)
        elif da.size > 0 and np.issubdtype(da.dtype, np.datetime64):
            info["min"] = str(da.values.min())
            info["max"] = str(da.values.max())
        coordinates[str(name)] = info

    variables: dict[str, dict[str, Any]] = {}
    for name, da in ds.data_vars.items():
        variables[str(name)] = {
            "dims": tuple(da.dims),
            "shape": tuple(int(s) for s in da.shape),
            "dtype": str(da.dtype),
            "standard_name": da.attrs.get("standard_name"),
            "long_name": da.attrs.get("long_name"),
            "units": da.attrs.get("units"),
        }

    return InspectionReport(
        path=str(path) if path is not None else "",
        dimensions=dimensions,
        coordinates=coordinates,
        variables=variables,
        global_attrs=dict(ds.attrs),
    )


def open_and_inspect(
    path: str | Path,
    *,
    print_report: bool = True,
) -> tuple[xr.Dataset, InspectionReport]:
    """Open a NetCDF file and print dims/coords/variables."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NetCDF file not found: {path}")

    ds = xr.open_dataset(path)
    report = inspect_dataset(ds, path=path)
    if print_report:
        print(report.format())
        if ds.attrs:
            print("Global attributes:")
            for key, value in list(ds.attrs.items())[:20]:
                print(f"  {key}: {value}")
    return ds, report


def resolve_variables(
    ds: xr.Dataset,
    config: dict[str, Any] | None = None,
    *,
    print_mapping: bool = True,
) -> dict[str, VariableResolution]:
    """Map canonical OceanEmbed names onto variables actually present in `ds`."""
    if config is None:
        config = load_config()
    specs = config["surface_inputs"]
    results: dict[str, VariableResolution] = {}

    for canonical, spec in specs.items():
        required = bool(spec.get("required", True))
        candidates: list[str] = list(spec.get("candidates", []))
        standard_names: list[str] = list(spec.get("standard_names", []))

        source_name, matched_by = _match_variable(ds, candidates, standard_names)
        results[canonical] = VariableResolution(
            canonical=canonical,
            required=required,
            found=source_name is not None,
            source_name=source_name,
            matched_by=matched_by,
            notes="" if source_name else "not found in this file",
        )

    if print_mapping:
        print("Variable mapping (canonical -> file variable):")
        for canonical, res in results.items():
            status = "REQUIRED" if res.required else "optional"
            if res.found:
                print(
                    f"  {canonical:10s} <- {res.source_name} "
                    f"[{res.matched_by}] ({status})"
                )
            else:
                print(f"  {canonical:10s} <- MISSING ({status})")

    missing_required = [
        res.canonical for res in results.values() if res.required and not res.found
    ]
    if missing_required:
        raise KeyError(
            "Required surface variables were not found in this NetCDF: "
            f"{missing_required}. Inspect the file and update configs/default.yaml."
        )
    return results


def load_ocean_dataset(
    path: str | Path,
    config: dict[str, Any] | None = None,
    *,
    print_report: bool = True,
) -> LoadedDataset:
    """Open, inspect, and resolve canonical variable names."""
    if config is None:
        config = load_config()
    ds, report = open_and_inspect(path, print_report=print_report)
    mapping = resolve_variables(ds, config, print_mapping=print_report)
    return LoadedDataset(dataset=ds, report=report, mapping=mapping)


def rename_to_canonical(ds: xr.Dataset, mapping: dict[str, VariableResolution]) -> xr.Dataset:
    """Keep only mapped data variables, renamed to canonical OceanEmbed names."""
    rename = {
        res.source_name: res.canonical
        for res in mapping.values()
        if res.found and res.source_name is not None
    }
    keep = list(rename.keys())
    subset = ds[keep].rename(rename)
    return subset


def _match_variable(
    ds: xr.Dataset,
    candidates: Iterable[str],
    standard_names: Iterable[str],
) -> tuple[str | None, str | None]:
    data_names = set(ds.data_vars)
    for name in candidates:
        if name in data_names:
            return name, "candidate_name"

    wanted = {s.lower() for s in standard_names}
    for name, da in ds.data_vars.items():
        std = str(da.attrs.get("standard_name", "")).lower()
        if std and std in wanted:
            return str(name), "standard_name"
    return None, None


def _to_builtin(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value
