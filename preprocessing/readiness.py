"""Verify whether local files satisfy the OceanEmbed PoC data contract.

This module only inspects metadata and coordinate values. It never fills
missing variables, fabricates dates/depths, or downloads data. Its purpose is
to stop model work early when the local scientific inputs are incomplete.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import xarray as xr

from preprocessing.config import load_config

DataRole = Literal["glorys", "wind", "argo"]


@dataclass(frozen=True)
class ReadinessIssue:
    """One transparent pass/fail finding from the local-data audit."""

    severity: Literal["error", "warning"]
    code: str
    message: str


@dataclass
class FileAudit:
    """Metadata needed for cross-file readiness checks."""

    path: str
    role: DataRole
    dates: set[np.datetime64] = field(default_factory=set)
    latitude_range: tuple[float, float] | None = None
    longitude_range: tuple[float, float] | None = None
    depth_range: tuple[float, float] | None = None
    depth_count: int = 0
    variables: dict[str, str] = field(default_factory=dict)
    issues: list[ReadinessIssue] = field(default_factory=list)


@dataclass
class ReadinessReport:
    """Separate model and independent-validation gates."""

    audits: list[FileAudit]
    model_issues: list[ReadinessIssue]
    validation_issues: list[ReadinessIssue]
    longest_aligned_run_days: int

    @property
    def model_ready(self) -> bool:
        return not any(issue.severity == "error" for issue in self.model_issues)

    @property
    def validation_ready(self) -> bool:
        return self.model_ready and not any(
            issue.severity == "error" for issue in self.validation_issues
        )

    def format(self) -> str:
        lines = [
            "OceanEmbed PoC data readiness",
            f"  model_ready: {self.model_ready}",
            f"  validation_ready: {self.validation_ready}",
            f"  longest GLORYS/wind aligned daily run: {self.longest_aligned_run_days}",
            "Files:",
        ]
        for audit in self.audits:
            lines.append(
                f"  [{audit.role}] {audit.path}: dates={len(audit.dates)}, "
                f"depth_levels={audit.depth_count}, variables={audit.variables}"
            )
        lines.append("Model gate findings:")
        if self.model_issues:
            lines.extend(_format_issue(issue) for issue in self.model_issues)
        else:
            lines.append("  none")
        lines.append("Validation gate findings:")
        if self.validation_issues:
            lines.extend(_format_issue(issue) for issue in self.validation_issues)
        else:
            lines.append("  none")
        return "\n".join(lines)


def audit_file(
    path: str | Path,
    role: DataRole,
    config: dict[str, Any] | None = None,
) -> FileAudit:
    """Inspect one NetCDF file for its declared role without loading arrays."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    with xr.open_dataset(path) as dataset:
        return audit_dataset(dataset, role, config, source_label=str(path))


def audit_dataset(
    dataset: xr.Dataset,
    role: DataRole,
    config: dict[str, Any] | None = None,
    *,
    source_label: str = "<in-memory>",
) -> FileAudit:
    """Inspect an opened dataset; exposed for tests and notebook use."""
    if config is None:
        config = load_config()
    coordinates = config["poc_readiness"]["coordinate_candidates"]
    audit = FileAudit(path=source_label, role=role)

    time_name = _find_coordinate(dataset, coordinates["time"])
    if time_name is None:
        audit.issues.append(_error("missing_time", "No supported time coordinate was found."))
    else:
        audit.dates = _daily_dates(dataset[time_name].values)

    for key in ("latitude", "longitude"):
        coord_name = _find_coordinate(dataset, coordinates[key])
        if coord_name is None:
            audit.issues.append(_error(f"missing_{key}", f"No supported {key} coordinate was found."))
            continue
        values = np.asarray(dataset[coord_name].values)
        if values.ndim != 1:
            audit.issues.append(_error(f"non_1d_{key}", f"{coord_name!r} is not a 1-D coordinate."))
            continue
        value_range = (float(np.nanmin(values)), float(np.nanmax(values)))
        if key == "latitude":
            audit.latitude_range = value_range
        else:
            audit.longitude_range = value_range

    if role == "glorys":
        temperature = config["targets"]["temperature"]
        resolved = _find_variable(
            dataset,
            temperature["candidates"],
            temperature["standard_names"],
        )
        if resolved is None:
            audit.issues.append(_error("missing_thetao", "No temperature target variable was found."))
        else:
            audit.variables["temperature_target"] = resolved
        for canonical in ("sst", "sss", "ssh", "current_u", "current_v"):
            resolved = _find_variable_from_spec(dataset, config["surface_inputs"][canonical])
            if resolved is None:
                audit.issues.append(_error("missing_" + canonical, f"Missing GLORYS field: {canonical}."))
            else:
                audit.variables[canonical] = resolved
        _record_depth_coverage(dataset, audit, coordinates["depth"])
    elif role == "wind":
        for canonical in ("wind_u", "wind_v"):
            resolved = _find_variable_from_spec(dataset, config["surface_inputs"][canonical])
            if resolved is None:
                audit.issues.append(_error("missing_" + canonical, f"Missing wind field: {canonical}."))
            else:
                audit.variables[canonical] = resolved
    else:
        temperature = config["targets"]["temperature"]
        resolved = _find_variable(dataset, temperature["candidates"], temperature["standard_names"])
        if resolved is None:
            audit.issues.append(_error("missing_argo_temperature", "No ARGO temperature variable was found."))
        else:
            audit.variables["argo_temperature"] = resolved

    return audit


def assess_readiness(
    *,
    glorys_files: Iterable[str | Path],
    wind_files: Iterable[str | Path],
    argo_files: Iterable[str | Path] = (),
    config: dict[str, Any] | None = None,
) -> ReadinessReport:
    """Assess file groups against the model and ARGO-validation gates."""
    if config is None:
        config = load_config()
    glorys = [audit_file(path, "glorys", config) for path in glorys_files]
    winds = [audit_file(path, "wind", config) for path in wind_files]
    argos = [audit_file(path, "argo", config) for path in argo_files]
    audits = glorys + winds + argos
    model_issues = [issue for audit in glorys + winds for issue in audit.issues]
    validation_issues = [issue for audit in argos for issue in audit.issues]

    if not glorys:
        model_issues.append(_error("no_glorys_files", "No GLORYS files were supplied."))
    if not winds:
        model_issues.append(_error("no_wind_files", "No wind files were supplied."))
    if not argos:
        validation_issues.append(_error("no_argo_files", "No independent ARGO files were supplied."))

    if glorys:
        _check_region_coverage(glorys, config, model_issues)
        _check_depth_coverage(glorys, config, model_issues)
    if winds:
        _check_region_coverage(winds, config, model_issues)

    aligned_dates = _all_dates(glorys) & _all_dates(winds)
    longest_run = _longest_daily_run(aligned_dates)
    minimum_days = int(config["poc_readiness"]["minimum_consecutive_days"])
    if longest_run < minimum_days:
        model_issues.append(
            _error(
                "too_few_aligned_days",
                f"Need {minimum_days} consecutive GLORYS/wind days; found {longest_run}.",
            )
        )
    return ReadinessReport(audits, model_issues, validation_issues, longest_run)


def _record_depth_coverage(dataset: xr.Dataset, audit: FileAudit, candidates: Iterable[str]) -> None:
    depth_name = _find_coordinate(dataset, candidates)
    if depth_name is None:
        audit.issues.append(_error("missing_depth", "No supported depth coordinate was found."))
        return
    values = np.asarray(dataset[depth_name].values, dtype=float)
    if values.ndim != 1:
        audit.issues.append(_error("non_1d_depth", f"{depth_name!r} is not a 1-D depth coordinate."))
        return
    audit.depth_count = int(values.size)
    audit.depth_range = (float(np.nanmin(values)), float(np.nanmax(values)))


def _check_region_coverage(
    audits: list[FileAudit], config: dict[str, Any], issues: list[ReadinessIssue]
) -> None:
    region = config["region"]
    for audit in audits:
        if audit.latitude_range is None or audit.longitude_range is None:
            continue
        lat_min, lat_max = audit.latitude_range
        lon_min, lon_max = audit.longitude_range
        if lat_min > region["lat_min"] or lat_max < region["lat_max"]:
            issues.append(_error("insufficient_latitude_coverage", f"{audit.path} does not cover 5–30°N."))
        if lon_min > region["lon_min"] or lon_max < region["lon_max"]:
            issues.append(_error("insufficient_longitude_coverage", f"{audit.path} does not cover 45–105°E."))


def _check_depth_coverage(
    audits: list[FileAudit], config: dict[str, Any], issues: list[ReadinessIssue]
) -> None:
    rules = config["poc_readiness"]
    required = rules["required_depth_coverage_metres"]
    for audit in audits:
        if audit.depth_range is None:
            continue
        depth_min, depth_max = audit.depth_range
        if audit.depth_count < int(rules["minimum_native_depth_levels"]):
            issues.append(_error("too_few_depth_levels", f"{audit.path} has {audit.depth_count} depth level(s); need at least {rules['minimum_native_depth_levels']}."))
        if depth_min > float(required["min_surface_depth"]):
            issues.append(_error("missing_near_surface_level", f"{audit.path} starts at {depth_min:g} m; need a level at or above {required['min_surface_depth']} m."))
        if depth_max < float(required["max_target_depth"]):
            issues.append(_error("insufficient_depth_coverage", f"{audit.path} ends at {depth_max:g} m; need coverage to {required['max_target_depth']} m."))


def _find_coordinate(dataset: xr.Dataset, candidates: Iterable[str]) -> str | None:
    for name in candidates:
        if name in dataset.coords:
            return str(name)
    for name, variable in dataset.coords.items():
        standard_name = str(variable.attrs.get("standard_name", "")).lower()
        axis = str(variable.attrs.get("axis", "")).upper()
        if standard_name == "depth" or axis == "Z":
            return str(name)
    return None


def _find_variable(dataset: xr.Dataset, candidates: Iterable[str], standard_names: Iterable[str]) -> str | None:
    for name in candidates:
        if name in dataset.data_vars:
            return str(name)
    wanted = {name.lower() for name in standard_names}
    for name, variable in dataset.data_vars.items():
        if str(variable.attrs.get("standard_name", "")).lower() in wanted:
            return str(name)
    return None


def _find_variable_from_spec(dataset: xr.Dataset, spec: dict[str, Any]) -> str | None:
    return _find_variable(dataset, spec.get("candidates", []), spec.get("standard_names", []))


def _daily_dates(values: np.ndarray) -> set[np.datetime64]:
    if values.size == 0:
        return set()
    return {np.datetime64(value, "D") for value in np.asarray(values).reshape(-1)}


def _all_dates(audits: list[FileAudit]) -> set[np.datetime64]:
    dates: set[np.datetime64] = set()
    for audit in audits:
        dates.update(audit.dates)
    return dates


def _longest_daily_run(dates: set[np.datetime64]) -> int:
    if not dates:
        return 0
    ordered = sorted(dates)
    best = current = 1
    for previous, following in zip(ordered, ordered[1:]):
        if following - previous == np.timedelta64(1, "D"):
            current += 1
        else:
            current = 1
        best = max(best, current)
    return best


def _error(code: str, message: str) -> ReadinessIssue:
    return ReadinessIssue("error", code, message)


def _format_issue(issue: ReadinessIssue) -> str:
    return f"  {issue.severity.upper()} [{issue.code}]: {issue.message}"
