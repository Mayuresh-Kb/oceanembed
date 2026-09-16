"""Configurable, explicitly prototype anomaly-monitoring helpers."""

from __future__ import annotations


from typing import Any

import numpy as np


def short_reference_alert(prediction: np.ndarray, reference_mean: np.ndarray, config: dict[str, Any]) -> dict[str, float | str]:
    """Classify a map against a short reference period for the demo only.

    This is anomaly monitoring, not an official disaster-warning service and
    not a climatological marine-heatwave classification.
    """
    anomaly = prediction - reference_mean
    valid = np.isfinite(anomaly)
    if not valid.any():
        return {"status": "UNAVAILABLE", "message": "No common valid ocean cells between prediction and reference."}
    magnitude = np.abs(anomaly[valid])
    moderate = float(config["moderate_abs_temperature_anomaly_c"])
    high = float(config["high_abs_temperature_anomaly_c"])
    required_fraction = float(config["minimum_affected_ocean_fraction"])
    moderate_fraction = float(np.mean(magnitude >= moderate))
    high_fraction = float(np.mean(magnitude >= high))
    status = "HIGH ANOMALY" if high_fraction >= required_fraction else "MODERATE ANOMALY" if moderate_fraction >= required_fraction else "NORMAL"
    return {
        "status": status,
        "mean_anomaly_c": float(np.mean(anomaly[valid])),
        "maximum_absolute_anomaly_c": float(np.max(magnitude)),
        "moderate_area_fraction": moderate_fraction,
        "high_area_fraction": high_fraction,
        "message": "DEMO: compared with the configured 21-day GLORYS reference mean, not a long-term climatology or official hazard warning.",
    }


def explain_alert_for_people(alert: dict[str, float | str], depth_metres: float, config: dict[str, Any]) -> list[str]:
    """Turn a technical alert result into short, non-specialist statements."""
    if alert["status"] == "UNAVAILABLE":
        return [str(alert["message"])]
    mean = float(alert["mean_anomaly_c"])
    direction = "warmer" if mean >= 0 else "cooler"
    moderate_percent = 100 * float(alert["moderate_area_fraction"])
    high_percent = 100 * float(alert["high_area_fraction"])
    required_percent = 100 * float(config["minimum_affected_ocean_fraction"])
    moderate = float(config["moderate_abs_temperature_anomaly_c"])
    high = float(config["high_abs_temperature_anomaly_c"])
    return [
        f"At {depth_metres:g} m, the predicted water is on average {abs(mean):.2f}°C {direction} than the short reference average.",
        f"About {moderate_percent:.1f}% of the valid ocean area differs by at least {moderate:g}°C. The dashboard needs {required_percent:.0f}% or more to call this a moderate anomaly.",
        f"About {high_percent:.1f}% differs by at least {high:g}°C. A high anomaly requires {required_percent:.0f}% or more at that stronger threshold.",
        "This is a temperature-change indicator for the demo. It is not a warning of a cyclone, tsunami, flood or other official disaster.",
    ]
