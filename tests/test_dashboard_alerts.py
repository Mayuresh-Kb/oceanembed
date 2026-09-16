from __future__ import annotations

import numpy as np

from dashboard.alerts import explain_alert_for_people, short_reference_alert


CONFIG = {
    "moderate_abs_temperature_anomaly_c": 1.0,
    "high_abs_temperature_anomaly_c": 2.0,
    "minimum_affected_ocean_fraction": 0.5,
}


def test_short_reference_alert_uses_configured_area_thresholds() -> None:
    reference = np.zeros((2, 2))
    high = short_reference_alert(np.full((2, 2), 2.5), reference, CONFIG)
    normal = short_reference_alert(np.full((2, 2), 0.5), reference, CONFIG)
    assert high["status"] == "HIGH ANOMALY"
    assert normal["status"] == "NORMAL"


def test_plain_language_explanation_includes_depth_and_direction() -> None:
    alert = short_reference_alert(np.full((2, 2), -1.5), np.zeros((2, 2)), CONFIG)
    explanation = explain_alert_for_people(alert, 50.0, CONFIG)
    assert "50 m" in explanation[0]
    assert "cooler" in explanation[0]
