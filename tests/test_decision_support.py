from __future__ import annotations

import numpy as np

from dashboard.decision_support import anomaly_regions, build_decision_summary, layer_csv, previous_difference


CONFIG = {
    "moderate_abs_temperature_anomaly_c": 1.0,
    "high_abs_temperature_anomaly_c": 2.0,
    "minimum_affected_ocean_fraction": 0.25,
    "minimum_anomaly_region_cells": 2,
    "day_to_day_change_threshold_c": 0.5,
}


def test_previous_difference_handles_first_date_and_masks() -> None:
    assert previous_difference(np.ones((2, 2)), None) is None
    result = previous_difference(np.array([[2.0, np.nan]]), np.array([[1.0, 0.0]]))
    assert result[0, 0] == 1.0
    assert np.isnan(result[0, 1])


def test_anomaly_regions_filters_small_components_and_marks_high() -> None:
    anomaly = np.array([[2.5, 2.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.2]])
    regions = anomaly_regions(anomaly, np.array([10.0, 11.0, 12.0]), np.array([80.0, 81.0, 82.0]), CONFIG)
    assert len(regions) == 1
    assert regions[0]["severity"] == "HIGH ANOMALY"
    assert regions[0]["cell_count"] == 2


def test_summary_and_layer_export_keep_provenance() -> None:
    prediction = np.full((2, 2), 1.5)
    summary = build_decision_summary(prediction, np.zeros((2, 2)), np.ones((2, 2)), np.array([10.0, 11.0]), np.array([80.0, 81.0]), CONFIG)
    assert summary["alert"]["status"] == "MODERATE ANOMALY"
    assert summary["change_mean_c"] == 0.5
    text = layer_csv(layer=prediction, latitude=np.array([10.0, 11.0]), longitude=np.array([80.0, 81.0]), layer_name="Temperature", units="°C", date="2024-06-10", depth=50.0, run_label="TEST").decode("utf-8")
    assert "2024-06-10" in text and "50.0" in text and "TEST" in text
