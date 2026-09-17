from __future__ import annotations

import csv
from io import StringIO

import numpy as np

from dashboard.research_portal import research_csv, subset_field, subset_indices


def test_subset_limits_export_to_requested_grid_area() -> None:
    latitude = np.array([10.0, 10.25, 10.5], dtype=np.float32)
    longitude = np.array([80.0, 80.25, 80.5], dtype=np.float32)
    field = np.arange(9, dtype=np.float32).reshape(3, 3)

    rows, columns = subset_indices(latitude, longitude, (10.25, 10.5), (80.0, 80.25))

    assert np.array_equal(rows, [1, 2])
    assert np.array_equal(columns, [0, 1])
    assert np.array_equal(subset_field(field, rows, columns), [[3.0, 4.0], [6.0, 7.0]])


def test_research_csv_contains_selection_provenance_and_validity() -> None:
    output = research_csv(
        field=np.array([[25.0, np.nan]], dtype=np.float32),
        latitude=np.array([12.0], dtype=np.float32),
        longitude=np.array([82.0, 82.25], dtype=np.float32),
        variable="Reconstructed subsurface temperature",
        units="°C",
        source="OceanEmbed reconstruction",
        date="2024-06-29",
        depth_metres=50.0,
        run_label="EXPERIMENT / DATA-LIMITED",
    )
    rows = list(csv.DictReader(StringIO(output.decode("utf-8"))))

    assert len(rows) == 2
    assert rows[0]["depth_m"] == "50.0"
    assert rows[0]["source"] == "OceanEmbed reconstruction"
    assert rows[0]["valid"] == "True"
    assert rows[1]["value"] == ""
    assert rows[1]["valid"] == "False"
