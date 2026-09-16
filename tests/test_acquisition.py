from __future__ import annotations

from pathlib import Path

from preprocessing.acquisition import (
    _has_cds_credentials,
    build_era5_request,
    build_glorys_request,
    load_acquisition_config,
)


def test_glorys_request_has_required_region_depth_and_variables() -> None:
    request = build_glorys_request(load_acquisition_config())
    assert request["variables"] == ["thetao", "so", "uo", "vo", "zos"]
    assert request["minimum_depth"] == 0.0
    assert request["maximum_depth"] == 1200.0
    assert request["minimum_latitude"] == 5.0
    assert request["maximum_longitude"] == 105.0


def test_era5_request_is_daily_mean_for_exact_30_day_period() -> None:
    dataset_id, request = build_era5_request(load_acquisition_config())
    assert dataset_id == "derived-era5-single-levels-daily-statistics"
    assert request["variable"] == ["10m_u_component_of_wind", "10m_v_component_of_wind"]
    assert request["daily_statistic"] == "daily_mean"
    assert len(request["day"]) == 30
    assert request["area"] == [30.0, 45.0, 5.0, 105.0]


def test_cds_credential_check_requires_nonempty_url_and_key(tmp_path: Path) -> None:
    path = tmp_path / ".cdsapirc"
    assert not _has_cds_credentials(path)
    path.write_text("url: https://cds.climate.copernicus.eu/api\nkey: \n", encoding="utf-8")
    assert not _has_cds_credentials(path)
    path.write_text("url: https://cds.climate.copernicus.eu/api\nkey: private-token\n", encoding="utf-8")
    assert _has_cds_credentials(path)
