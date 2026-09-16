from __future__ import annotations

import pytest
import torch
import xarray as xr
from datetime import date
from netCDF4 import Dataset

from models.depth_decoder import DepthConditionedDecoder
from models.oceanembed import OceanEmbed
from training.checkpoint import load_checkpoint, save_checkpoint
from training.losses import masked_mse
from validation.argo_validation import index_argo_profiles, inspect_argo_profile_file


DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]


def test_complete_model_shape_and_depth_count() -> None:
    model = OceanEmbed(DEPTHS)
    inputs = torch.randn(1, 5, 7, 101, 241)
    mask = torch.ones(1, 5, 1, 101, 241, dtype=torch.bool)
    output = model(inputs, mask)
    assert tuple(output.shape) == (1, 15, 101, 241)


def test_temporal_order_changes_convlstm_prediction() -> None:
    torch.manual_seed(7)
    model = OceanEmbed(DEPTHS).eval()
    inputs = torch.randn(1, 5, 7, 31, 41)
    forward = model(inputs)
    reversed_days = model(torch.flip(inputs, dims=[1]))
    assert not torch.allclose(forward, reversed_days)


def test_decoder_is_shared_across_all_depths() -> None:
    decoder = DepthConditionedDecoder(DEPTHS)
    output = decoder(torch.randn(2, 32, 26, 61), (101, 241))
    assert tuple(output.shape) == (2, 15, 101, 241)
    assert decoder.decode[0].out_channels == 32


def test_masked_mse_ignores_invalid_fill_values_and_rejects_empty_mask() -> None:
    prediction = torch.tensor([[[[2.0, 1000.0]]]])
    target = torch.tensor([[[[1.0, 0.0]]]])
    mask = torch.tensor([[[[True, False]]]])
    assert masked_mse(prediction, target, mask).item() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="empty"):
        masked_mse(prediction, target, torch.zeros_like(mask))


def test_checkpoint_round_trip(tmp_path) -> None:
    model = OceanEmbed(DEPTHS)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, {"label": "DEMO / DATA-LIMITED"})
    restored = OceanEmbed(DEPTHS)
    metadata = load_checkpoint(path, restored)
    assert metadata["label"] == "DEMO / DATA-LIMITED"
    for expected, actual in zip(model.parameters(), restored.parameters()):
        assert torch.equal(expected, actual)


def test_argo_inspection_reports_actual_candidate_fields(tmp_path) -> None:
    profile = xr.Dataset(
        data_vars={
            "TEMP_ADJUSTED": (("N_PROF", "N_LEVELS"), [[25.0, 20.0]]),
            "TEMP_ADJUSTED_QC": (("N_PROF", "N_LEVELS"), [[b"1", b"1"]]),
            "PRES_ADJUSTED": (("N_PROF", "N_LEVELS"), [[5.0, 100.0]]),
            "JULD": ("N_PROF", [27000.0]),
            "LATITUDE": ("N_PROF", [15.0]),
            "LONGITUDE": ("N_PROF", [70.0]),
        }
    )
    path = tmp_path / "argo_profile.nc"
    profile.to_netcdf(path)
    report = inspect_argo_profile_file(path)
    assert report["ready_for_manual_mapping"] is True
    assert report["candidate_matches"]["temperature"] == ["TEMP_ADJUSTED"]


def test_argo_index_selects_only_nearby_profiles(tmp_path) -> None:
    path = tmp_path / "D9999999_001.nc"
    with Dataset(path, "w") as dataset:
        dataset.createDimension("N_PROF", 1)
        for name, values in {"JULD": [27054.0], "LATITUDE": [15.0], "LONGITUDE": [70.0]}.items():
            variable = dataset.createVariable(name, "f8", ("N_PROF",))
            variable[:] = values
            if name == "JULD":
                variable.units = "days since 1950-01-01 00:00:00 UTC"
        for name, value in {"DATA_MODE": b"D", "POSITION_QC": b"1", "JULD_QC": b"1"}.items():
            variable = dataset.createVariable(name, "S1", ("N_PROF",))
            variable[:] = [value]
    output = tmp_path / "candidates.csv"
    result = index_argo_profiles(tmp_path, output, candidate_dates=[date(2024, 1, 26)], max_time_difference_days=3)
    assert result["candidate_profiles"] == 1
    assert "2024-01-27" in output.read_text(encoding="utf-8")
