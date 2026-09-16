"""Export dated OceanEmbed predictions for later independent ARGO collocation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader

from datasets.ocean_dataset import OceanDataBundle, OceanDataset
from models.oceanembed import OceanEmbed
from training.checkpoint import load_checkpoint


def export_prediction_cube(
    *,
    checkpoint_path: str | Path,
    dataset: OceanDataset,
    bundle: OceanDataBundle,
    output_path: str | Path,
    device: torch.device,
    run_label: str,
) -> Path:
    """Write held-out model predictions with coordinates and masks.

    The file deliberately includes GLORYS reference temperatures only as a
    smoke-test diagnostic. ARGO remains the independent validation source.
    Shapes written are (sample_time, depth, latitude, longitude).
    """
    # Read architecture metadata before constructing the model so a future
    # configuration change cannot silently make this export incompatible.
    payload = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    metadata = dict(payload["metadata"])
    model = OceanEmbed(
        depths_metres=bundle.target_depths.tolist(),
        spatial_channels=tuple(metadata["spatial_channels"]),
        convlstm_hidden_channels=int(metadata["convlstm_hidden_channels"]),
        depth_embedding_channels=int(metadata["depth_embedding_channels"]),
        residual_to_sst=bool(metadata.get("residual_to_sst", True)),
    ).to(device)
    metadata = load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    dates: list[np.datetime64] = []
    with torch.no_grad():
        for batch in loader:
            prediction = model(batch["inputs"].to(device), batch["input_mask"].to(device))
            predictions.append(prediction[0].cpu().numpy().astype(np.float32))
            targets.append(batch["target"][0].numpy().astype(np.float32))
            masks.append(batch["target_mask"][0].numpy().astype(bool))
            dates.append(np.datetime64(batch["end_time"][0], "D"))

    prediction_array = np.stack(predictions)
    target_array = np.stack(targets)
    mask_array = np.stack(masks)
    target_array = np.where(mask_array, target_array, np.nan)
    dataset_out = xr.Dataset(
        data_vars={
            "oceanembed_temperature": (("time", "depth", "latitude", "longitude"), prediction_array),
            "glorys_reference_temperature": (("time", "depth", "latitude", "longitude"), target_array),
            "glorys_reference_mask": (("time", "depth", "latitude", "longitude"), mask_array.astype(np.int8)),
        },
        coords={"time": np.asarray(dates), "depth": bundle.target_depths, "latitude": bundle.latitude, "longitude": bundle.longitude},
        attrs={
            "label": run_label,
            "warning": "DEMO / DATA-LIMITED. GLORYS is a training reference/reanalysis, not independent validation.",
            "checkpoint_epoch": int(metadata["epoch"]),
            "temporal_window_days": int(metadata["temporal_window"]),
            "prediction_units": "degrees_Celsius",
        },
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset_out.to_netcdf(output)
    return output
