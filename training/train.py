"""Run a labelled, data-limited OceanEmbed training smoke test."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.ocean_dataset import OceanDataset, fit_input_normalization, load_poc_bundle
from models.oceanembed import OceanEmbed
from preprocessing.config import load_config
from training.checkpoint import save_checkpoint
from training.losses import masked_mse
from validation.prediction_export import export_prediction_cube


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_smoke_test(
    *,
    epochs: int | None = None,
    output_dir: str | Path | None = None,
    training_key: str = "training",
    glorys_paths: str | Path | list[str | Path] | None = None,
    wind_paths: str | Path | list[str | Path] | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Train a labelled experimental dataset; no generalization claim is implied."""
    config = load_config(config_path)
    train_config = config[training_key]
    _set_seed(int(train_config["seed"]))
    root = Path(__file__).resolve().parents[1]
    artifacts = Path(output_dir) if output_dir else root / "data" / "processed" / "smoke_test"
    artifacts.mkdir(parents=True, exist_ok=True)
    bundle = load_poc_bundle(
        glorys_paths or root / "data" / "glorys_nio_20240101_20240130_z1200m.nc",
        wind_paths or root / "data" / "era5_winds_nio_20240101_20240130_merged.nc",
        config,
        cache_dir=artifacts / "dataset_cache",
    )
    train_end = list(train_config["train_end_day_indices"])
    validation_end = list(train_config["validation_end_day_indices"])
    stats = fit_input_normalization(bundle, range(max(train_end) + 1))
    window = int(config["time"]["window"])
    common = dict(normalization=stats, temporal_window=window, fill_value=float(config["dataset"]["tensor_fill_value"]))
    train_dataset = OceanDataset(bundle, end_day_indices=train_end, **common)
    validation_dataset = OceanDataset(bundle, end_day_indices=validation_end, **common)
    batch_size = int(train_config["batch_size"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    device = select_device()
    model_config = config["model"]
    model = OceanEmbed(
        depths_metres=bundle.target_depths.tolist(),
        spatial_channels=tuple(model_config["spatial_channels"]),
        convlstm_hidden_channels=int(model_config["convlstm_hidden_channels"]),
        depth_embedding_channels=int(model_config["depth_embedding_channels"]),
        sst_mean=float(stats["sst"]["mean"]),
        sst_std=float(stats["sst"]["std"]),
        residual_to_sst=bool(model_config["residual_to_sst"]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_config["learning_rate"]), weight_decay=float(train_config["weight_decay"]))
    epochs = int(epochs if epochs is not None else train_config["smoke_test_epochs"])
    history: list[dict[str, float]] = []
    best_validation = float("inf")
    best_checkpoint = artifacts / "best_checkpoint.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            prediction = model(batch["inputs"].to(device), batch["input_mask"].to(device))
            loss = masked_mse(prediction, batch["target"].to(device), batch["target_mask"].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        validation_loss, baseline_loss, first_prediction = _evaluate(model, validation_loader, stats, device)
        record = {"epoch": float(epoch), "train_masked_mse": float(np.mean(train_losses)), "validation_masked_mse": validation_loss, "surface_temperature_baseline_mse": baseline_loss}
        history.append(record)
        if validation_loss < best_validation:
            best_validation = validation_loss
            metadata = {"label": train_config["label"], "epoch": epoch, "validation_masked_mse": validation_loss, "device": str(device), "temporal_window": window, "target_depths_metres": bundle.target_depths.tolist(), "spatial_channels": list(model_config["spatial_channels"]), "convlstm_hidden_channels": int(model_config["convlstm_hidden_channels"]), "depth_embedding_channels": int(model_config["depth_embedding_channels"]), "residual_to_sst": bool(model_config["residual_to_sst"]), "normalization": stats, "source": "GLORYS training reference/reanalysis; not perfect ground truth."}
            save_checkpoint(best_checkpoint, model, optimizer, metadata)
            np.savez_compressed(artifacts / "validation_prediction_sample.npz", prediction=first_prediction)

    gap_end_days = list(range(max(train_end) + 1, min(validation_end)))
    metadata = {"label": train_config["label"], "warning": "Limited-data experiment only; not independent validation or generalizable skill.", "device": str(device), "train_samples": len(train_dataset), "validation_samples": len(validation_dataset), "gap_end_days": gap_end_days, "normalization_fit_days": [0, max(train_end)], "history": history}
    (artifacts / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (artifacts / "loss_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    export_prediction_cube(
        checkpoint_path=best_checkpoint,
        dataset=validation_dataset,
        bundle=bundle,
        output_path=artifacts / "validation_predictions.nc",
        device=device,
        run_label=train_config["label"],
    )
    return {"artifacts": str(artifacts), "best_checkpoint": str(best_checkpoint), **metadata}


def _evaluate(model: OceanEmbed, loader: DataLoader, stats: dict[str, dict[str, float]], device: torch.device) -> tuple[float, float, np.ndarray]:
    model.eval()
    losses, baseline_losses = [], []
    first_prediction: np.ndarray | None = None
    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            mask = batch["target_mask"].to(device)
            target = batch["target"].to(device)
            prediction = model(inputs, batch["input_mask"].to(device))
            losses.append(float(masked_mse(prediction, target, mask).cpu()))
            # Input SST is normalized. Convert it back before this sanity baseline.
            surface_temp = inputs[:, -1, 0] * stats["sst"]["std"] + stats["sst"]["mean"]
            baseline = surface_temp[:, None].expand_as(target)
            baseline_losses.append(float(masked_mse(baseline, target, mask).cpu()))
            if first_prediction is None:
                first_prediction = prediction[0].detach().cpu().numpy()
    if first_prediction is None:
        raise RuntimeError("Validation loader was empty.")
    return float(np.mean(losses)), float(np.mean(baseline_losses)), first_prediction


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
