# OceanEmbed

Satellite embedding-based deep learning framework for reconstructing **subsurface ocean temperature** in the North Indian Ocean from **surface observations**, trained against **GLORYS reanalysis** (not treated as perfect ground truth). Independent validation against **gridded ARGO** is planned and is **not mixed into training**.

This repository is being built in stages. The end-to-end temporal model smoke
test is implemented, but its outputs are **DEMO / DATA-LIMITED**, not a
scientific result or independent validation.

## Study region (configurable)

- Latitude 5°N–30°N, longitude 45°E–105°E
- Target grid: 0.25° (`H=101`, `W=241`)
- Time: daily (`T` is a config value, not hard-coded in a model yet)

## What is implemented now

- NetCDF **inspection** (prints dimensions, coordinates, variable names)
- Canonical variable mapping from `configs/default.yaml` (no silent name guessing in code)
- North Indian Ocean subset
- Daily time alignment hook
- Regrid to 0.25° (`xarray` linear interpolation)
- Land/missing mask from NaNs (land stays NaN)
- Ocean-only z-score normalization
- Channel stack of shape **`(T, C, H, W)`**

## What the sample file actually contains

File: `data/cmems_mod_glo_phy_my_0.083deg_P1D-m_1788172075365.nc`

| Fact | Value |
|---|---|
| Product | CMEMS `GLOBAL_MULTIYEAR_PHY_001_030` / MERCATOR **GLORYS12V1** |
| Native grid | global, ~0.083° |
| Time | **1 day** (`2026-06-23`) |
| Depth | **1 level** (~0.49 m) — **not** the 15-level profile |
| Mapped inputs | `thetao→sst`, `so→sss`, `zos→ssh`, `uo→current_u`, `vo→current_v` |
| Missing | **surface winds**; **subsurface temperature at 5–1000 m**; ARGO |

`zos` is **sea surface height above the geoid**, not SLA. Winds are **omitted** from `C` rather than filled with zeros.

This file is enough to test **surface preprocessing**. It is **not** enough to train the depth-conditioned model.

## Setup

```bash
cd ~/Projects/oceanembed
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Commands

Inspect a NetCDF (always do this for a new file):

```bash
.venv/bin/python scripts/inspect_netcdf.py data/cmems_mod_glo_phy_my_0.083deg_P1D-m_1788172075365.nc
```

Run preprocessing on that sample:

```bash
.venv/bin/python scripts/preprocess_sample.py data/cmems_mod_glo_phy_my_0.083deg_P1D-m_1788172075365.nc
```

Tests:

```bash
.venv/bin/python -m pytest tests/ -q
```

## PoC data-readiness gate

Before creating a PyTorch model, run the read-only checker on the files you
download. It verifies field discovery, regional coverage, 30 consecutive
aligned GLORYS/wind dates, multi-level depth coverage from near surface to
1000 m, and separate ARGO availability.

```bash
.venv/bin/python scripts/check_data_readiness.py \
  --glorys data/glorys_*.nc \
  --winds data/era5_winds_*.nc \
  --argo data/argo_*.nc
```

It exits successfully only when both model and independent-validation gates
pass. It never fills missing data or treats the current one-day, surface-only
sample as model-ready.

## Acquire the configured PoC subset

The acquisition manifest is [configs/poc_acquisition.yaml](configs/poc_acquisition.yaml).
It requests 30 daily dates (2024-01-01 through 2024-01-30), the NIO region,
GLORYS fields through 1000 m, and ERA5 daily-mean 10 m winds. First inspect
the requests without network access:

```bash
.venv/bin/python scripts/acquire_poc_data.py
```

To execute, authenticate using your own provider accounts. Do not paste any
password or token into project files or chat:

```bash
copernicusmarine login
.venv/bin/python scripts/acquire_poc_data.py --execute
```

After downloading, inspect every new file and run the readiness gate. ARGO is
still obtained separately for independent validation.

CDS can return separate U/V NetCDF files inside a ZIP archive even if its
download filename ends in `.nc`. Prepare that archive before readiness checks:

```bash
.venv/bin/python scripts/prepare_era5_winds.py \
  data/era5_winds_nio_20240101_20240130.nc
```

## Tensor shapes (this stage)

After preprocessing the current sample:

- `channels`: `(T=1, C=5, H=101, W=241)`
- `C` names: `sst, sss, ssh, current_u, current_v`
- Target profile ` (15, H, W) ` is **not produced** because the file has no 15 depths

Intended later:

- `X ∈ R^(T × 7 × H × W)` when winds and a time window exist
- `Y ∈ R^(15 × H × W)` from multi-level GLORYS temperature

## PyTorch dataset stage

With the downloaded 30-day GLORYS and ERA5 PoC files, the implemented dataset
uses a configurable five-day window (`T=5`) and creates 26 overlapping samples.

- `inputs`: `(T, C, H, W) = (5, 7, 101, 241)`
- `target`: `(D, H, W) = (15, 101, 241)` for the final day in the window
- `input_mask`: `(T, 1, H, W)`
- `target_mask`: `(D, H, W)`

The seven input channels are `sst`, `sss`, `ssh`, `current_u`, `current_v`,
`wind_u`, and `wind_v`. The 15 target layers are vertically interpolated from
GLORYS only inside its native depth coverage. The 0 m layer uses GLORYS's
nearest 0.49 m level under a configured 1 m tolerance. Invalid/land values are
filled with zero only for tensor transport and always have an accompanying mask.

Verify the real tensor build:

```bash
.venv/bin/python scripts/inspect_ocean_dataset.py
```

## Temporal model smoke test

The model core is explicitly temporal: a shared spatial CNN creates a feature
map for each of five days, then a ConvLSTM processes those maps in date order.
Its final latent ocean state is decoded by **one shared depth-conditioned
decoder** into all 15 temperatures. It is not a plain 2D CNN and does not use
15 independent heads.

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python scripts/train_smoke_test.py
```

Artifacts are stored in `data/processed/smoke_test/` and are always labelled
`DEMO / DATA-LIMITED`. The 30-day dataset is sufficient for an end-to-end
pipeline check only; expand to at least 180 days before interpreting metrics.
The default local PoC run is 50 epochs over the same 17 training windows; this
improves convergence but does not turn the 30-day subset into a robust model.

The best held-out-window predictions are additionally written to
`data/processed/smoke_test/validation_predictions.nc`. Its explicit contract
is `(time=5, depth=15, latitude=101, longitude=241)` and it carries the
prediction, GLORYS diagnostic reference/mask, dates and geographic
coordinates. It is the model-side input for ARGO collocation; it is not itself
ARGO validation.

## Independent ARGO validation: next required input

No ARGO profile data has been downloaded into this repository yet, so no ARGO
metric or profile plot can honestly be generated. Obtain a small selection of
official profile NetCDF files within 5–30°N, 45–105°E and dates close to the
five exported held-out dates (2024-01-26 to 2024-01-30). The files must include
profile time, latitude, longitude, pressure/depth, temperature and temperature
QC flags. Prefer delayed-mode adjusted fields when available; otherwise record
the real-time data mode and QC policy. The official [Argo profile-file guide](https://argo.ucsd.edu/data/how-to-use-argo-files/)
and [GDAC access page](https://argo.ucsd.edu/data/data-from-gdacs/) describe
the profile NetCDF and selection workflow.

Before mapping any new ARGO file, inspect it—this prints its actual dimensions,
variable names and candidate fields rather than assuming them:

```bash
.venv/bin/python scripts/inspect_argo_profiles.py data/argo_profile_file.nc
```

## Local dashboard prototype

The Streamlit dashboard reads the saved model prediction cube, independent
ARGO-validation artifacts, and the real GLORYS/ERA5 source fields. It does not
retrain the model. It provides a date selector, the 15-depth selector, maps of
prediction/SST/SSH/currents/winds, ARGO locations and profile comparison,
metrics, short Ocean Insights explanations, and honest alert/security status.

```bash
.venv/bin/streamlit run dashboard/app.py
```

Create an integrity manifest for the dashboard artifacts before presenting:

```bash
.venv/bin/python scripts/create_artifact_manifest.py
```

Use [docs/HACKATHON_DEMO.md](docs/HACKATHON_DEMO.md) as the concise presenter
flow and scientific-wording checklist.

It is intentionally local and labelled **DEMO / DATA-LIMITED**. The alert tab
implements a configurable short-reference anomaly monitor using the Jan 1–21
GLORYS mean, with NORMAL/MODERATE/HIGH thresholds. This is **not** a long-term
climatological anomaly or official disaster warning. Official disaster warnings
are not connected. Authentication and role-based access are planned for
deployment; they are not claimed as implemented in this local dashboard.

## Scientific caveats (read these)

1. GLORYS is a **reanalysis / training reference**, not in-situ truth.
2. Near-surface `thetao` is **not** satellite SST; it is the reanalysis first level.
3. Linear interpolation is **not** conservative remapping (xESMF can replace it later).
4. A 2D CNN is **not** a temporal model. `T=1` here, so no temporal encoder yet.
5. Do not treat NaNs as zeros. Land is masked.

## Priority order (not implemented)

The model pipeline takes precedence over web-app development:

1. Obtain and inspect multi-day, multi-depth data plus winds.
2. Build the PyTorch dataset and shape/masking tests.
3. Build the spatial CNN, ConvLSTM temporal module, depth-conditioned decoder, and masked loss.
4. Train against GLORYS as a reanalysis reference; save checkpoints and prediction artifacts.
5. Validate separately against ARGO.
6. Build the dashboard as a viewer of saved, versioned prediction and validation artifacts.
7. Add alerts, educational content, and security controls.

## Data still needed

To go beyond this stage you will need:

1. **Multi-level GLORYS** `thetao` at the 15 target depths (or nearby native depths to interpolate in z)
2. **Multiple days** if we want `T > 1`
3. **Surface winds** (e.g. ERA5 or a CMEMS wind product)
4. Later: **gridded ARGO** for independent validation
