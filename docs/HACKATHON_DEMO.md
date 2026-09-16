# OceanEmbed hackathon demo flow

1. Open the local dashboard with `streamlit run dashboard/app.py`.
2. Start with the predicted-temperature map and move the depth selector from 0 m to 1000 m. Explain that one temporal ConvLSTM model reconstructs all depths through a shared depth-conditioned decoder.
3. Switch to SST, SSH, currents and winds. State that this PoC uses GLORYS/ERA5 surface-style inputs; satellite-only operational inputs are future work.
4. Open **ARGO comparison**. Show an independent in-situ profile and the ARGO RMSE/MAE/bias panel.
5. Open **Alerts**. Explain that it is a configurable short-reference anomaly monitor, not an official disaster-warning feed.
6. Open **Ocean Insights** and **Security**. State what is implemented locally and what is planned before deployment.

## Required scientific wording

- GLORYS is the training reference/reanalysis, not perfect ground truth.
- ARGO is independent in-situ validation, not perfect ground truth.
- Current result: January 2024 North Indian Ocean, **DEMO / DATA-LIMITED** PoC.
- Do not call the alert an official cyclone, tsunami, flood or marine-hazard warning.
