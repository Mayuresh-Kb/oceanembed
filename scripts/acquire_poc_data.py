#!/usr/bin/env python3
"""Preview or execute the configured 30-day OceanEmbed data request.

The default is dry-run/preview. Use --execute only after configuring:
  1. Copernicus Marine login via `copernicusmarine login`
  2. CDS API token in the standard local .cdsapirc file
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocessing.acquisition import (  # noqa: E402
    build_era5_request,
    build_glorys_request,
    execute_era5_download,
    execute_glorys_download,
    load_acquisition_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire the configured OceanEmbed PoC subset.")
    parser.add_argument("--execute", action="store_true", help="Perform authenticated downloads.")
    parser.add_argument(
        "--only",
        choices=("glorys", "era5"),
        default=None,
        help="Execute only one provider request; preview always displays both.",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()

    acquisition = load_acquisition_config(args.config)
    glorys_request = build_glorys_request(acquisition)
    print("GLORYS request:\n" + json.dumps(glorys_request, indent=2))
    era5_dataset = era5_request = None
    if args.only != "glorys":
        era5_dataset, era5_request = build_era5_request(acquisition)
        print("\nERA5 dataset:\n" + era5_dataset)
        print("ERA5 request:\n" + json.dumps(era5_request, indent=2))
    else:
        print("\nERA5 request skipped (--only glorys). Multi-month winds are requested separately by month.")
    if not args.execute:
        print("\nPreview only. Re-run with --execute after authenticating both providers.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.only in (None, "glorys"):
        execute_glorys_download(
            glorys_request,
            output_dir=args.output_dir,
            output_filename=acquisition["glorys"]["output_filename"],
        )
    if args.only in (None, "era5"):
        if era5_dataset is None or era5_request is None:
            raise RuntimeError("ERA5 request was not built.")
        try:
            execute_era5_download(
                era5_dataset,
                era5_request,
                output_path=args.output_dir / acquisition["era5_winds"]["output_filename"],
            )
        except RuntimeError as error:
            raise SystemExit(f"ERA5 download not started: {error}") from error
        except requests.HTTPError as error:
            if "required licences not accepted" in str(error).lower():
                raise SystemExit(
                    "ERA5 download not started: accept the required licence while "
                    "signed in to the same CDS account, then rerun this command:\n"
                    "https://cds.climate.copernicus.eu/datasets/"
                    "derived-era5-single-levels-daily-statistics?tab=download#manage-licences"
                ) from error
            raise
    print("\nDownloads requested. Run scripts/check_data_readiness.py on the resulting files.")


if __name__ == "__main__":
    main()
