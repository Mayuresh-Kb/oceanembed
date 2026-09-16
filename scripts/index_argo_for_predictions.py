#!/usr/bin/env python3
"""Build an ARGO candidate index from a saved OceanEmbed prediction cube.

This keeps ARGO selection independent of training: only prediction dates and
map bounds are read, never ARGO temperatures or model weights.
"""

from __future__ import annotations

import argparse
from datetime import date
import sys
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from validation.argo_validation import index_argo_profiles  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Index ARGO profiles near OceanEmbed prediction dates and map bounds.")
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-time-difference-days", type=int, default=3)
    args = parser.parse_args()
    with xr.open_dataset(args.predictions) as cube:
        dates = [date.fromisoformat(str(value)[:10]) for value in np.asarray(cube.time.values, dtype="datetime64[D]")]
        result = index_argo_profiles(
            args.profile_root,
            args.output,
            candidate_dates=dates,
            max_time_difference_days=args.max_time_difference_days,
            lat_min=float(cube.latitude.min()),
            lat_max=float(cube.latitude.max()),
            lon_min=float(cube.longitude.min()),
            lon_max=float(cube.longitude.max()),
        )
        print({"prediction_label": cube.attrs.get("label"), "prediction_days": len(dates), **result})


if __name__ == "__main__":
    main()
