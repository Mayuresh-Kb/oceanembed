#!/usr/bin/env python3
"""Prepare monthly ERA5 archives and concatenate the 150-day extension."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocessing.era5 import concatenate_monthly_winds, extract_era5_archive, merge_wind_netcdfs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Safely prepare and combine monthly ERA5 wind archives.")
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "era5_winds_nio_20240201_20240629_merged.nc")
    args = parser.parse_args()
    monthly = []
    for archive in args.archives:
        raw_dir = archive.parent / f"{archive.stem}_raw"
        merged = archive.parent / f"{archive.stem}_merged.nc"
        monthly.append(merge_wind_netcdfs(extract_era5_archive(archive, raw_dir), merged))
        monthly[-1].close()
    combined = concatenate_monthly_winds([path.parent / f"{path.stem}_merged.nc" for path in args.archives], args.output)
    print("Combined winds:", args.output)
    print("Dimensions:", dict(combined.sizes))
    combined.close()


if __name__ == "__main__":
    main()
