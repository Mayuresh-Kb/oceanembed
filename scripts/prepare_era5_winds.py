#!/usr/bin/env python3
"""Extract and merge a CDS ERA5 wind ZIP archive into one NetCDF file.

Example:
  .venv/bin/python scripts/prepare_era5_winds.py \\
    data/era5_winds_nio_20240101_20240130.nc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocessing.era5 import extract_era5_archive, merge_wind_netcdfs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Safely prepare a CDS ERA5 wind archive.")
    parser.add_argument("archive", type=Path, help="ZIP archive returned by CDS (extension may be .nc)")
    parser.add_argument("--raw-dir", type=Path, default=None, help="Directory for extracted provider files")
    parser.add_argument("--output", type=Path, default=None, help="Merged NetCDF output path")
    args = parser.parse_args()

    stem = args.archive.stem
    raw_dir = args.raw_dir or args.archive.parent / f"{stem}_raw"
    output = args.output or args.archive.parent / f"{stem}_merged.nc"
    files = extract_era5_archive(args.archive, raw_dir)
    merged = merge_wind_netcdfs(files, output)
    print(f"\nMerged ERA5 winds written to: {output}")
    print(f"Variables: {list(merged.data_vars)}")
    print(f"Dimensions: {dict(merged.sizes)}")
    merged.close()


if __name__ == "__main__":
    main()
