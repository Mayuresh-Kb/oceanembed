#!/usr/bin/env python3
"""Print dimensions, coordinates, and variables for a NetCDF file.

Usage (from the repo root):
  .venv/bin/python scripts/inspect_netcdf.py data/your_file.nc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocessing.config import load_config  # noqa: E402
from preprocessing.loader import open_and_inspect, resolve_variables  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect an ocean NetCDF file.")
    parser.add_argument("path", type=Path, help="Path to a .nc file")
    args = parser.parse_args()
    dataset, _ = open_and_inspect(args.path, print_report=True)
    try:
        mapping = resolve_variables(dataset, load_config(), print_mapping=True)
        print("\nResolved canonical mapping:")
        for canonical, result in mapping.items():
            if result.found:
                print(f"  {canonical} <- {result.source_name}")
    except KeyError as error:
        # A single-source input (for example an ERA5 wind-only file) is still
        # scientifically useful and must be inspectable on its own.
        print(f"\nFull surface bundle not present in this file: {error}")
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
