#!/usr/bin/env python3
"""Concatenate already prepared ERA5 U/V NetCDF files along time."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocessing.era5 import concatenate_monthly_winds  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Concatenate prepared ERA5 wind files after coordinate validation.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = concatenate_monthly_winds(args.inputs, args.output)
    print("Combined winds:", args.output)
    print("Dimensions:", dict(dataset.sizes))
    dataset.close()


if __name__ == "__main__":
    main()
