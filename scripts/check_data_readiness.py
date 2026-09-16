#!/usr/bin/env python3
"""Report whether local NetCDF files satisfy the OceanEmbed PoC data contract.

Example:
  .venv/bin/python scripts/check_data_readiness.py \\
    --glorys data/glorys_*.nc --winds data/era5_winds_*.nc --argo data/argo_*.nc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocessing.readiness import assess_readiness  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Check OceanEmbed local-data readiness.")
    parser.add_argument("--glorys", nargs="+", required=True, type=Path)
    parser.add_argument("--winds", nargs="*", default=[], type=Path)
    parser.add_argument("--argo", nargs="*", default=[], type=Path)
    args = parser.parse_args()
    report = assess_readiness(
        glorys_files=args.glorys,
        wind_files=args.winds,
        argo_files=args.argo,
    )
    print(report.format())
    raise SystemExit(0 if report.validation_ready else 1)


if __name__ == "__main__":
    main()
