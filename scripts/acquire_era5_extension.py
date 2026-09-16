#!/usr/bin/env python3
"""Download one exact ERA5 daily-wind request per month for the 180-day extension."""

from __future__ import annotations

import argparse
import calendar
from copy import deepcopy
from datetime import date
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocessing.acquisition import build_era5_request, execute_era5_download, load_acquisition_config  # noqa: E402


def monthly_ranges(start: date, end: date) -> list[tuple[date, date]]:
    ranges: list[tuple[date, date]] = []
    current = date(start.year, start.month, 1)
    while current <= end:
        last = date(current.year, current.month, calendar.monthrange(current.year, current.month)[1])
        ranges.append((max(start, current), min(end, last)))
        current = date(current.year + (current.month == 12), 1 if current.month == 12 else current.month + 1, 1)
    return ranges


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ERA5 daily winds as explicit one-month requests.")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2024, 2, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2024, 6, 29))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    base = load_acquisition_config()
    for start, end in monthly_ranges(args.start, args.end):
        acquisition = deepcopy(base)
        acquisition["period"] = {"start": start.isoformat(), "end": end.isoformat()}
        dataset_id, request = build_era5_request(acquisition)
        output = args.output_dir / f"era5_winds_nio_{start:%Y%m%d}_{end:%Y%m%d}.nc"
        print(f"{start} to {end}: {output.name}")
        if args.execute:
            execute_era5_download(dataset_id, request, output_path=output)
    print("Completed monthly ERA5 requests." if args.execute else "Preview only. Add --execute to download.")


if __name__ == "__main__":
    main()
