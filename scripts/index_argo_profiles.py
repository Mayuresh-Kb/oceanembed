#!/usr/bin/env python3
"""Index ARGO profile metadata near the exported OceanEmbed prediction dates."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from validation.argo_validation import index_argo_profiles  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Find ARGO profiles near OceanEmbed's held-out prediction dates.")
    parser.add_argument("profile_root", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "processed" / "argo_candidate_profiles.csv")
    parser.add_argument("--max-time-difference-days", type=int, default=3)
    args = parser.parse_args()
    prediction_start = date(2024, 1, 26)
    prediction_dates = [prediction_start + timedelta(days=offset) for offset in range(5)]
    result = index_argo_profiles(args.profile_root, args.output, candidate_dates=prediction_dates, max_time_difference_days=args.max_time_difference_days)
    print("Independent ARGO candidate index; ARGO is not used in training.")
    print("Output:", args.output)
    print(result)


if __name__ == "__main__":
    main()
