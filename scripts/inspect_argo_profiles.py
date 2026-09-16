#!/usr/bin/env python3
"""Inspect an ARGO profile NetCDF before independent validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from validation.argo_validation import inspect_argo_profile_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect, but do not alter, an ARGO profile NetCDF.")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect_argo_profile_file(args.path), indent=2))


if __name__ == "__main__":
    main()
