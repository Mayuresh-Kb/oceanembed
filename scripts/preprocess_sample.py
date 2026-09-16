#!/usr/bin/env python3
"""Run the preprocessing pipeline on one NetCDF file and print tensor shapes.

Usage (from the repo root):
  .venv/bin/python scripts/preprocess_sample.py data/your_file.nc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocessing.preprocessing import preprocess_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess one OceanEmbed NetCDF.")
    parser.add_argument("path", type=Path, help="Path to a .nc file")
    args = parser.parse_args()

    result = preprocess_file(str(args.path), print_report=True)
    print("\n===== PREPROCESS NOTES =====")
    for note in result.notes:
        print(f"- {note}")
    print("channel_names:", result.channel_names)
    print("channels shape (T, C, H, W):", result.channels.shape)
    print("ocean_mask True fraction:", float(result.ocean_mask.mean()))
    print("normalization stats:")
    for name, stats in result.norm_stats.items():
        print(f"  {name}: mean={stats['mean']:.4f} std={stats['std']:.4f}")
    result.dataset.close()


if __name__ == "__main__":
    main()
