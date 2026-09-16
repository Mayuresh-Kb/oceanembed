#!/usr/bin/env python3
"""Train the bounded-memory regional-patch OceanEmbed experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.train import run_smoke_test  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 150-day regional-patch OceanEmbed experiment.")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    result = run_smoke_test(
        epochs=args.epochs,
        training_key="hackathon_patch_training",
        glorys_paths=ROOT / "data" / "glorys_nio_20240201_20240629_z1200m.nc",
        wind_paths=ROOT / "data" / "era5_winds_nio_20240201_20240629_merged.nc",
        output_dir=ROOT / "data" / "processed" / "hackathon_patch_150d",
        config_path=ROOT / "configs" / "hackathon_patch_150d.yaml",
    )
    print(result["label"])
    print("Artifacts:", result["artifacts"])
    print("Final history entry:", result["history"][-1])


if __name__ == "__main__":
    main()
