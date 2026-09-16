#!/usr/bin/env python3
"""Run the labelled OceanEmbed 30-day training smoke test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.train import run_smoke_test  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train OceanEmbed on the data-limited PoC only.")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    result = run_smoke_test(epochs=args.epochs)
    print("DEMO / DATA-LIMITED")
    print("Artifacts:", result["artifacts"])
    print("Best checkpoint:", result["best_checkpoint"])
    print("Final history entry:", result["history"][-1])


if __name__ == "__main__":
    main()
