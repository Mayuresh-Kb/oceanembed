#!/usr/bin/env python3
"""Run separate ARGO validation against saved OceanEmbed predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from validation.argo_validation import run_independent_argo_validation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ARGO-only independent validation; it never trains a model.")
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, default=ROOT / "data" / "processed" / "argo_candidate_profiles.csv")
    parser.add_argument("--predictions", type=Path, default=ROOT / "data" / "processed" / "smoke_test" / "validation_predictions.nc")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed" / "argo_validation")
    args = parser.parse_args()
    print(json.dumps(run_independent_argo_validation(candidate_csv=args.candidates, profile_root=args.profile_root, prediction_path=args.predictions, output_dir=args.output_dir), indent=2))


if __name__ == "__main__":
    main()
