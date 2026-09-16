#!/usr/bin/env python3
"""Create an integrity manifest for the small dashboard artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from security.integrity import write_manifest  # noqa: E402


def main() -> None:
    processed = ROOT / "data" / "processed"
    output = processed / "dashboard_artifact_manifest.json"
    manifest = write_manifest(
        [
            processed / "smoke_test" / "validation_predictions.nc",
            processed / "argo_validation" / "argo_validation_metrics.json",
            processed / "argo_validation" / "argo_collocations.csv",
        ],
        output,
    )
    print("Wrote:", manifest)


if __name__ == "__main__":
    main()
