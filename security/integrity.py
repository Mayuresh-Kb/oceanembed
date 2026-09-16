"""Dataset/artifact integrity manifest helpers for the local prototype."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file incrementally without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(paths: list[str | Path], output_path: str | Path) -> Path:
    """Write hashes and byte sizes for dashboard artifacts."""
    records = []
    for item in paths:
        path = Path(item)
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"algorithm": "SHA-256", "artifacts": records}, indent=2), encoding="utf-8")
    return output
