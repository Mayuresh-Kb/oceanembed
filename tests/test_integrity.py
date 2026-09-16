from __future__ import annotations

import json

from security.integrity import sha256_file, write_manifest


def test_manifest_contains_deterministic_sha256(tmp_path) -> None:
    source = tmp_path / "artifact.txt"
    source.write_text("oceanembed", encoding="utf-8")
    manifest = write_manifest([source], tmp_path / "manifest.json")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["algorithm"] == "SHA-256"
    assert payload["artifacts"][0]["sha256"] == sha256_file(source)
